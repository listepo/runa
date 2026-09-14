//! `runa` — a single command-line binary that runs AI models locally
//! (GGUF via ggml/llama.cpp) or through the OpenAI and Anthropic APIs
//! (plan §1).
//!
//! P2.3 delivers `run` (one-shot) and `chat` (REPL); `fit` → P1.11,
//! `pull` → P2.4, `auto`/`on_unfit` → P2.5, `serve` → P3.9, `bench` → P2.10,
//! cloud backends → P3. Model refs in P2.3 are local files; `hf:`/aliases
//! need `runa pull` (P2.4) and error with a pointer instead of a download.

use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use runa_core::{ThinkConfig, ThinkOverrides, parse_budget};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, KvKind, LoadConfig, Mode, Placement, PromptCache,
    SamplingConfig, StopReason, Usage, VisionFrame, VisionSource, load, parse_device_list,
    parse_tensor_split, planner_kv_type,
};
use runa_fit::{
    Descriptor, FitConfig, HwSpec, PlannerConfig, Reader, check_fit, estimate_compute, estimate_kv,
    read_local_prefix,
};
use runa_memory::{ClaimError, FakeBackend, MemoryManager, TaskRegistry};

mod bench;
mod cloud;
mod config;
mod pull;
mod serve;

/// Run AI models locally or through the OpenAI and Anthropic APIs.
#[derive(Debug, Parser)]
#[command(name = "runa", version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
// CLI subcommands are inherently size-diverse and parsed once per process;
// boxing the large variants would ripple through every clap match site.
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// One-shot generation: `runa run <model> [prompt]`.
    Run(RunArgs),
    /// Interactive chat (history, `/think`, `/mode`, `/model`, `\` continuation).
    Chat {
        /// Local GGUF file.
        model: Option<String>,
        /// Compute mode (default: cpu).
        #[arg(long, default_value = "cpu")]
        mode: String,
        /// Context length.
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// Thinking: on | off (P3.1).
        #[arg(long, value_name = "on|off")]
        think: Option<String>,
        /// Reasoning token budget.
        #[arg(long, value_name = "N")]
        think_budget: Option<u32>,
        /// Effort level: low | medium | high | max.
        #[arg(long, value_name = "LEVEL")]
        effort: Option<String>,
        /// Print reasoning.
        #[arg(long, default_value_t = false)]
        show_reasoning: bool,
        /// Hide reasoning even if config enables it.
        #[arg(long, default_value_t = false)]
        no_show_reasoning: bool,
    },
    /// Download a model: `runa pull hf:<repo>:<file-or-quant>` (P2.4).
    Pull {
        /// Model reference (`hf:<repo>:<file.gguf>` or `hf:<repo>:<quant>`).
        model: String,
    },
    /// List downloaded models + configured aliases (P2.4).
    Models {},
    /// Prefill/decode throughput like llama-bench (pp512/tg128).
    Bench {
        /// Local GGUF file.
        model: String,
        /// Compute mode: cpu | gpu | hybrid | auto (default: gpu).
        #[arg(long, default_value = "gpu")]
        mode: String,
        /// Context length (must fit pp + tg + 1).
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// Prompt-processing tokens (llama-bench `-p`).
        #[arg(long, default_value_t = 512)]
        pp: u32,
        /// Generated tokens (llama-bench `-n`).
        #[arg(long, default_value_t = 128)]
        tg: u32,
        /// Emit one JSON object instead of the human summary.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Do not append this run to the calibration DB.
        #[arg(long, default_value_t = false)]
        no_calibrate: bool,
        /// KV cache type for both K and V (`f16`, `q8_0`, `q4_0`).
        #[arg(long, value_name = "TYPE")]
        kv: Option<String>,
        /// KV type for K only (overrides `--kv`).
        #[arg(long, value_name = "TYPE")]
        kv_k: Option<String>,
        /// KV type for V only (overrides `--kv`).
        #[arg(long, value_name = "TYPE")]
        kv_v: Option<String>,
        /// ggml backends to use (`0,1` or `CUDA0,CUDA1`).
        #[arg(long, value_name = "LIST")]
        device: Option<String>,
        /// Per-GPU proportions (`3,1`). Requires multiple GPUs.
        #[arg(long, value_name = "LIST")]
        tensor_split: Option<String>,
    },
    /// OpenAI-compatible HTTP server (P3.9 / P6.1).
    Serve {
        /// Local GGUF file to load (single-model shorthand).
        model: Option<String>,
        /// Additional GGUF paths (comma-separated or repeatable).
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Max in-flight HTTP generations (queued beyond this).
        #[arg(long, default_value_t = 1)]
        parallel: usize,
        /// Max models kept loaded (LRU unloads the rest).
        #[arg(long)]
        max_loaded: Option<usize>,
        /// Bind address (D11: 127.0.0.1 by default).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Bind port. `0` picks an ephemeral port.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Compute mode: cpu | gpu | hybrid | auto.
        #[arg(long, default_value = "cpu")]
        mode: String,
        /// Context length.
        #[arg(long, default_value_t = 4096)]
        ctx: u32,
    },
    /// Report compiled-in backends and native-build flags (P6.3).
    Doctor {
        /// Emit machine-readable JSON (`backends`, `native_build`).
        #[arg(long)]
        json: bool,
    },
    /// Decode / inspect media files (P4.1).
    Media {
        #[command(subcommand)]
        action: MediaAction,
    },
    /// Plan task claims (`docs/tasks.md`, P7.4).
    Tasks {
        #[command(subcommand)]
        action: TaskAction,
    },
}

#[derive(Debug, Subcommand)]
enum MediaAction {
    /// Decode audio to f32 mono 16 kHz and print codec / PCM hash.
    Probe {
        /// Audio file (wav, flac, ogg, mp3, aac).
        path: PathBuf,
        /// Machine-readable JSON (`AudioProbe`).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Sample video frames (P4.5): fps, max-frames, scene-change.
    Video {
        path: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        fps: f32,
        #[arg(long, default_value_t = 32)]
        max_frames: usize,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Transcribe audio via whisper.cpp (P4.2). Auto-pulls ggml-base / turbo.
    Transcribe {
        /// Audio file (wav, flac, ogg, mp3, aac).
        path: PathBuf,
        /// `base` or `large-v3-turbo`.
        #[arg(long, default_value = "base")]
        model: String,
        /// ISO-639-1 or `auto`.
        #[arg(long, default_value = "auto")]
        lang: String,
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Do not download missing ggml files.
        #[arg(long, default_value_t = false)]
        no_pull: bool,
    },
}

#[derive(Debug, Subcommand)]
enum TaskAction {
    /// List `free` task IDs.
    List {},
    /// Claim a task: `runa tasks claim P7.1 --agent <name>`.
    Claim {
        task_id: String,
        #[arg(long)]
        agent: String,
    },
    /// Release a claim.
    Release {
        task_id: String,
        #[arg(long)]
        agent: String,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = config::reject_inline_in_config_files() {
        eprintln!("runa: error: {e}");
        std::process::exit(2);
    }
    let rc = match cli.command {
        Commands::Run(args) => cmd_run(&args),
        Commands::Chat {
            model,
            mode,
            ctx,
            think,
            think_budget,
            effort,
            show_reasoning,
            no_show_reasoning,
        } => cmd_chat(
            model.as_deref(),
            &mode,
            ctx,
            think,
            think_budget,
            effort,
            show_reasoning,
            no_show_reasoning,
        ),
        Commands::Pull { model } => cmd_pull(&model),
        Commands::Models {} => cmd_models(),
        Commands::Bench {
            model,
            mode,
            ctx,
            pp,
            tg,
            json,
            no_calibrate,
            kv,
            kv_k,
            kv_v,
            device,
            tensor_split,
        } => match resolve_kv(kv.as_deref(), kv_k.as_deref(), kv_v.as_deref()) {
            Ok((kv_k, kv_v)) => bench::cmd_bench(
                &model,
                &mode,
                ctx,
                pp,
                tg,
                json,
                no_calibrate,
                kv_k,
                kv_v,
                device.as_deref(),
                tensor_split.as_deref(),
            ),
            Err(e) => Err(e),
        },
        Commands::Media { action } => match action {
            MediaAction::Probe { path, json } => cmd_media_probe(&path, json),
            MediaAction::Video {
                path,
                fps,
                max_frames,
                json,
            } => cmd_media_video(&path, fps, max_frames, json),
            MediaAction::Transcribe {
                path,
                model,
                lang,
                json,
                no_pull,
            } => cmd_media_transcribe(&path, &model, &lang, json, no_pull),
        },
        Commands::Serve {
            model,
            models,
            parallel,
            max_loaded,
            host,
            port,
            mode,
            ctx,
        } => {
            let mut paths = Vec::new();
            if let Some(m) = model {
                paths.push(std::path::PathBuf::from(m));
            }
            for entry in models {
                for part in entry.split(',') {
                    let p = part.trim();
                    if !p.is_empty() {
                        paths.push(std::path::PathBuf::from(p));
                    }
                }
            }
            serve::model_specs_from_paths(paths).and_then(|models| {
                serve::cmd_serve(serve::ServeOpts {
                    models,
                    host,
                    port,
                    mode,
                    ctx,
                    parallel,
                    max_loaded,
                })
            })
        }
        Commands::Doctor { json } => {
            doctor(json);
            Ok(())
        }
        Commands::Tasks { action } => cmd_tasks(action),
    };
    if let Err(e) = rc {
        let (code, msg) = if let Some(msg) = e.strip_prefix("unfit: ") {
            (2, msg)
        } else {
            (1, e.as_str())
        };
        eprintln!("runa: error: {msg}");
        std::process::exit(code);
    }
}

#[derive(Debug, clap::Args)]
struct RunArgs {
    /// Local GGUF file (`hf:` refs need `runa pull`, P2.4).
    model: String,
    /// Prompt text (else read from stdin when piped).
    prompt: Option<String>,
    /// Compute mode: cpu | gpu | hybrid | auto (default: auto).
    #[arg(long, default_value = "auto")]
    mode: String,
    /// Context length.
    #[arg(long, default_value_t = 8192)]
    ctx: u32,
    /// Max new tokens.
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    /// Sampling temperature (<= 0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    /// Sampler seed.
    #[arg(long, default_value_t = 42)]
    seed: u32,
    /// Emit one JSON object instead of streaming text.
    #[arg(long, default_value_t = false)]
    json: bool,
    /// When auto cannot fit: error | cpu | cloud:<backend>:<model> (D12).
    #[arg(long)]
    on_unfit: Option<String>,
    /// Keep MoE expert tensors of the first N layers on CPU (`--n-cpu-moe`).
    #[arg(long, value_name = "N")]
    n_cpu_moe: Option<u32>,
    /// LMDB dir for prompt KV cache (default: ~/.cache/runa/kv).
    #[arg(long, value_name = "DIR")]
    prompt_cache: Option<PathBuf>,
    /// Disable the LMDB prompt cache (P2.8).
    #[arg(long, default_value_t = false)]
    no_prompt_cache: bool,
    /// KV cache type for both K and V (`f16`, `q8_0`, `q4_0`). Requires flash-attn.
    #[arg(long, value_name = "TYPE")]
    kv: Option<String>,
    /// KV type for K only (overrides `--kv`).
    #[arg(long, value_name = "TYPE")]
    kv_k: Option<String>,
    /// KV type for V only (overrides `--kv`).
    #[arg(long, value_name = "TYPE")]
    kv_v: Option<String>,
    /// Thinking: on | off (P3.1). Budget/effort override `on`.
    #[arg(long, value_name = "on|off")]
    think: Option<String>,
    /// Reasoning token budget (`ThinkMode::Budget`).
    #[arg(long, value_name = "N")]
    think_budget: Option<u32>,
    /// Effort level: low | medium | high | max.
    #[arg(long, value_name = "LEVEL")]
    effort: Option<String>,
    /// Print reasoning (`Event::Reasoning` lands in P3.2).
    #[arg(long, default_value_t = false)]
    show_reasoning: bool,
    /// Hide reasoning even if config `[think] show = true`.
    #[arg(long, default_value_t = false)]
    no_show_reasoning: bool,
    /// ggml backends to use (`0,1` or `CUDA0,CUDA1`).
    #[arg(long, value_name = "LIST")]
    device: Option<String>,
    /// Per-GPU proportions (`3,1`). Requires multiple GPUs.
    #[arg(long, value_name = "LIST")]
    tensor_split: Option<String>,
    /// Audio file (PCM 16 kHz). Routed by `--audio-route`.
    #[arg(long, value_name = "PATH")]
    audio: Option<PathBuf>,
    /// Audio/vision mmproj GGUF. Sibling `*mmproj*.gguf` if `--audio` and omitted.
    #[arg(long, value_name = "PATH")]
    mmproj: Option<PathBuf>,
    /// Audio route: auto | native | asr (P4.4).
    #[arg(long, value_name = "auto|native|asr")]
    audio_route: Option<String>,
    /// Image file(s) for native mtmd vision (P4.6).
    #[arg(long, value_name = "PATH")]
    image: Vec<PathBuf>,
    /// Video file: sampled frames with `[t=12.0s]` markers (P4.6).
    #[arg(long, value_name = "PATH")]
    video: Option<PathBuf>,
    /// Trigram speculative decoding; greedy-verified (use --temperature 0).
    #[arg(long, default_value_t = false)]
    ngram: bool,
    /// Draft-model GGUF: counted in fit; speculation still uses n-gram.
    #[arg(long, value_name = "PATH")]
    draft: Option<PathBuf>,
    /// Constrain the answer to a JSON Schema (file path or inline JSON).
    #[arg(long, value_name = "FILE|JSON", conflicts_with = "grammar")]
    json_schema: Option<String>,
    /// Constrain the answer to a GBNF grammar file.
    #[arg(long, value_name = "FILE")]
    grammar: Option<PathBuf>,
}

/// Resolve a model reference: local path → alias → pull store (P2.4).
/// Never downloads (that is `pull`'s job — plan D12, no silent fetches).
pub(crate) fn resolve_model(model_ref: &str) -> Result<PathBuf, String> {
    let aliases = config::load_aliases()?;
    pull::find_local(model_ref, &aliases)
}

fn parse_mode(mode: &str) -> Result<Mode, String> {
    Mode::parse(mode).ok_or_else(|| format!("{mode}: --mode must be cpu | gpu | hybrid"))
}

pub(crate) enum ModeChoice {
    Auto,
    Fixed(Mode),
}

pub(crate) fn parse_mode_choice(mode: &str) -> Result<ModeChoice, String> {
    if mode.eq_ignore_ascii_case("auto") {
        return Ok(ModeChoice::Auto);
    }
    Mode::parse(mode)
        .map(ModeChoice::Fixed)
        .ok_or_else(|| format!("{mode}: --mode must be cpu | gpu | hybrid | auto"))
}

pub(crate) fn vram_bytes() -> Result<(u64, &'static str), String> {
    if let Ok(s) = std::env::var("RUNA_FAKE_VRAM") {
        let mib: u64 = s
            .parse()
            .map_err(|_| format!("RUNA_FAKE_VRAM={s}: expected integer MiB"))?;
        return Ok((mib.saturating_mul(1024 * 1024), "RUNA_FAKE_VRAM"));
    }
    #[cfg(target_os = "macos")]
    {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        // Unified memory stand-in until the P1.6 wired_limit probe is wired here.
        let assumed = sys.total_memory().saturating_mul(3) / 4;
        Ok((assumed, "macos-unified-75pct"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok((0, "unknown"))
    }
}

pub(crate) fn ram_bytes() -> u64 {
    // Test hook like RUNA_FAKE_VRAM: forces a CPU NO FIT on big hosts.
    if let Some(mib) = std::env::var("RUNA_FAKE_RAM")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        return mib.saturating_mul(1024 * 1024);
    }
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory()
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes.div_ceil(1024 * 1024)
}

fn memory_ceiling_mib() -> u64 {
    if let Ok(s) = std::env::var("RUNA_MEMORY_CEILING_MIB")
        && let Ok(n) = s.parse::<u64>()
    {
        return n;
    }
    ram_bytes() / (1024 * 1024)
}

/// P7.3: refuse a run whose KV+compute demand exceeds the fit ceiling
/// and `max_growth_mib` before llama allocates arenas.
pub(crate) fn preflight_grow(
    path: &Path,
    ctx: u32,
    kv_type: &str,
    extra_bytes: u64,
) -> Result<(), String> {
    let policy = config::resolve_memory_policy()?;
    let header = read_local_prefix(path).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let kv = estimate_kv(&desc, u64::from(ctx), kv_type);
    let compute = estimate_compute(&desc, 512);
    let demand = bytes_to_mib(
        kv.kv_bytes
            .saturating_add(compute.compute_bytes)
            .saturating_add(extra_bytes),
    );
    if demand == 0 {
        return Ok(());
    }
    let mm = MemoryManager::new(
        policy,
        memory_ceiling_mib(),
        Box::new(FakeBackend::new(0, 0)),
    );
    mm.grow_for(demand).map_err(|e| format!("unfit: {e}"))
}

fn cmd_media_probe(path: &Path, json: bool) -> Result<(), String> {
    let decoded = runa_media::decode_audio(path).map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&decoded.probe).map_err(|e| e.to_string())?
        );
    } else {
        println!("{}", decoded.probe);
    }
    Ok(())
}

fn cmd_media_video(path: &Path, fps: f32, max_frames: usize, json: bool) -> Result<(), String> {
    let opts = runa_media::VideoOpts {
        fps,
        max_frames,
        ..runa_media::VideoOpts::default()
    };
    let sampled = runa_media::sample_video(path, &opts).map_err(|e| e.to_string())?;
    if json {
        let times: Vec<f32> = sampled.frames.iter().map(|f| f.t_sec).collect();
        println!(
            "{}",
            serde_json::json!({
                "n_frames": sampled.frames.len(),
                "t_sec": times,
                "audio": sampled.audio.is_some(),
            })
        );
    } else {
        println!("frames: {}", sampled.frames.len());
        for f in &sampled.frames {
            println!("  [t={:.1}s] {}x{}", f.t_sec, f.width, f.height);
        }
        if sampled.audio.is_some() {
            println!("audio: yes");
        }
    }
    Ok(())
}

fn cmd_media_transcribe(
    path: &Path,
    model: &str,
    lang: &str,
    json: bool,
    no_pull: bool,
) -> Result<(), String> {
    let kind = runa_media::WhisperKind::parse(model).map_err(|e| e.to_string())?;
    let opts = runa_media::AsrOptions {
        kind,
        pull: !no_pull,
        language: match lang {
            "auto" | "" => None,
            other => Some(other.to_string()),
        },
        ..runa_media::AsrOptions::default()
    };
    let t = runa_media::transcribe_file(path, &opts).map_err(|e| e.to_string())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&t).map_err(|e| e.to_string())?
        );
    } else {
        if let Some(lang) = &t.language {
            eprintln!("language: {lang}");
        }
        println!("{}", t.text);
    }
    Ok(())
}

fn auto_verdict_line(report: &runa_fit::FitReport, vram: u64, src: &str) -> String {
    let gpu_mib = report.plan.gpu_total_bytes as f64 / 1_048_576.0;
    let vram_mib = vram / 1_048_576;
    let speed = report
        .speed_gpu
        .as_ref()
        .or(report.speed_cpu.as_ref())
        .map(|s| format!(" · {:.1} tok/s decode", s.decode_toks_per_sec))
        .unwrap_or_default();
    format!(
        "runa auto {} · GPU {gpu_mib:.1} MiB · VRAM {vram_mib} MiB ({src}){speed}",
        report.verdict
    )
}

fn placement_from_report(report: &runa_fit::FitReport) -> Placement {
    match &report.verdict {
        runa_fit::Verdict::Gpu => Placement::gpu(),
        runa_fit::Verdict::Cpu => Placement::cpu(),
        runa_fit::Verdict::Hybrid {
            gpu_layers,
            total_layers,
        } => {
            if *gpu_layers >= *total_layers && *total_layers > 0 {
                Placement::hybrid_moe()
            } else {
                Placement {
                    n_gpu_layers: *gpu_layers as u32,
                    cpu_patterns: Vec::new(),
                    main_gpu: 0,
                    devices: Vec::new(),
                    tensor_split: Vec::new(),
                }
            }
        }
        runa_fit::Verdict::NoFit => Placement::cpu(),
    }
}

pub(crate) enum AutoPlacement {
    Local(Placement),
    Cloud(runa_cloud::CloudRef),
}

pub(crate) fn auto_placement(
    path: &Path,
    ctx: u32,
    on_unfit: &config::OnUnfit,
    kv_type: &str,
    mmproj: Option<&Path>,
    draft: Option<&Path>,
) -> Result<AutoPlacement, String> {
    let header = read_local_prefix(path).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let (vram, vram_src) = vram_bytes()?;
    let mmproj_path = mmproj
        .map(Path::to_path_buf)
        .or_else(|| runa_fit::sibling_mmproj(path));
    let mmproj_bytes = mmproj_path
        .as_ref()
        .map(|p| runa_fit::mmproj_file_bytes(p))
        .unwrap_or(0);
    let draft_bytes = draft.map(runa_fit::mmproj_file_bytes).unwrap_or(0);
    let planner = PlannerConfig {
        vram_bytes: vram,
        ram_bytes: ram_bytes(),
        ctx_len: u64::from(ctx),
        kv_type: kv_type.to_owned(),
        mmproj_bytes: mmproj_bytes.saturating_add(draft_bytes),
        ..PlannerConfig::default()
    };
    let gpu_hw = if vram > 0 {
        Some(if cfg!(target_os = "macos") {
            HwSpec::metal()
        } else {
            HwSpec::cuda()
        })
    } else {
        None
    };
    let report = check_fit(
        &desc,
        &FitConfig {
            planner,
            gpu_hw,
            cpu_hw: HwSpec::cpu(),
            has_mmproj: false,
            media: runa_fit::MediaFit::default(),
        },
    );
    let line = auto_verdict_line(&report, vram, vram_src);
    match &report.verdict {
        runa_fit::Verdict::NoFit => match on_unfit {
            config::OnUnfit::Cpu => {
                eprintln!("{line} · on_unfit=cpu → CPU");
                Ok(AutoPlacement::Local(Placement::cpu()))
            }
            config::OnUnfit::Error => Err(format!("unfit: {line} · on_unfit=error")),
            config::OnUnfit::Cloud(spec) => {
                let cloud = cloud::cloud_from_on_unfit(spec)?;
                eprintln!("{line} · on_unfit=cloud:{spec}");
                Ok(AutoPlacement::Cloud(cloud))
            }
        },
        _ => {
            eprintln!("{line}");
            Ok(AutoPlacement::Local(placement_from_report(&report)))
        }
    }
}

/// Prompt from argv, else from piped stdin.
fn read_prompt(arg: Option<String>) -> Result<String, String> {
    if let Some(p) = arg {
        if p == "-" {
            return read_stdin();
        }
        return Ok(p);
    }
    if io::stdin().is_terminal() {
        return Err("no prompt: pass one as an argument or pipe it on stdin".into());
    }
    read_stdin()
}

fn read_stdin() -> Result<String, String> {
    let mut s = String::new();
    io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("stdin: {e}"))?;
    if s.trim().is_empty() {
        return Err("empty prompt on stdin".into());
    }
    Ok(s)
}

/// `--json-schema`: inline JSON when it starts with `{`, else a file path.
/// Returned verbatim after a parse check.
fn read_json_schema(arg: &str) -> Result<String, String> {
    let text = if arg.trim_start().starts_with('{') {
        arg.to_owned()
    } else {
        fs::read_to_string(arg).map_err(|e| format!("--json-schema {arg}: {e}"))?
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .map_err(|e| format!("--json-schema is not valid JSON: {e}"))?;
    Ok(text)
}

fn default_prompt_cache_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RUNA_PROMPT_CACHE")
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("XDG_CACHE_HOME")
        && !p.is_empty()
    {
        return PathBuf::from(p).join("runa").join("kv");
    }
    match std::env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".cache").join("runa").join("kv"),
        None => std::env::temp_dir().join("runa-kv"),
    }
}

fn attach_prompt_cache(
    loaded: &mut runa_engine::LoadedModel,
    disabled: bool,
    explicit: Option<&Path>,
) -> Result<(), String> {
    if disabled || std::env::var_os("RUNA_NO_PROMPT_CACHE").is_some() {
        return Ok(());
    }
    let (dir, required) = match explicit {
        Some(p) => (p.to_path_buf(), true),
        None => (default_prompt_cache_dir(), false),
    };
    match PromptCache::open(&dir) {
        Ok(cache) => {
            loaded.attach_prompt_cache(cache);
            Ok(())
        }
        Err(e) if required => Err(format!("prompt-cache: {e}")),
        Err(e) => {
            eprintln!("prompt-cache: disabled ({e})");
            Ok(())
        }
    }
}

fn parse_kv_kind(raw: &str) -> Result<KvKind, String> {
    raw.parse::<KvKind>().map_err(|e| format!("--kv: {e}"))
}

fn resolve_kv(
    kv: Option<&str>,
    kv_k: Option<&str>,
    kv_v: Option<&str>,
) -> Result<(Option<KvKind>, Option<KvKind>), String> {
    let both = kv.map(parse_kv_kind).transpose()?;
    let k = match kv_k {
        Some(s) => Some(parse_kv_kind(s)?),
        None => both,
    };
    let v = match kv_v {
        Some(s) => Some(parse_kv_kind(s)?),
        None => both,
    };
    Ok((k, v))
}

fn cli_think(
    think: Option<&str>,
    think_budget: Option<u32>,
    effort: Option<&str>,
    show_reasoning: bool,
    no_show_reasoning: bool,
) -> Result<ThinkConfig, String> {
    let mut o = ThinkOverrides::default();
    if let Some(s) = think {
        o.think = Some(ThinkOverrides::parse_think(s)?);
    }
    if let Some(n) = think_budget {
        o.budget = Some(parse_budget(&n.to_string())?);
    }
    if let Some(s) = effort {
        o.effort = Some(runa_core::Effort::parse(s)?);
    }
    if no_show_reasoning {
        o.show = Some(false);
    } else if show_reasoning {
        o.show = Some(true);
    }
    config::resolve_think(o)
}

fn collect_vision_frames(
    images: &[PathBuf],
    video: Option<&Path>,
) -> Result<Vec<VisionFrame>, String> {
    let mut out = Vec::new();
    for p in images {
        out.push(VisionFrame {
            t_sec: None,
            source: VisionSource::Path(p.clone()),
        });
    }
    if let Some(v) = video {
        let sampled = runa_media::sample_video(v, &runa_media::VideoOpts::default())
            .map_err(|e| e.to_string())?;
        for f in sampled.frames {
            out.push(VisionFrame {
                t_sec: Some(f.t_sec),
                source: VisionSource::Rgb {
                    width: f.width,
                    height: f.height,
                    rgb: f.rgb,
                },
            });
        }
    }
    Ok(out)
}

fn cmd_run(args: &RunArgs) -> Result<(), String> {
    let think = cli_think(
        args.think.as_deref(),
        args.think_budget,
        args.effort.as_deref(),
        args.show_reasoning,
        args.no_show_reasoning,
    )?;
    let mut prompt = read_prompt(args.prompt.clone())?;
    let audio_pref = config::resolve_audio_route(args.audio_route.as_deref())?;
    let json_schema = args
        .json_schema
        .as_deref()
        .map(read_json_schema)
        .transpose()?;
    let grammar = args
        .grammar
        .as_ref()
        .map(|p| fs::read_to_string(p).map_err(|e| format!("--grammar {}: {e}", p.display())))
        .transpose()?;
    let cloud_run = |prompt: &str, cloud: &runa_cloud::CloudRef| {
        if grammar.is_some() {
            return Err("--grammar is local-only; use --json-schema with cloud models".into());
        }
        cloud::run_cloud(
            cloud,
            &cloud::CloudRun {
                prompt,
                think,
                max_tokens: args.max_tokens,
                json: args.json,
                audio: args.audio.as_deref(),
                audio_pref,
                json_schema: json_schema.as_deref(),
            },
        )
    };
    if let Some(cloud) = runa_cloud::parse_cloud_ref(&args.model) {
        if !args.image.is_empty() || args.video.is_some() {
            return Err(
                "cloud --image/--video: native mtmd is local-only; send images through the API adapters (P4.7) or run a local VL model".into(),
            );
        }
        return cloud_run(&prompt, &cloud);
    }
    let path = resolve_model(&args.model)?;
    if let Some(p) = args.draft.as_ref()
        && !p.is_file()
    {
        return Err(format!("draft model not found: {}", p.display()));
    }
    let on_unfit = config::resolve_on_unfit(args.on_unfit.as_deref())?;
    let (kv_k, kv_v) = resolve_kv(
        args.kv.as_deref(),
        args.kv_k.as_deref(),
        args.kv_v.as_deref(),
    )?;
    let mut placement = match parse_mode_choice(&args.mode)? {
        ModeChoice::Fixed(mode) => Placement::from_mode(mode),
        ModeChoice::Auto => {
            match auto_placement(
                &path,
                args.ctx,
                &on_unfit,
                planner_kv_type(kv_k, kv_v),
                args.mmproj.as_deref(),
                args.draft.as_deref(),
            )? {
                AutoPlacement::Local(p) => p,
                AutoPlacement::Cloud(cloud) => return cloud_run(&prompt, &cloud),
            }
        }
    };
    if let Some(n) = args.n_cpu_moe {
        placement = placement.with_n_cpu_moe(n);
    }
    if let Some(s) = args.device.as_deref() {
        placement = placement.with_devices(parse_device_list(s)?);
    }
    if let Some(s) = args.tensor_split.as_deref() {
        placement = placement.with_tensor_split(parse_tensor_split(s)?);
    }
    let mmproj_for_audio = args.mmproj.clone().or_else(|| {
        args.audio
            .as_ref()
            .and_then(|_| runa_fit::sibling_mmproj(&path))
    });
    let audio_plan = if args.audio.is_some() {
        Some(runa_media::select_audio_route(
            audio_pref,
            runa_media::AudioBackend::Local {
                has_audio_mmproj: mmproj_for_audio.is_some(),
            },
        )?)
    } else {
        None
    };
    let mmproj = args.mmproj.clone().or_else(|| {
        if matches!(audio_plan, Some(runa_media::AudioPlan::Native))
            || !args.image.is_empty()
            || args.video.is_some()
        {
            runa_fit::sibling_mmproj(&path)
        } else {
            None
        }
    });
    if let (Some(p), Some(runa_media::AudioPlan::Transcribe)) = (&args.audio, audio_plan) {
        prompt = cloud::fold_audio_transcript(&prompt, p)?;
    }
    let config = LoadConfig {
        n_ctx: args.ctx,
        kv_k,
        kv_v,
        mmproj,
        ..LoadConfig::default()
    };
    let draft_bytes = args
        .draft
        .as_ref()
        .map(|p| runa_fit::mmproj_file_bytes(p))
        .unwrap_or(0);
    preflight_grow(&path, args.ctx, planner_kv_type(kv_k, kv_v), draft_bytes)?;
    if (args.ngram || args.draft.is_some()) && args.temperature > 0.0 {
        eprintln!("ngram: skipped (needs --temperature 0)");
    }
    let sampling = SamplingConfig {
        temperature: args.temperature,
        seed: args.seed,
        ..SamplingConfig::default()
    };
    let mut loaded = load(&path, &placement, &config).map_err(|e| e.to_string())?;
    attach_prompt_cache(
        &mut loaded,
        args.no_prompt_cache,
        args.prompt_cache.as_deref(),
    )?;
    let audio_pcm = match (&args.audio, audio_plan) {
        (None, _) | (Some(_), None) => None,
        (Some(_), Some(runa_media::AudioPlan::Transcribe)) => None,
        (Some(p), Some(runa_media::AudioPlan::Native)) => {
            if loaded.supports_native_audio() {
                Some(
                    runa_media::decode_audio(p)
                        .map_err(|e| e.to_string())?
                        .samples,
                )
            } else if audio_pref == runa_media::AudioRoutePref::Auto {
                prompt = cloud::fold_audio_transcript(&prompt, p)?;
                None
            } else {
                return Err(
                    "loaded mmproj has no native audio; use --audio-route asr or auto".into(),
                );
            }
        }
        (Some(_), Some(runa_media::AudioPlan::OpenAiInputAudio)) => {
            return Err("openai input_audio is cloud-only".into());
        }
    };
    let req = GenerateRequest {
        messages: vec![ChatMessage::user(&prompt)],
        sampling,
        max_tokens: args.max_tokens,
        stop: Vec::new(),
        add_generation_prompt: true,
        think,
        audio_pcm,
        images: collect_vision_frames(&args.image, args.video.as_deref())?,
        speculative: runa_engine::Speculative {
            ngram: args.ngram || args.draft.is_some(),
            draft_n: 4,
            draft: args.draft.clone(),
        },
        json_schema,
        grammar,
    };
    let stream = loaded.generate(req).map_err(|e| e.to_string())?;
    if args.json {
        let (text, usage, reason) = stream.collect_text().map_err(|e| e.to_string())?;
        println!("{}", run_json(&text, &usage, &reason));
    } else {
        let mut usage = None;
        for ev in stream {
            match ev.map_err(|e| e.to_string())? {
                GenEvent::Text(piece) => {
                    print!("{piece}");
                    io::stdout().flush().map_err(|e| format!("stdout: {e}"))?;
                }
                GenEvent::Reasoning(piece) => {
                    eprint!("{piece}");
                    io::stderr().flush().map_err(|e| format!("stderr: {e}"))?;
                }
                GenEvent::Usage(u) => usage = Some(u),
                GenEvent::Done(_) => {}
            }
        }
        println!();
        if let Some(u) = usage {
            eprintln!(
                "tokens: prompt {} / generated {} · {:.1} pp tok/s · {:.1} tg tok/s",
                u.prompt_tokens, u.generated_tokens, u.pp_toks_per_s, u.tg_toks_per_s
            );
        }
    }
    Ok(())
}

/// Minimal JSON string escaping (no serde in the binary crate).
pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn run_json(text: &str, usage: &Usage, reason: &StopReason) -> String {
    let stop = match reason {
        StopReason::Eos => "eos".to_owned(),
        StopReason::MaxTokens => "max_tokens".to_owned(),
        StopReason::StopString(s) => format!("stop:{}", json_escape(s)),
    };
    format!(
        "{{\"text\":\"{}\",\"usage\":{{\"prompt_tokens\":{},\"generated_tokens\":{},\"pp_toks_per_s\":{:.1},\"tg_toks_per_s\":{:.1}}},\"stop\":\"{stop}\"}}",
        json_escape(text),
        usage.prompt_tokens,
        usage.generated_tokens,
        usage.pp_toks_per_s,
        usage.tg_toks_per_s,
    )
}

/// REPL session: loaded model + pending mode + stored think args (P3).
struct Session {
    path: PathBuf,
    mode: Mode,
    ctx: u32,
    think: ThinkConfig,
    last_usage: Option<Usage>,
}

#[allow(clippy::too_many_arguments)]
fn cmd_chat(
    model: Option<&str>,
    mode: &str,
    ctx: u32,
    think: Option<String>,
    think_budget: Option<u32>,
    effort: Option<String>,
    show_reasoning: bool,
    no_show_reasoning: bool,
) -> Result<(), String> {
    use rustyline::error::ReadlineError;
    use rustyline::history::FileHistory;
    use rustyline::{Config, Editor};

    let mut mode = parse_mode(mode)?;
    let mut path = match model {
        Some(m) => resolve_model(m)?,
        None => {
            return Err(
                "chat needs a model path or alias (local file, hf: ref, or [models] alias)".into(),
            );
        }
    };
    let mut loaded = load(
        &path,
        &Placement::from_mode(mode),
        &LoadConfig {
            n_ctx: ctx,
            ..LoadConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;
    attach_prompt_cache(&mut loaded, false, None)?;

    let config = Config::builder().auto_add_history(true).build();
    let mut rl: Editor<(), FileHistory> =
        Editor::with_config(config).map_err(|e| format!("readline: {e}"))?;
    if let Some(home) = std::env::var_os("HOME") {
        let hist = PathBuf::from(home).join(".runa_history");
        let _ = rl.load_history(&hist);
    }
    let mut session = Session {
        path: path.clone(),
        mode,
        ctx,
        think: cli_think(
            think.as_deref(),
            think_budget,
            effort.as_deref(),
            show_reasoning,
            no_show_reasoning,
        )?,
        last_usage: None,
    };
    println!("runa chat ({}). /help for commands.", path.display());

    let mut pending_line = String::new();
    loop {
        let prompt = if pending_line.is_empty() {
            "> "
        } else {
            "... "
        };
        match rl.readline(prompt) {
            Ok(line) => {
                // `\` continues on the next line (multi-line paste).
                if line.ends_with('\\') {
                    pending_line.push_str(&line[..line.len() - 1]);
                    pending_line.push('\n');
                    continue;
                }
                pending_line.push_str(&line);
                let input = std::mem::take(&mut pending_line);
                if input.trim().is_empty() {
                    continue;
                }
                if input.starts_with('/') {
                    if chat_command(&input, &mut session, &mut loaded, &mut path, &mut mode)? {
                        break;
                    }
                    continue;
                }
                chat_turn(&mut loaded, &input, &mut session);
            }
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => break,
            Err(e) => {
                eprintln!("input: {e}");
                break;
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let hist = PathBuf::from(home).join(".runa_history");
        let _ = rl.save_history(&hist);
    }
    Ok(())
}

/// Returns true when the session should exit.
fn chat_command(
    input: &str,
    session: &mut Session,
    loaded: &mut runa_engine::LoadedModel,
    path: &mut PathBuf,
    mode: &mut Mode,
) -> Result<bool, String> {
    let mut parts = input[1..].split_whitespace();
    match parts.next().unwrap_or("") {
        "quit" | "exit" | "q" => Ok(true),
        "help" => {
            println!("/mode <cpu|gpu|hybrid>  reload with a placement");
            println!("/model <path>           load another local model");
            println!("/think [on|off|budget N|effort L|show|hide]  thinking (P3.1)");
            println!("/reset                  clear KV cache");
            println!("/usage                  show last turn counters");
            println!("/quit                   leave");
            println!("end a line with \\ to continue it (multi-line paste)");
            Ok(false)
        }
        "mode" => {
            let m = parts.next().ok_or("usage: /mode <cpu|gpu|hybrid>")?;
            *mode = parse_mode(m)?;
            session.mode = *mode;
            reload(session, loaded, path)?;
            Ok(false)
        }
        "model" => {
            let m = parts.next().ok_or("usage: /model <local-path>")?;
            *path = resolve_model(m)?;
            session.path = path.clone();
            reload(session, loaded, path)?;
            Ok(false)
        }
        "think" => {
            let rest: String = parts.collect::<Vec<_>>().join(" ");
            if rest.is_empty() {
                println!("think: {}", session.think);
                return Ok(false);
            }
            session.think = session
                .think
                .apply(&ThinkOverrides::from_slash_args(&rest)?)?;
            println!("think: {}", session.think);
            Ok(false)
        }
        "reset" => {
            loaded.reset_context().map_err(|e| e.to_string())?;
            println!("(context cleared)");
            Ok(false)
        }
        "usage" => {
            match &session.last_usage {
                Some(u) => println!(
                    "prompt {} / generated {} · {:.1} pp tok/s · {:.1} tg tok/s",
                    u.prompt_tokens, u.generated_tokens, u.pp_toks_per_s, u.tg_toks_per_s
                ),
                None => println!("(no turn yet)"),
            }
            Ok(false)
        }
        other => {
            println!("unknown command /{other} — /help lists commands");
            Ok(false)
        }
    }
}

fn reload(
    session: &Session,
    loaded: &mut runa_engine::LoadedModel,
    path: &Path,
) -> Result<(), String> {
    let fresh = load(
        path,
        &Placement::from_mode(session.mode),
        &LoadConfig {
            n_ctx: session.ctx,
            ..LoadConfig::default()
        },
    )
    .map_err(|e| e.to_string())?;
    *loaded = fresh;
    attach_prompt_cache(loaded, false, None)?;
    Ok(())
}

fn chat_turn(loaded: &mut runa_engine::LoadedModel, input: &str, session: &mut Session) {
    let req = GenerateRequest {
        messages: vec![ChatMessage::user(input)],
        sampling: SamplingConfig::default(),
        max_tokens: 512,
        think: session.think,
        ..GenerateRequest::default()
    };
    let stream = match loaded.generate(req) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("generate: {e}");
            return;
        }
    };
    for ev in stream {
        match ev {
            Ok(GenEvent::Text(piece)) => {
                print!("{piece}");
                let _ = io::stdout().flush();
            }
            Ok(GenEvent::Reasoning(piece)) => {
                eprint!("{piece}");
                let _ = io::stderr().flush();
            }
            Ok(GenEvent::Usage(u)) => session.last_usage = Some(u),
            Ok(GenEvent::Done(_)) => {}
            Err(e) => {
                eprintln!("\ngenerate: {e}");
                return;
            }
        }
    }
    println!();
}

fn cmd_pull(model_ref: &str) -> Result<(), String> {
    let pulled = pull::pull(model_ref)?;
    if pulled.fresh {
        println!(
            "{} ({}/{}; {} bytes)",
            pulled.path.display(),
            pulled.repo,
            pulled.file,
            pulled.size
        );
    } else {
        println!(
            "{} ({}/{}; already present, verified)",
            pulled.path.display(),
            pulled.repo,
            pulled.file
        );
    }
    Ok(())
}

fn task_registry_path() -> PathBuf {
    std::env::var("RUNA_TASK_REGISTRY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/tasks.md"))
}

fn cmd_tasks(action: TaskAction) -> Result<(), String> {
    let reg = TaskRegistry::open(task_registry_path());
    match action {
        TaskAction::List {} => {
            for id in reg.list_free() {
                println!("{id}");
            }
            Ok(())
        }
        TaskAction::Claim { task_id, agent } => match reg.claim(&task_id, &agent) {
            Ok(claim) => {
                println!(
                    "claimed {task_id} as {agent} at {started}",
                    started = claim.started_at
                );
                Ok(())
            }
            Err(ClaimError::AlreadyClaimed {
                agent: owner,
                started_at,
            }) => Err(format!(
                "{task_id}: in progress (owner: {owner}, since {started_at})"
            )),
            Err(e) => Err(e.to_string()),
        },
        TaskAction::Release { task_id, agent } => {
            reg.release(&task_id, &agent).map_err(|e| e.to_string())?;
            println!("released {task_id}");
            Ok(())
        }
    }
}

fn cmd_models() -> Result<(), String> {
    let aliases = config::load_aliases()?;
    println!("models in {}:", pull::models_dir().display());
    for p in pull::list_models() {
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({size} bytes)", p.display());
    }
    if !aliases.models.is_empty() {
        println!("aliases:");
        let mut names: Vec<&String> = aliases.models.keys().collect();
        names.sort();
        for n in names {
            println!("  {n} -> {}", aliases.models[n].source);
        }
    }
    Ok(())
}

/// Cargo features compiled into this binary (plan D13 / P6.3).
fn compiled_backends() -> Vec<&'static str> {
    let mut out = vec!["cpu"];
    if cfg!(feature = "metal") {
        out.push("metal");
    }
    if cfg!(feature = "cuda") {
        out.push("cuda");
    }
    if cfg!(feature = "vulkan") {
        out.push("vulkan");
    }
    if cfg!(feature = "mtmd") {
        out.push("mtmd");
    }
    out
}

fn doctor(json: bool) {
    let native_build = cfg!(feature = "native");
    let backends = compiled_backends();
    if json {
        let payload = serde_json::json!({
            "backends": backends,
            "native_build": native_build,
        });
        println!("{payload}");
    } else {
        println!("runa doctor");
        println!(
            "native build (-C target-cpu=native): {}",
            if native_build {
                "yes"
            } else {
                "no (portable; ggml runtime dispatch)"
            }
        );
        println!("backends compiled in: {}", backends.join(", "));
    }
}

#[cfg(test)]
mod docs_lint {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn man_page_via_clap_mangen() {
        let cmd = Cli::command();
        let mut buf = Vec::new();
        clap_mangen::Man::new(cmd.clone())
            .render(&mut buf)
            .expect("render man");
        let s = String::from_utf8(buf).expect("utf8 man");
        assert!(s.contains("runa"), "{s}");
        assert!(s.contains("SUBCOMMANDS"), "{s}");
        let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs");
        std::fs::write(docs.join("runa.1"), &s).expect("write runa.1");
        let run = cmd.find_subcommand("run").expect("run subcommand").clone();
        let mut run_buf = Vec::new();
        clap_mangen::Man::new(run)
            .render(&mut run_buf)
            .expect("render run man");
        let run_s = String::from_utf8(run_buf).expect("utf8 run man");
        assert!(
            run_s.contains("audio") && run_s.contains("route"),
            "{run_s}"
        );
        std::fs::write(docs.join("runa-run.1"), &run_s).expect("write runa-run.1");
    }
}
