//! `runa fit` (P1.11): will a model run here, and how fast, before any
//! download. `--recommend` (P8.4) ranks the built-in catalog instead.

use std::io::Write;

use runa_fit::recommend::{self, Pick};
use runa_fit::{
    Descriptor, Fetcher, FitConfig, FitReport, HwSpec, ModelSource, PlannerConfig, Reader,
    check_fit, format_report, parse_model_ref,
};
use serde_json::{Value, json};

use crate::config::AliasTable;

#[derive(Debug, clap::Args)]
pub(crate) struct FitArgs {
    /// Local GGUF, alias, `hf:<repo>:<file-or-quant>`, or an http(s) URL.
    #[arg(required_unless_present = "recommend")]
    model: Option<String>,
    /// Context length.
    #[arg(long, default_value_t = 8192)]
    ctx: u32,
    /// KV cache type: f16 | q8_0 | q4_0.
    #[arg(long, value_name = "TYPE", default_value = "f16")]
    kv: String,
    /// Emit JSON instead of the report or table.
    #[arg(long)]
    json: bool,
    /// Rank the built-in model catalog for this machine.
    #[arg(long, conflicts_with = "model")]
    recommend: bool,
    /// With --recommend: only models for this use.
    #[arg(long = "use", value_name = "USE", value_parser = recommend::USES, requires = "recommend")]
    use_: Option<String>,
    /// With --recommend: how many models to list.
    #[arg(long, value_name = "N", default_value_t = 5)]
    top: usize,
    /// With --recommend: fit on catalog file sizes, no network (KV not counted).
    #[arg(long, requires = "recommend")]
    offline: bool,
    /// llama.cpp RPC endpoints (`host:port,…`); estimation stays local and
    /// an explicit warning is printed — the pinned sys crate has no ggml
    /// RPC backend, so `runa run --rpc` errors instead of running (P9.3).
    #[arg(long, value_name = "LIST")]
    rpc: Option<String>,
}

/// This machine as the fit checker sees it.
pub(crate) fn machine_config(
    ctx: u32,
    kv_type: &str,
    mmproj_bytes: u64,
) -> Result<(FitConfig, &'static str), String> {
    let (vram, vram_src) = crate::vram_bytes()?;
    let gpu_hw = (vram > 0).then(|| {
        if cfg!(target_os = "macos") {
            HwSpec::metal()
        } else {
            HwSpec::cuda()
        }
    });
    let config = FitConfig {
        planner: PlannerConfig {
            vram_bytes: vram,
            ram_bytes: crate::ram_bytes(),
            ctx_len: u64::from(ctx),
            kv_type: kv_type.to_owned(),
            mmproj_bytes,
            ..PlannerConfig::default()
        },
        gpu_hw,
        cpu_hw: HwSpec::cpu(),
        has_mmproj: false,
        media: runa_fit::MediaFit::default(),
    };
    Ok((config, vram_src))
}

pub(crate) fn cmd_fit(args: &FitArgs) -> Result<(), String> {
    let kv: runa_engine::KvKind = args.kv.parse().map_err(|e| format!("--kv: {e}"))?;
    let code = if args.recommend {
        cmd_recommend(args, kv.as_str())?
    } else {
        let model = args.model.as_deref().ok_or("fit: a model is required")?;
        fit_one(model, args, kv.as_str())?
    };
    if code != 0 {
        let _ = std::io::stdout().flush();
        std::process::exit(code);
    }
    Ok(())
}

/// Pulled copy first (no network), else the header over HTTP.
fn model_source(model: &str) -> Result<ModelSource, String> {
    let aliases = crate::config::load_aliases()?;
    let target = aliases
        .get(model)
        .map_or_else(|| model.to_owned(), |a| a.source.clone());
    match crate::pull::find_local(&target, &AliasTable::default()) {
        Ok(path) => Ok(ModelSource::Local(path)),
        Err(_) => parse_model_ref(&target).map_err(|e| e.to_string()),
    }
}

fn fit_one(model: &str, args: &FitArgs, kv: &str) -> Result<i32, String> {
    let src = model_source(model)?;
    if let Some(rpc) = args.rpc.as_deref() {
        // Validate the shape now (same parser as `run --rpc`) so a typo
        // fails here instead of passing silently into a local estimate.
        let servers = runa_engine::parse_rpc_list(rpc).map_err(|e| format!("--rpc: {e}"))?;
        eprintln!(
            "warning: --rpc {} requested but this build has no ggml RPC backend \
             (llama-cpp-sys-2 0.1.133; see docs/versions.md): estimating local placement; \
             `runa run --rpc` will error explicitly",
            servers.join(",")
        );
    }
    // P9.2: `runa fit` estimates GGUF placement; a mistral model directory
    // has no GGUF header to check — refuse explicitly, never silently.
    if let ModelSource::Local(p) = &src
        && runa_core::is_mistral_dir(p)
    {
        return Err(format!(
            "fit: {} is a safetensors (mistral) model; `runa fit` only estimates GGUF models",
            p.display()
        ));
    }
    let fetcher = Fetcher::new().map_err(|e| e.to_string())?;
    let header = fetcher.fetch_header(&src).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let mmproj_bytes = match &src {
        ModelSource::Local(p) => runa_fit::sibling_mmproj(p)
            .map(|m| runa_fit::mmproj_file_bytes(&m))
            .unwrap_or(0),
        _ => 0,
    };
    let (config, _) = machine_config(args.ctx, kv, mmproj_bytes)?;
    let report = check_fit(&desc, &config);
    if args.json {
        let decode = recommend::decode_for(&desc, &report, &config);
        println!("{}", report_json(model, &report, decode));
    } else {
        print!("{}", format_report(&report));
    }
    Ok(i32::from(report.exit_code))
}

fn report_json(model: &str, report: &FitReport, decode: f64) -> Value {
    let plan = &report.plan;
    let speed = report.speed_gpu.as_ref().or(report.speed_cpu.as_ref());
    json!({
        "model": model,
        "verdict": report.verdict.to_string(),
        "exit_code": report.exit_code,
        "gpu_layers": plan.gpu_layers,
        "cpu_layers": plan.cpu_layers,
        "gpu_weight_bytes": plan.gpu_weight_bytes,
        "cpu_weight_bytes": plan.cpu_weight_bytes,
        "kv_bytes": plan.kv_bytes,
        "compute_bytes": plan.compute_buffer_bytes,
        "decode_toks_per_sec": decode,
        "prefill_toks_per_sec": speed.map(|s| s.prefill_toks_per_sec),
        "warnings": report.warnings.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "suggestions": report.suggestions.iter().map(ToString::to_string).collect::<Vec<_>>(),
    })
}

fn cmd_recommend(args: &FitArgs, kv: &str) -> Result<i32, String> {
    let use_ = args.use_.as_deref();
    let (base, vram_src) = machine_config(args.ctx, kv, 0)?;
    let config_for = |e: &recommend::CatalogEntry| {
        let mut c = base.clone();
        if use_ == Some("vision") {
            c.planner.mmproj_bytes = e.mmproj;
        }
        c
    };
    let catalog = recommend::catalog();
    let (picks, errors) = if args.offline {
        recommend::recommend(&catalog, use_, args.top, |e| {
            Ok(recommend::probe_offline(e, &config_for(e)))
        })
    } else {
        let fetcher = Fetcher::new().map_err(|e| e.to_string())?;
        recommend::recommend(&catalog, use_, args.top, |e| {
            recommend::probe_remote(&fetcher, e, &config_for(e))
        })
    };
    for (name, e) in &errors {
        eprintln!("skip {name}: {e}");
    }
    if args.json {
        let rows: Vec<Value> = picks.iter().map(pick_json).collect();
        println!("{}", Value::Array(rows));
    } else {
        let gib = |b: u64| b as f64 / f64::from(1 << 30);
        println!(
            "runa fit --recommend · VRAM {:.0} GiB ({vram_src}) · RAM {:.0} GiB · ctx {} · kv {kv}{}",
            gib(base.planner.vram_bytes),
            gib(base.planner.ram_bytes),
            args.ctx,
            if args.offline {
                " · offline: sizes only"
            } else {
                ""
            },
        );
        print!("{}", table(&picks));
    }
    if picks.is_empty() {
        eprintln!("nothing in the catalog fits; try a smaller --ctx or --kv q8_0");
        return Ok(2);
    }
    Ok(0)
}

fn pick_json(p: &Pick) -> Value {
    json!({
        "name": p.entry.name,
        "ref": p.entry.model,
        "uses": p.entry.uses,
        "tier": p.entry.tier,
        "verdict": p.verdict.to_string(),
        "decode_toks_per_sec": p.decode_toks_per_sec,
        "estimated": p.estimated,
    })
}

fn table(picks: &[Pick]) -> String {
    let name_w = picks.iter().map(|p| p.entry.name.len()).max().unwrap_or(0);
    let mut out = String::new();
    for (i, p) in picks.iter().enumerate() {
        let tps = format!(
            "{}{:.0} tok/s",
            if p.estimated { "~" } else { "" },
            p.decode_toks_per_sec
        );
        out.push_str(&format!(
            "{:>2}. {:<name_w$}  tier {}  {:>11}  {}\n    runa pull {}\n",
            i + 1,
            p.entry.name,
            p.entry.tier,
            tps,
            p.verdict,
            p.entry.model,
        ));
    }
    out
}
