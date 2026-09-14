//! P2.3/P2.5 e2e: `runa run` (one-shot, stdin piping, `--json`, auto/`on_unfit`)
//! and `runa chat` (piped REPL with slash commands), via `assert_cmd`.
//!
//! Each test loads the qwen2 fixture on CPU (~seconds); the binary is built
//! once by cargo.

use std::path::PathBuf;

use assert_cmd::Command;
use predicates::prelude::*;

fn runa() -> Command {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    // Isolate tests from ~/.cache/runa/kv (LMDB is process-locked).
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn run_streams_text_and_reports_usage() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "The capital of France is",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(!stdout.trim().is_empty(), "streams text to stdout");
    assert!(
        stderr.contains("tokens:"),
        "usage line on stderr: {stderr:?}"
    );
}

#[test]
fn run_json_is_parseable() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    // Minimal structural check without a JSON dep in e2e scope.
    assert!(stdout.contains("\"text\""), "{stdout:?}");
    assert!(stdout.contains("\"usage\""), "{stdout:?}");
    assert!(stdout.contains("\"stop\""), "{stdout:?}");
}

#[test]
fn run_reads_prompt_from_stdin() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "--max-tokens",
            "4",
        ])
        .write_stdin("Say the word blue.")
        .assert()
        .success();
}

#[test]
fn run_missing_model_fails_cleanly() {
    // Partial stderr match via predicates; exit code via assert_cmd.
    runa()
        .args(["run", "no-such-model.gguf", "hi"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no such model file"));
}

#[test]
fn chat_answers_then_quits() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args(["chat", model.to_str().unwrap()])
        .write_stdin("What is 2+2?\n/usage\n/quit\n")
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(
        stdout.contains("prompt"),
        "usage counters shown: {stdout:?}"
    );
}

#[test]
fn chat_second_turn_reuses_the_context() {
    // A second turn used to prefill onto the first turn's KV cells (P8.3).
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args(["chat", model.to_str().unwrap()])
        .write_stdin("Say hi.\nSay bye.\n/quit\n")
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(!stderr.contains("generate:"), "{stderr}");
}

#[test]
fn run_mcp_tool_loop_calls_the_server() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let dir = assert_fs::TempDir::new().unwrap();
    let log = dir.path().join("calls.jsonl");
    let server = format!("python3 {}", fixture("mcp-echo.py").display());
    let out = runa()
        .env("MCP_ECHO_LOG", &log)
        .args([
            "run",
            "--mode",
            "cpu",
            model.to_str().unwrap(),
            "What is the weather in Paris? Use the get_weather tool.",
            "--mcp",
            &server,
            "--temperature",
            "0",
            "--max-tokens",
            "96",
            "--max-tool-rounds",
            "2",
        ])
        .output()
        .expect("runa runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The 0.5B model may keep calling instead of answering; the round cap
    // is then the only acceptable failure.
    assert!(
        out.status.success() || stderr.contains("tool rounds"),
        "{stderr}"
    );
    assert!(stderr.contains("mcp: 1 tools from 1 servers"), "{stderr}");
    assert!(stderr.contains("[tool] get_weather("), "{stderr}");
    let calls = std::fs::read_to_string(&log).expect("server saw a call");
    assert!(calls.contains("\"get_weather\""), "{calls}");
}

#[test]
fn run_mcp_missing_command_fails_before_loading() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .args([
            "run",
            model.to_str().unwrap(),
            "hi",
            "--mcp",
            "runa-no-such-mcp-server --flag",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("spawn runa-no-such-mcp-server"));
}

#[test]
fn auto_unfit_falls_back_to_cpu() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "64")
        .env("RUNA_MEMORY_CEILING_MIB", "65536")
        .env_remove("RUNA_ON_UNFIT")
        .args([
            "run",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
            "--on-unfit",
            "cpu",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("NO FIT"),
        "verdict line on stderr: {stderr:?}"
    );
    assert!(
        stderr.contains("on_unfit=cpu"),
        "explicit CPU fallback warning: {stderr:?}"
    );
    assert!(
        stderr.contains("cpu (0 layers on GPU)"),
        "load on CPU after fallback: {stderr:?}"
    );
}

#[test]
fn auto_unfit_error_exits_2() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_FAKE_VRAM", "0")
        .env("RUNA_FAKE_RAM", "64")
        .env("RUNA_MEMORY_CEILING_MIB", "65536")
        .env_remove("RUNA_ON_UNFIT")
        .args([
            "run",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
            "--on-unfit",
            "error",
        ])
        .assert()
        .code(2)
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("NO FIT"), "{stderr:?}");
    assert!(stderr.contains("on_unfit=error"), "{stderr:?}");
    assert!(
        !stderr.contains("cpu (0 layers on GPU)"),
        "must not load after on_unfit=error: {stderr:?}"
    );
}

#[test]
fn run_over_ceiling_memory_suggests() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .env("RUNA_MEMORY_CEILING_MIB", "1")
        .env("RUNA_MEMORY_MAX_GROWTH_MIB", "1")
        .args([
            "run",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .code(2)
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("ceiling") && stderr.contains("smaller --ctx"),
        "over-ceiling must suggest a smaller ctx: {stderr:?}"
    );
    assert!(
        !stderr.contains("cpu (0 layers on GPU)"),
        "must not load after grow_for over-ceiling: {stderr:?}"
    );
}

#[test]
fn run_n_cpu_moe_prints_expert_overrides() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--n-cpu-moe",
            "2",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("experts-on-cpu:"),
        "verdict lists expert overrides: {stderr:?}"
    );
    assert!(
        stderr.contains("blk\\.(0|1)"),
        "first-N-layers expert pattern: {stderr:?}"
    );
}

#[test]
fn run_prompt_cache_hits_on_second_run() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let cache = std::env::temp_dir().join(format!(
        "runa-p28-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    std::fs::create_dir_all(&cache).expect("prompt-cache dir");
    let cache_s = cache.to_str().expect("utf8 cache path");
    let model_s = model.to_str().unwrap();
    let first = runa()
        .env_remove("RUNA_NO_PROMPT_CACHE")
        .args([
            "run",
            "--mode",
            "cpu",
            "--prompt-cache",
            cache_s,
            model_s,
            "The capital of France is",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr1 = String::from_utf8(first.stderr).expect("utf8 stderr");
    assert!(
        stderr1.contains("prompt-cache: store"),
        "first run stores KV: {stderr1:?}"
    );
    assert!(
        !stderr1.contains("prompt-cache: hit"),
        "first run must miss: {stderr1:?}"
    );
    let second = runa()
        .env_remove("RUNA_NO_PROMPT_CACHE")
        .args([
            "run",
            "--mode",
            "cpu",
            "--prompt-cache",
            cache_s,
            model_s,
            "The capital of France is",
            "--max-tokens",
            "8",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr2 = String::from_utf8(second.stderr).expect("utf8 stderr");
    assert!(
        stderr2.contains("prompt-cache: hit"),
        "second run restores KV from LMDB: {stderr2:?}"
    );
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn run_kv_q8_prints_type_on_verdict() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--kv",
            "q8_0",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("kv k=q8_0 v=q8_0"),
        "verdict lists KV types: {stderr:?}"
    );
}

#[test]
fn run_kv_unknown_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--kv",
            "int8",
            model.to_str().unwrap(),
            "hi",
        ])
        .assert()
        .failure();
}

#[test]
fn run_kv_k_v_overrides_kv() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--kv",
            "q4_0",
            "--kv-k",
            "q8_0",
            "--kv-v",
            "q8_0",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "4",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("kv k=q8_0 v=q8_0"),
        "--kv-k/--kv-v override --kv: {stderr:?}"
    );
}

#[test]
fn run_help_lists_multi_gpu_flags() {
    let out = runa()
        .args(["run", "--help"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(stdout.contains("--device"), "{stdout}");
    assert!(stdout.contains("--tensor-split"), "{stdout}");
    assert!(stdout.contains("--audio"), "{stdout}");
    assert!(stdout.contains("--mmproj"), "{stdout}");
    assert!(stdout.contains("--audio-route"), "{stdout}");
    assert!(stdout.contains("--image"), "{stdout}");
    assert!(stdout.contains("--video"), "{stdout}");
    assert!(stdout.contains("--ngram"), "{stdout}");
    assert!(stdout.contains("--draft"), "{stdout}");
}

#[test]
fn run_draft_missing_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--draft",
            "/no/such/draft.gguf",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("draft model not found"), "{stderr}");
}

#[test]
fn run_lora_missing_fails() {
    // P8.5: a mistyped `--lora` fails fast naming the adapter (before the
    // model finishes loading).
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
            "--lora",
            "no-such-adapter.gguf:0.5",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(stderr.contains("no-such-adapter"), "{stderr}");
    assert!(stderr.contains("lora"), "{stderr}");
}

#[test]
fn run_device_unknown_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "gpu",
            "--device",
            "999",
            "--ctx",
            "512",
            model.to_str().unwrap(),
            "hi",
            "--max-tokens",
            "1",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("device index 999") || stderr.contains("out of range"),
        "unknown device must fail: {stderr:?}"
    );
}

#[test]
fn run_tensor_split_on_cpu_fails() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let out = runa()
        .args([
            "run",
            "--mode",
            "cpu",
            "--tensor-split",
            "3,1",
            model.to_str().unwrap(),
            "hi",
        ])
        .assert()
        .failure()
        .get_output()
        .clone();
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    assert!(
        stderr.contains("tensor-split") && stderr.contains("GPU"),
        "CPU + tensor-split must error: {stderr:?}"
    );
}

#[test]
fn serve_help_lists_flags() {
    let out = runa()
        .args(["serve", "--help"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    assert!(stdout.contains("--host"), "{stdout}");
    assert!(stdout.contains("--port"), "{stdout}");
    assert!(stdout.contains("--models"), "{stdout}");
    assert!(stdout.contains("--parallel"), "{stdout}");
    assert!(stdout.contains("OpenAI"), "{stdout}");
}

#[test]
fn serve_health_models_and_chat() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(rest.to_string());
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve printed listening on …");
    let mut kill = || {
        let _ = child.kill();
        let _ = child.wait();
    };

    let health = curl(&format!("{base}/health"));
    if !health.contains("ok") {
        kill();
        panic!("health: {health}");
    }
    let models = curl(&format!("{base}/v1/models"));
    if !models.contains("owned_by") {
        kill();
        panic!("models: {models}");
    }
    let chat = curl_post(
        &format!("{base}/v1/chat/completions"),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
    );
    if !(chat.contains("assistant") && chat.contains("content")) {
        kill();
        panic!("non-stream chat: {chat}");
    }
    let streamed = curl_post(
        &format!("{base}/v1/chat/completions"),
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#,
    );
    if !(streamed.contains("data:") && streamed.contains("[DONE]")) {
        kill();
        panic!("stream completion: {streamed}");
    }
    openai_python_sdk_smoke(&base);
    let anthropic = curl_post(
        &format!("{base}/v1/messages"),
        r#"{"max_tokens":4,"messages":[{"role":"user","content":"hi"}]}"#,
    );
    if !(anthropic.contains("\"type\":\"message\"")
        || anthropic.contains("\"type\": \"message\"")
        || anthropic.contains("end_turn"))
    {
        kill();
        panic!("anthropic messages: {anthropic}");
    }
    anthropic_python_sdk_smoke(&base);
    kill();
}

#[test]
fn serve_embeddings_and_transcriptions_routes() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(rest.to_string());
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve listening");
    let mut kill = || {
        let _ = child.kill();
        let _ = child.wait();
    };

    let emb = curl_post(&format!("{base}/v1/embeddings"), r#"{"input":"hello"}"#);
    if !(emb.contains("\"embedding\"") && emb.contains("\"object\"")) {
        kill();
        panic!("embeddings response: {emb}");
    }

    let wav = std::env::temp_dir().join(format!("runa-asr-{}.wav", std::process::id()));
    // Minimal mono 16-bit PCM WAV (silence).
    let wav_bytes: [u8; 44] = [
        0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74,
        0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x80, 0x3e, 0x00, 0x00, 0x00, 0x7d,
        0x00, 0x00, 0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00,
    ];
    std::fs::write(&wav, &wav_bytes).expect("temp wav");
    let asr = curl_post_multipart(&format!("{base}/v1/audio/transcriptions"), &wav);
    let _ = std::fs::remove_file(&wav);
    if !(asr.contains("whisper") || asr.contains("\"text\"") || asr.contains("missing")) {
        kill();
        panic!("transcriptions route: {asr}");
    }
    kill();
}

#[test]
fn serve_parallel_eight_chat() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    use std::time::Duration;

    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let bin = assert_cmd::cargo::cargo_bin("runa");
    let mut child = std::process::Command::new(bin)
        .args([
            "serve",
            "--mode",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ctx",
            "512",
            "--parallel",
            "8",
            model.to_str().unwrap(),
        ])
        .env("RUNA_NO_PROMPT_CACHE", "1")
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn serve");
    let stderr = child.stderr.take().expect("stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if let Some(rest) = line.strip_prefix("listening on ") {
                let _ = tx.send(rest.to_string());
            }
        }
    });
    let base = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("serve listening");
    let mut kill = || {
        let _ = child.kill();
        let _ = child.wait();
    };

    let url = format!("{base}/v1/chat/completions");
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let u = url.clone();
            let b = body.to_owned();
            std::thread::spawn(move || curl_post_timeout(&u, &b, "600"))
        })
        .collect();
    for h in handles {
        let chat = h.join().expect("thread");
        if !(chat.contains("assistant") && chat.contains("content")) {
            kill();
            panic!("parallel chat: {chat}");
        }
    }
    kill();
}

fn curl(url: &str) -> String {
    let out = std::process::Command::new("curl")
        .args(["-sS", "--max-time", "60", url])
        .output()
        .expect("curl");
    assert!(
        out.status.success(),
        "GET {url}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8")
}

fn openai_python_sdk_smoke(base: &str) {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/serve-openai-smoke.py");
    sdk_python_smoke("OpenAI", &script, base);
}

fn anthropic_python_sdk_smoke(base: &str) {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/serve-anthropic-smoke.py");
    sdk_python_smoke("Anthropic", &script, base);
}

fn sdk_python_smoke(label: &str, script: &std::path::Path, base: &str) {
    let required = std::env::var("RUNA_REQUIRE_OPENAI_SMOKE").is_ok();
    let out = match std::process::Command::new("python3")
        .arg(script)
        .arg(base)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            if required {
                panic!("python3 {label} smoke: {e}");
            }
            eprintln!("skip {label} SDK smoke (no python3): {e}");
            return;
        }
    };
    if out.status.success() {
        return;
    }
    let err = String::from_utf8_lossy(&out.stderr);
    let skip = err.contains("ModuleNotFoundError") || err.contains("No module named");
    if skip && !required {
        eprintln!("skip {label} SDK smoke: {err}");
        return;
    }
    panic!(
        "{label} Python SDK smoke failed: {} {}",
        String::from_utf8_lossy(&out.stdout),
        err
    );
}

fn curl_post(url: &str, body: &str) -> String {
    curl_post_timeout(url, body, "120")
}

fn curl_post_timeout(url: &str, body: &str, max_secs: &str) -> String {
    let out = std::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            max_secs,
            "-H",
            "content-type: application/json",
            "-d",
            body,
            url,
        ])
        .output()
        .expect("curl");
    assert!(
        out.status.success(),
        "POST {url}: {} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8")
}

fn curl_post_multipart(url: &str, file: &PathBuf) -> String {
    let out = std::process::Command::new("curl")
        .args([
            "-sS",
            "--max-time",
            "120",
            "-F",
            &format!("file=@{}", file.display()),
            url,
        ])
        .output()
        .expect("curl");
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        return format!("{} {}", stdout, stderr);
    }
    stdout
}
