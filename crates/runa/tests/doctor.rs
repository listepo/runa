//! P5.7 / P6.3: portable default reports `native_build: false` and `cpu`.

use assert_cmd::Command;
use pretty_assertions::assert_eq;
use serde_json::Value;

#[test]
fn doctor_json_default_is_portable() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor", "--json"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(
        v["native_build"], false,
        "CI must ship the portable feature set (no AVX-512 required): {v}"
    );
    let backends = v["backends"]
        .as_array()
        .expect("backends array")
        .iter()
        .map(|b| b.as_str().expect("backend string"))
        .collect::<Vec<_>>();
    assert!(
        backends.contains(&"cpu"),
        "every binary includes the CPU ggml backend: {v}"
    );
    // P9.4: NPU capabilities are opt-in only — a default binary must not
    // advertise them (plan D12, no silent claims). (Feature builds assert
    // the opposite in the cfg-gated tests below.)
    #[cfg(not(any(feature = "hexagon", feature = "openvino")))]
    assert!(
        !backends
            .iter()
            .any(|b| b.contains("hexagon") || b.contains("openvino")),
        "default binary must not list NPU backends: {v}"
    );
}

#[test]
fn doctor_text_lists_cpu() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("backends compiled in: cpu"),
        "expected compiled-backend line, got: {stdout}"
    );
}

// P9.4: each NPU stub string appears only in a binary built with its
// feature. These tests compile only under the matching feature.
#[cfg(feature = "hexagon")]
#[test]
fn doctor_json_reports_hexagon_stub() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor", "--json"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    let backends = v["backends"]
        .as_array()
        .expect("backends array")
        .iter()
        .map(|b| b.as_str().expect("backend string"))
        .collect::<Vec<_>>();
    assert!(
        backends.contains(&"hexagon-stub"),
        "hexagon feature must advertise the stub: {v}"
    );
}

#[cfg(feature = "openvino")]
#[test]
fn doctor_json_reports_openvino_stub() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor", "--json"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    let backends = v["backends"]
        .as_array()
        .expect("backends array")
        .iter()
        .map(|b| b.as_str().expect("backend string"))
        .collect::<Vec<_>>();
    assert!(
        backends.contains(&"openvino-stub"),
        "openvino feature must advertise the stub: {v}"
    );
}

/// P9.3: RPC is reported explicitly — `rpc: false` in JSON (no ggml RPC
/// backend in llama-cpp-sys-2 0.1.133) and a one-line note in text.
#[test]
fn doctor_json_reports_rpc_unavailable() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor", "--json"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    assert_eq!(v["rpc"], false, "this pin has no ggml RPC backend: {v}");
}

#[test]
fn doctor_text_reports_rpc() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("rpc:") && stdout.contains("--rpc"),
        "expected explicit rpc line, got: {stdout}"
    );
}

/// P9.2: the default (portable) build has no mistral backend.
#[test]
#[cfg(not(feature = "mistralrs"))]
fn doctor_json_default_has_no_mistralrs() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor", "--json"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    let backends = v["backends"]
        .as_array()
        .expect("backends array")
        .iter()
        .map(|b| b.as_str().expect("backend string"))
        .collect::<Vec<_>>();
    assert!(
        !backends.contains(&"mistralrs"),
        "default build must stay GGUF-only: {v}"
    );
}

/// P9.2: the mistralrs build reports the backend.
#[test]
#[cfg(feature = "mistralrs")]
fn doctor_json_mistralrs_reports_backend() {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd.env("RUNA_NO_KEYRING", "1");
    let out = cmd.args(["doctor", "--json"]).output().expect("doctor");
    assert!(out.status.success(), "{out:?}");
    let v: Value = serde_json::from_slice(&out.stdout).expect("json");
    let backends = v["backends"]
        .as_array()
        .expect("backends array")
        .iter()
        .map(|b| b.as_str().expect("backend string"))
        .collect::<Vec<_>>();
    assert!(
        backends.contains(&"mistralrs"),
        "mistralrs build must report the backend: {v}"
    );
}
