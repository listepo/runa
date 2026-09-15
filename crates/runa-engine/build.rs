//! P5.7: remind that `--features native` needs RUSTFLAGS for GGML_NATIVE.
//!
//! P9.4: warn when an NPU stub feature (`hexagon`, `openvino`) is on but its
//! SDK env is missing. The features are probe-only stubs (no ggml backend in
//! llama-cpp-2 through 0.1.154 — see `docs/versions.md`); the warning keeps
//! that explicit at build time (plan D12: no silent acceleration claims).

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=HEXAGON_SDK_ROOT");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
    let feat = std::env::var("CARGO_FEATURE_NATIVE").is_ok();
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let rustc_native = flags
        .split('\u{1f}')
        .any(|f| f.contains("target-cpu=native"));
    if feat && !rustc_native {
        println!(
            "cargo:warning=feature `native` is on, but RUSTFLAGS does not contain              -C target-cpu=native; llama-cpp-sys-2 will keep GGML_NATIVE=OFF              (portable runtime dispatch). For a host-tuned build:              RUSTFLAGS='-C target-cpu=native' cargo build --release --features native"
        );
    }
    // P9.4: stub features with no ggml backend behind them. Warn twice: the
    // feature changes no placement (stub), and the vendor SDK is missing
    // (needed for any future real enablement + manual on-device validation).
    if std::env::var("CARGO_FEATURE_HEXAGON").is_ok() {
        println!(
            "cargo:warning=feature `hexagon` is a probe-only stub: llama-cpp-2 has               no `hexagon` backend through 0.1.154 (see docs/versions.md), so              placement stays CPU. Runtime validation is manual on-device (Tier 3)."
        );
        if std::env::var("HEXAGON_SDK_ROOT").is_err() {
            println!(
                "cargo:warning=feature `hexagon` is on but HEXAGON_SDK_ROOT is unset              (Qualcomm Hexagon SDK); install it before any on-device validation."
            );
        }
    }
    if std::env::var("CARGO_FEATURE_OPENVINO").is_ok() {
        println!(
            "cargo:warning=feature `openvino` is a probe-only stub: llama-cpp-2 has               no `openvino` backend through 0.1.154 (see docs/versions.md), so              placement stays CPU. Runtime validation is manual on-device (Tier 3)."
        );
        if std::env::var("INTEL_OPENVINO_DIR").is_err() {
            println!(
                "cargo:warning=feature `openvino` is on but INTEL_OPENVINO_DIR is unset              (OpenVINO setupvars.sh); source it before any on-device validation."
            );
        }
    }
}
