//! P5.7: remind that `--features native` needs RUSTFLAGS for GGML_NATIVE.

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
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
}
