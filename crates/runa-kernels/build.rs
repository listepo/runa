fn main() {
    println!("cargo:rerun-if-changed=zig/kernels.zig");

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let dest = std::path::Path::new(&out).join("libruna_kernels_zig.a");
    let zig = std::env::var("ZIG").unwrap_or_else(|_| "zig".into());
    let mut cmd = std::process::Command::new(&zig);
    cmd.args([
        "build-lib",
        "zig/kernels.zig",
        "-OReleaseFast",
        "-fPIC",
        &format!("-femit-bin={}", dest.display()),
    ]);
    // Windows/MSVC link fails with LNK2019 on `___chkstk_ms` (stack probe
    // emitted by Zig std, e.g. mem.sort): the static archive carries no
    // compiler-rt. Bundling it fixes the link (CI windows-2022).
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        cmd.arg("-fcompiler-rt");
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("zig 0.14.1 (mise, D23): {e}"));
    if !status.success() {
        panic!("zig build-lib failed: {status}");
    }

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=runa_kernels_zig");
}
