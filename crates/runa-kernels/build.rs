fn main() {
    println!("cargo:rerun-if-changed=zig/kernels.zig");

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let dest = std::path::Path::new(&out).join("libruna_kernels_zig.a");
    let zig = std::env::var("ZIG").unwrap_or_else(|_| "zig".into());
    let status = std::process::Command::new(&zig)
        .args([
            "build-lib",
            "zig/kernels.zig",
            "-OReleaseFast",
            "-fPIC",
            "-fcompiler-rt",
            &format!("-femit-bin={}", dest.display()),
        ])
        .status()
        .unwrap_or_else(|e| panic!("zig 0.14.1 (mise, D23): {e}"));
    if !status.success() {
        panic!("zig build-lib failed: {status}");
    }

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=runa_kernels_zig");
}
