fn main() {
    println!("cargo:rerun-if-changed=zig/kernels.zig");

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let dest = std::path::Path::new(&out).join("libruna_kernels_zig.a");
    let zig = std::env::var("ZIG").unwrap_or_else(|_| {
        // P14.2: dist's Windows build step runs under PowerShell, where the
        // mise shims dir is not on PATH (mise-action adds it for bash steps
        // only) — `zig` resolves in CI's bash steps but not here. The shims
        // are wrappers, not real exes, so probe the real install dirs.
        // USERPROFILE (~.local/share/mise, seen on GHA windows-2022) first,
        // then LOCALAPPDATA (mise.exe's own dir: shims live next to it, and
        // installs may sit under %LOCALAPPDATA/mise/installs).
        let mut bases = Vec::new();
        if let Ok(home) = std::env::var("USERPROFILE") {
            bases.push(std::path::Path::new(&home).join(".local/share/mise/installs/zig"));
        }
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            bases.push(std::path::Path::new(&local).join("mise/installs/zig"));
        }
        for base in bases {
            if let Ok(rd) = std::fs::read_dir(&base) {
                let mut vers: Vec<_> = rd.filter_map(|e| e.ok()).collect();
                vers.sort_by_key(|e| e.file_name());
                if let Some(latest) = vers.pop() {
                    // zig layout: <ver>/bin/zig.exe (core backend);
                    // aqua backend nests one level deeper.
                    for cand in [
                        latest.path().join("bin/zig.exe"),
                        latest.path().join("zig.exe"),
                    ] {
                        if cand.exists() {
                            return cand.to_string_lossy().into_owned();
                        }
                    }
                    if let Ok(rd2) = std::fs::read_dir(latest.path()) {
                        for sub in rd2.filter_map(|e| e.ok()) {
                            let cand = sub.path().join("bin/zig.exe");
                            if cand.exists() {
                                return cand.to_string_lossy().into_owned();
                            }
                        }
                    }
                }
            }
        }
        "zig".into()
    });
    let mut cmd = std::process::Command::new(&zig);
    cmd.args([
        "build-lib",
        "zig/kernels.zig",
        "-OReleaseFast",
        "-fPIC",
        &format!("-femit-bin={}", dest.display()),
    ]);
    // Windows/MSVC link needs CRT-provided `__chkstk`, but Zig's native
    // target detection emits MinGW-style `___chkstk_ms` probes (LNK2019).
    // The explicit triple emits `__chkstk` instead (verified via nm);
    // unix builds stay on native detection (no glibc-floor surprises).
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        cmd.arg("-target").arg("x86_64-windows-msvc");
    }
    // Zig tunes for the build host by default (AVX-512 on some CI runners).
    // A cached or released library then dies with SIGILL on an older CPU,
    // so stay on the baseline ISA unless Rust itself targets the host (P5.7).
    let native = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .is_ok_and(|f| f.split('\x1f').any(|f| f.contains("target-cpu=native")));
    cmd.arg(if native {
        "-mcpu=native"
    } else {
        "-mcpu=baseline"
    });
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("zig 0.16.0 (mise, D23): {e}"));
    if !status.success() {
        panic!("zig build-lib failed: {status}");
    }

    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=runa_kernels_zig");
}
