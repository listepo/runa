# D16 — Version pins

- Status: accepted (2026-09-08)
- Context: upstream (llama.cpp ~50 builds/week, Rust, cloud APIs) drifts
  fast; silent drift breaks fit math and the build.
- Decision: Rust toolchain, `llama-cpp-2`, whisper-rs, async-openai pinned
  in `Cargo.lock` and `docs/versions.md` (P0.3 owns the file);
  `rust-toolchain.toml` + `mise.toml` pin the toolchain. llama.cpp upgraded
  monthly through the benchmark gate.
- Consequences: upgrades are deliberate PRs with bench evidence.
- Verification: P0.3 (`cargo --version` matches pin; `versions.md` exists).
