# `rpc/` — vendored llama.cpp RPC backend (P9.3)

`ggml/src/ggml-rpc/ggml-rpc.cpp` is the upstream ggml RPC backend,
**verbatim** from the exact commit our `llama-cpp-sys-2 =0.1.133` pin builds:

- llama.cpp tag **`b7709`**, commit
  `1051ecd28907d2ca0a15c135f190fe415d0a3d1b` (2026-01-12)
- source: `https://raw.githubusercontent.com/ggerganov/llama.cpp/<commit>/ggml/src/ggml-rpc/ggml-rpc.cpp`

The `ggml/include/` + the five `ggml/src/*.h` files are byte copies of the
same commit (identical to what the sys crate ships in its own tree and
compiles against) — they exist only as the compile-time include path for
`ggml-rpc.cpp`. Nothing here is modified, and nothing here is a rewrite
(plan D1/D23: ggml stays upstream code).

Why vendored instead of a fork or a pin bump:

- No published `llama-cpp-2` (surveyed 0.1.131–0.1.156) exposes an `rpc`
  cargo feature, and the sys crate strips the `ggml-rpc/` sources, so
  `-DGGML_RPC=ON` cannot work on this pin (see `docs/versions.md`).
- `GGML_USE_RPC` in ggml core only auto-registers an *empty* base reg; the
  real work (`ggml_backend_rpc_add_server` + `ggml_backend_register`) goes
  through the public C API in the shipped `ggml-rpc.h` / `ggml-backend.h`,
  so compiling this one file against the same-version headers and linking it
  next to the sys static lib is ABI-safe by construction.
- Only the `rpc` cargo feature compiles this file; the default build is
  untouched (no new objects, no new symbols).

Refresh rule (plan D16): on every `llama-cpp-2` upgrade, re-fetch
`ggml-rpc.cpp` **and** the headers from the new pin's llama.cpp commit. The
`build.rs` pin-guard fails the `rpc` build if `llama-cpp-sys-2` drifts from
`=0.1.133`, so a stale vendoring breaks loudly, never silently.
