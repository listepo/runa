# Roadmap

Approved work that is not yet in the active plan.

## After 1.0

- `runa daemon`: long-running background service (launchd / systemd) that keeps models warm between CLI calls, owns the adaptive memory manager, and serves `run` / `chat` over a local socket.
- `mistral.rs` backend behind a feature for safetensors and omni models ggml cannot run.
- Distributed inference via llama.cpp RPC across machines.
- NPU backends (Hexagon, OpenVINO) where ggml supports them.
