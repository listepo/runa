# D13 — Platform tiers

- Status: accepted (2026-09-08)
- Context: local inference happens on Apple Silicon, NVIDIA Linux, and
  plain CPUs; backends are cargo features with different build needs.
- Decision: Tier 1 — macOS arm64 (Metal), Linux x86_64 (CUDA, Vulkan, CPU).
  Tier 2 — Linux aarch64, Windows x86_64 (CUDA/Vulkan). `runa doctor`
  lists compiled backends.
- Tier 3 — manual (P9.4, 2026-09-15): Qualcomm Hexagon and Intel OpenVINO
  NPUs. Opt-in cargo features (`hexagon`, `openvino`) are probe-only
  stubs: `llama-cpp-2` has no such backends through 0.1.154 (see
  `docs/versions.md`), there are no NPU CI runners, and placement never
  defaults to NPU (plan D12). `runa doctor` reports `hexagon-stub` /
  `openvino-stub` only when built with the feature; runtime validation and
  speed calibration are manual on-device.
- Consequences: CI covers Tier 1 fully; Tier 2 is build (+smoke) only.
  Tier 3 is docs + probe + build-gating only; no CI job (no runners).
- Verification: P0.2 CI matrix green.
