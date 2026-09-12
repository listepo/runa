# D13 — Platform tiers

- Status: accepted (2026-09-08)
- Context: local inference happens on Apple Silicon, NVIDIA Linux, and
  plain CPUs; backends are cargo features with different build needs.
- Decision: Tier 1 — macOS arm64 (Metal), Linux x86_64 (CUDA, Vulkan, CPU).
  Tier 2 — Linux aarch64, Windows x86_64 (CUDA/Vulkan). `runa doctor`
  lists compiled backends.
- Consequences: CI covers Tier 1 fully; Tier 2 is build (+smoke) only.
- Verification: P0.2 CI matrix green.
