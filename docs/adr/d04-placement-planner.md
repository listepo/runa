# D04 — Three modes = one placement planner

- Status: accepted (2026-09-08)
- Context: llama.cpp exposes placement as knobs (`-ngl`, `-ot`,
  `--n-cpu-moe`, `--device`), not modes; MoE hybrids matter (experts ≈ 90 %
  of file, ~3B active per token).
- Decision: `--mode cpu|gpu|hybrid|auto` produces one
  `Placement { n_gpu_layers, tensor_overrides, kv_device, mmproj_device }`.
  `hybrid` keeps MoE experts on CPU first, then sheds GPU layers. `auto` is
  the fit checker's best plan.
- Consequences: one planner replaces three code paths; hybrid speed follows
  the harmonic model (CPU side dominates).
- Verification: P1.8 fixture tests (all-GPU / cpu / experts-on-CPU).
