# D14 — Kernel candidates (ordered by expected payoff)

- Status: accepted (2026-09-08)
- Context: own kernels only pay where ggml is weak; guessing wastes weeks.
- Decision: profile first, then in order — (1) sampling over 150k vocabs,
  (2) image preprocessing, (3) audio front-end, (4) SME2/AMX quantized
  mat-vec only where ggml lacks a path. Implement new kernels in Zig when
  Zig is better (D23); C/`.S` otherwise. Each: scalar reference, fuzz
  equivalence, `criterion` bench, runtime dispatch, ≥ 5 % gate.
- Consequences: P5 is time-boxed; scalar reference always ships.
- Verification: `docs/kernels.md` per-kernel results; P5.8 perf CI.
