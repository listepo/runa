# D06 — Speed prediction = bandwidth model + calibration

- Status: accepted (2026-09-08)
- Context: decode is memory-bound; first-principles FLOPS models mislead.
- Decision: `decode tok/s = eff(device, quant) × BW / bytes_per_token` with
  defaults (CUDA/Metal 0.60, Vulkan/CPU 0.50), updated from measured runs in
  a local SQLite calibration DB (median measured/predicted per
  device/backend/quant). Predictions are always ranges.
- Consequences: cold error ≤ ±30 %, ≤ ±15 % after 3 runs (M3).
- Verification: P1.9 (within ±30 % of P0.6 baselines cold), P1.10 (DB test).
