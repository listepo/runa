# Ideas

- P8.7 follow-up: wire `CalibrationDb.get_efficiency` into `predicted_speeds` (M3: predictions ignore calibration DB).
- P8.7 follow-up: P-core thread default or `--threads` knob (M4: runa pins 16 threads incl. E-cores, loses to llama-bench auto).
- P8.7 follow-up (bug): `--lang auto` ASR returns empty transcript, `set_detect_language` suppresses decode (M7 blocker).
- P8.7 follow-up: expose reasoning token counts in `--json` (M6 compliance unverifiable externally).
- P8.7 follow-up: call `Loaded::on_idle` on serve idle tick / revisit M11 gate feasibility (RSS flat with resident model).
- P8.7 follow-up: quiet-machine re-runs (M5 TTFT, M7 60 s, M8 audio, M4 8B) + release build for M10 size + Metal build for real M4/M5 gates.
