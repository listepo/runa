# D08 — Media pipeline in Rust, encoders in C

- Status: accepted (2026-09-08)
- Context: neither cloud API takes video; Anthropic takes no audio. Local
  audio/vision needs encoder models (mmproj) or ASR.
- Decision: decode in Rust (`symphonia`/`hound` → f32 mono 16 kHz;
  `ffmpeg-sidecar` → ≤ 32 frames @1 fps + scene keyframes). Three routes:
  **native** (mtmd audio/vision), **asr** (whisper.cpp / Parakeet), **cloud**
  (OpenAI `input_audio`/images; Anthropic images/transcript only).
- Consequences: honest capability matrix per backend; `runa fit` prints the
  media route and counts mmproj + frame tokens.
- Verification: P0.5 mtmd spike; P4.4 route matrix test; P4.8 fit tests.
