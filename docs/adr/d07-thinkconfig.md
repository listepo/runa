# D07 — Thinking primitive (`ThinkConfig`)

- Status: accepted (2026-09-08)
- Context: no analog unifies reasoning control across local models and
  OpenAI/Anthropic clouds.
- Decision: one `ThinkConfig { Off | On | Budget{tokens,grace} |
  Effort(Low|Medium|High|Max) }`. Local: budget forcing in our sampling loop
  (count after think-open, bias close at budget−grace, inject at budget).
  Cloud: OpenAI `reasoning.effort`; Anthropic `thinking: adaptive` +
  `output_config.effort`. Output splits `Event::Reasoning` / `Event::Text`.
- Consequences: same flag works for Qwen3, gpt-oss, DeepSeek, GPT-x,
  Claude; fit accounts reasoning time.
- Verification: P3.3 — 100 runs ≤ budget+grace in 100 % of runs.
