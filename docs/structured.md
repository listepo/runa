# docs/structured.md — structured output

`runa` can force a model's answer into a JSON Schema or a GBNF grammar.
Local models use llama.cpp grammar sampling, so the output always matches.
OpenAI models get the schema as `response_format` (`strict: true`).

## CLI

```sh
runa run qwen "Capital of France and its population?" \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"]}'
runa run qwen "Is Paris in France?" --grammar yes-no.gbnf
```

| Flag | Value | Notes |
| --- | --- | --- |
| `--json-schema` | file path, or inline JSON starting with `{` | Local and OpenAI. The Anthropic adapter has no structured output yet. |
| `--grammar` | GBNF file | Local only. Conflicts with `--json-schema`. |

A constrained request turns thinking off, and it turns off the Zig kernel
sampler and n-gram speculation. The grammar must see every token.

## `runa serve`

`POST /v1/chat/completions` accepts the OpenAI `response_format`:

| `type` | Effect |
| --- | --- |
| `text` | No constraint |
| `json_object` | Any JSON object (`{"type":"object"}`) |
| `json_schema` | `json_schema.schema` is required; the answer matches it |

An invalid schema returns `400`.

## How it renders

Constrained requests render through llama.cpp's Jinja chat handler
(`apply_chat_template_oaicompat`). The handler knows the model's chat
format. It returns the prompt, the grammar, lazy-grammar triggers, and extra
stop strings. Models without a chat template fall back to the plain prompt
plus an eager grammar. Schemas are converted with llama.cpp's
`json_schema_to_grammar` (`runa_engine::schema_to_grammar`).
