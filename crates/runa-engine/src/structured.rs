//! Structured output (P8.1): JSON Schema / GBNF constrained generation.
//!
//! Rendering goes through llama.cpp's own Jinja chat handler
//! (`apply_chat_template_oaicompat`), which knows each model family's
//! reply format and returns the grammar that constrains it, plus any extra
//! stop strings. Models without a template fall back to a plain prompt and
//! a grammar built straight from the schema.

use llama_cpp_2::model::{ChatTemplateResult, GrammarTrigger, GrammarTriggerType};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::generate::ChatMessage;
use crate::load::{EngineError, LoadedModel};

/// A GBNF grammar ready to become a sampler.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Grammar {
    gbnf: String,
    /// Lazy: enforced only after a trigger matches (tool calls, P8.2).
    lazy: bool,
    patterns: Vec<String>,
    tokens: Vec<LlamaToken>,
}

impl Grammar {
    fn eager(gbnf: String) -> Grammar {
        Grammar {
            gbnf,
            lazy: false,
            patterns: Vec::new(),
            tokens: Vec::new(),
        }
    }

    pub(crate) fn sampler(&self, loaded: &LoadedModel) -> Result<LlamaSampler, EngineError> {
        let model = loaded.model();
        let built = if self.lazy {
            LlamaSampler::grammar_lazy_patterns(
                model,
                &self.gbnf,
                "root",
                &self.patterns,
                &self.tokens,
            )
        } else {
            LlamaSampler::grammar(model, &self.gbnf, "root")
        };
        built.map_err(|e| EngineError::Grammar(format!("{e:?}")))
    }
}

/// Prompt + grammar for one constrained request.
pub(crate) struct Constrained {
    pub prompt: String,
    pub templated: bool,
    pub grammar: Option<Grammar>,
    pub stops: Vec<String>,
}

/// JSON Schema → GBNF (llama.cpp's converter).
pub fn schema_to_grammar(schema: &str) -> Result<String, EngineError> {
    serde_json::from_str::<serde_json::Value>(schema)
        .map_err(|e| EngineError::Grammar(format!("json schema is not valid JSON: {e}")))?;
    llama_cpp_2::json_schema_to_grammar(schema).map_err(|e| EngineError::Grammar(e.to_string()))
}

impl LoadedModel {
    /// Render `messages` for a request that carries a JSON Schema or a raw
    /// GBNF grammar. Thinking is disabled: an eager grammar leaves no room
    /// for a reasoning block.
    pub(crate) fn render_constrained(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        json_schema: Option<&str>,
        grammar: Option<&str>,
    ) -> Result<Constrained, EngineError> {
        // Validates both inputs up front and is the fallback grammar.
        let fallback = eager_grammar(json_schema, grammar)?;
        let Ok(tmpl) = self.model().chat_template(None) else {
            return Ok(Constrained {
                prompt: self.render_prompt(messages, add_generation_prompt)?,
                templated: false,
                grammar: fallback,
                stops: Vec::new(),
            });
        };
        let messages_json = messages_json(messages);
        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json,
            tools_json: None,
            tool_choice: None,
            json_schema,
            grammar,
            reasoning_format: None,
            chat_template_kwargs: Some(r#"{"enable_thinking":false}"#),
            add_generation_prompt,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: false,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };
        let result = self
            .model()
            .apply_chat_template_oaicompat(&tmpl, &params)
            .map_err(|e| EngineError::Template(format!("{e:?}")))?;
        Ok(from_template(result, fallback))
    }
}

/// Grammar straight from `--grammar` / `--json-schema`, for prompts that do
/// not go through the Jinja handler (no template, media prefill).
pub(crate) fn eager_grammar(
    json_schema: Option<&str>,
    grammar: Option<&str>,
) -> Result<Option<Grammar>, EngineError> {
    Ok(match (json_schema, grammar) {
        (Some(_), Some(_)) => {
            return Err(EngineError::Grammar(
                "json schema and grammar are mutually exclusive".into(),
            ));
        }
        (Some(s), None) => Some(Grammar::eager(schema_to_grammar(s)?)),
        (None, Some(g)) => Some(Grammar::eager(g.to_owned())),
        (None, None) => None,
    })
}

/// The template's own grammar (format-aware: channel prefixes, lazy tool
/// triggers) wins over the plain fallback.
fn from_template(result: ChatTemplateResult, fallback: Option<Grammar>) -> Constrained {
    let (patterns, tokens) = trigger_patterns(&result.grammar_triggers);
    let grammar = match result.grammar {
        Some(gbnf) => Some(Grammar {
            gbnf,
            lazy: result.grammar_lazy,
            patterns,
            tokens,
        }),
        None => fallback,
    };
    Constrained {
        prompt: result.prompt,
        templated: true,
        grammar,
        stops: result.additional_stops,
    }
}

/// OpenAI-style `[{role, content}]` for the Jinja handler.
fn messages_json(messages: &[ChatMessage]) -> String {
    serde_json::Value::Array(
        messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect(),
    )
    .to_string()
}

/// Lazy-grammar triggers → regex patterns + trigger tokens, the way
/// llama-server feeds `llama_sampler_init_grammar_lazy_patterns`.
fn trigger_patterns(triggers: &[GrammarTrigger]) -> (Vec<String>, Vec<LlamaToken>) {
    let mut patterns = Vec::new();
    let mut tokens = Vec::new();
    for t in triggers {
        match t.trigger_type {
            GrammarTriggerType::Token => tokens.extend(t.token),
            GrammarTriggerType::Word => patterns.push(regex_escape(&t.value)),
            GrammarTriggerType::Pattern => patterns.push(t.value.clone()),
            GrammarTriggerType::PatternFull => {
                let mut p = t.value.clone();
                if !p.starts_with('^') {
                    p.insert(0, '^');
                }
                if !p.ends_with('$') {
                    p.push('$');
                }
                patterns.push(p);
            }
        }
    }
    (patterns, tokens)
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if r"\.^$|?*+()[]{}-".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trigger(trigger_type: GrammarTriggerType, value: &str) -> GrammarTrigger {
        GrammarTrigger {
            trigger_type,
            value: value.to_owned(),
            token: None,
        }
    }

    #[test]
    fn schema_to_grammar_has_root() {
        let g = schema_to_grammar(
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#,
        )
        .unwrap();
        assert!(g.contains("root ::="), "{g}");
    }

    #[test]
    fn schema_must_be_json() {
        assert!(schema_to_grammar("{not json").is_err());
    }

    #[test]
    fn triggers_become_patterns() {
        let (p, t) = trigger_patterns(&[
            trigger(GrammarTriggerType::Word, "<tool_call>"),
            trigger(GrammarTriggerType::Pattern, "[\\s\\S]*?(<tool_call>)"),
            trigger(GrammarTriggerType::PatternFull, "(?:x)"),
        ]);
        assert_eq!(p, ["<tool_call>", "[\\s\\S]*?(<tool_call>)", "^(?:x)$"]);
        assert!(t.is_empty());
    }

    #[test]
    fn word_triggers_are_escaped() {
        assert_eq!(regex_escape("[TOOL_CALLS]"), "\\[TOOL_CALLS\\]");
        assert_eq!(regex_escape("a.b"), "a\\.b");
    }

    #[test]
    fn messages_json_is_openai_shape() {
        let j = messages_json(&[ChatMessage::user("hi \"x\"")]);
        assert_eq!(j, r#"[{"content":"hi \"x\"","role":"user"}]"#);
    }
}
