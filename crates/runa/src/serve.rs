//! OpenAI-compatible HTTP server (plan P3.9 / D11).
//!
//! Binds `127.0.0.1` by default. `/v1/chat/completions` (stream + non-stream),
//! `/v1/models`, `/health`. Request fields `reasoning_effort` and
//! `reasoning_budget_tokens`. Reasoning is surfaced as `reasoning_content`.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use futures::stream;
use runa_core::{Effort, ThinkConfig, ThinkOverrides};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Mode, Placement, SamplingConfig, load,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::oneshot;

pub(crate) fn cmd_serve(
    model: &str,
    host: &str,
    port: u16,
    mode: &str,
    ctx: u32,
) -> Result<(), String> {
    let path = crate::resolve_model(model)?;
    let placement = match crate::parse_mode_choice(mode)? {
        crate::ModeChoice::Fixed(m) => Placement::from_mode(m),
        crate::ModeChoice::Auto => Placement::from_mode(Mode::Cpu),
    };
    let config = LoadConfig {
        n_ctx: ctx,
        ..LoadConfig::default()
    };
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("runa")
        .to_owned();
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(listen(host, port, path, placement, config, id))
}

async fn listen(
    host: &str,
    port: u16,
    path: PathBuf,
    placement: Placement,
    config: LoadConfig,
    model_id: String,
) -> Result<(), String> {
    let jobs = spawn_engine(path, placement, config)?;
    let state = AppState { jobs, model_id };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| format!("bind {host}:{port}: {e}"))?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    eprintln!("listening on http://{bound}");
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}

struct Job {
    req: GenerateRequest,
    resp: oneshot::Sender<Result<Vec<GenEvent>, String>>,
}

fn spawn_engine(
    path: PathBuf,
    placement: Placement,
    config: LoadConfig,
) -> Result<std::sync::mpsc::Sender<Job>, String> {
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("runa-engine".into())
        .spawn(move || {
            let mut loaded = match load(&path, &placement, &config) {
                Ok(m) => {
                    let _ = ready_tx.send(Ok(()));
                    m
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                    return;
                }
            };
            let mut used = false;
            while let Ok(job) = rx.recv() {
                let out = (|| {
                    if used {
                        loaded.clear_kv();
                    }
                    used = true;
                    let generation = loaded.generate(job.req).map_err(|e| e.to_string())?;
                    generation
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| e.to_string())
                })();
                let _ = job.resp.send(out);
            }
        })
        .map_err(|e| e.to_string())?;
    ready_rx
        .recv()
        .map_err(|_| "engine thread stopped during load".to_string())??;
    Ok(tx)
}

#[derive(Clone)]
struct AppState {
    jobs: std::sync::mpsc::Sender<Job>,
    model_id: String,
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn list_models(State(st): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.model_id,
            "object": "model",
            "owned_by": "runa"
        }]
    }))
}

async fn chat_completions(
    State(st): State<AppState>,
    Json(body): Json<ChatCompletionBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let think = think_from_request(
        body.reasoning_effort.as_deref(),
        body.reasoning_budget_tokens,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let messages = messages_from_body(&body.messages);
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    let max_tokens = body
        .max_completion_tokens
        .or(body.max_tokens)
        .unwrap_or(512);
    let sampling = SamplingConfig {
        temperature: body.temperature.unwrap_or(0.8),
        ..SamplingConfig::default()
    };
    let req = GenerateRequest {
        messages,
        sampling,
        max_tokens,
        stop: Vec::new(),
        add_generation_prompt: true,
        think,
        audio_pcm: None,
        images: Vec::new(),
        speculative: runa_engine::Speculative::default(),
    };
    let model_id = body.model.unwrap_or_else(|| st.model_id.clone());
    if body.stream.unwrap_or(false) {
        let events = generate_events(&st.jobs, req)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let sse = stream_chunks(&model_id, &events);
        return Ok(Sse::new(stream::iter(
            sse.into_iter().map(Ok::<_, std::convert::Infallible>),
        ))
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let events = generate_events(&st.jobs, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(non_stream_body(&model_id, &events)).into_response())
}

async fn anthropic_messages(
    State(st): State<AppState>,
    Json(body): Json<MessagesBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let think =
        think_from_anthropic(body.thinking.as_ref()).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mut messages = Vec::new();
    if let Some(sys) = body.system.as_ref() {
        let text = content_text(Some(sys));
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: text,
            });
        }
    }
    messages.extend(messages_from_body(&body.messages));
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    let max_tokens = body.max_tokens.unwrap_or(512);
    let sampling = SamplingConfig {
        temperature: body.temperature.unwrap_or(0.8),
        ..SamplingConfig::default()
    };
    let req = GenerateRequest {
        messages,
        sampling,
        max_tokens,
        stop: Vec::new(),
        add_generation_prompt: true,
        think,
        audio_pcm: None,
        images: Vec::new(),
        speculative: runa_engine::Speculative::default(),
    };
    let model_id = body.model.unwrap_or_else(|| st.model_id.clone());
    if body.stream.unwrap_or(false) {
        let events = generate_events(&st.jobs, req)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let sse = anthropic_stream_chunks(&model_id, &events);
        return Ok(Sse::new(stream::iter(
            sse.into_iter().map(Ok::<_, std::convert::Infallible>),
        ))
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let events = generate_events(&st.jobs, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(anthropic_message_body(&model_id, &events)).into_response())
}

async fn generate_events(
    jobs: &std::sync::mpsc::Sender<Job>,
    req: GenerateRequest,
) -> Result<Vec<GenEvent>, String> {
    let (resp, rx) = oneshot::channel();
    jobs.send(Job { req, resp })
        .map_err(|_| "engine thread stopped".to_string())?;
    rx.await.map_err(|e| e.to_string())?
}

fn stream_chunks(model: &str, events: &[GenEvent]) -> Vec<Event> {
    let id = completion_id();
    let mut out = Vec::new();
    for ev in events {
        let delta = match ev {
            GenEvent::Text(t) if !t.is_empty() => json!({"content": t}),
            GenEvent::Reasoning(t) if !t.is_empty() => json!({"reasoning_content": t}),
            _ => continue,
        };
        let body = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
        });
        out.push(Event::default().data(body.to_string()));
    }
    let done = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    out.push(Event::default().data(done.to_string()));
    out.push(Event::default().data("[DONE]"));
    out
}

fn non_stream_body(model: &str, events: &[GenEvent]) -> Value {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage = None;
    for ev in events {
        match ev {
            GenEvent::Text(t) => content.push_str(t),
            GenEvent::Reasoning(t) => reasoning.push_str(t),
            GenEvent::Usage(u) => usage = Some(u),
            GenEvent::Done(_) => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    let usage = usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.generated_tokens,
            "total_tokens": u.prompt_tokens + u.generated_tokens
        })
    });
    json!({
        "id": completion_id(),
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop"
        }],
        "usage": usage
    })
}

fn completion_id() -> String {
    format!(
        "chatcmpl-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    )
}

#[derive(Debug, Deserialize)]
struct ChatCompletionBody {
    model: Option<String>,
    messages: Vec<IncomingMessage>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    reasoning_effort: Option<String>,
    reasoning_budget_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MessagesBody {
    model: Option<String>,
    messages: Vec<IncomingMessage>,
    stream: Option<bool>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    system: Option<IncomingContent>,
    thinking: Option<ThinkingBody>,
}

#[derive(Debug, Deserialize)]
struct ThinkingBody {
    #[serde(rename = "type")]
    kind: Option<String>,
    budget_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct IncomingMessage {
    role: String,
    #[serde(default)]
    content: Option<IncomingContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum IncomingContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    text: Option<String>,
}

pub(crate) fn think_from_request(
    reasoning_effort: Option<&str>,
    reasoning_budget_tokens: Option<u32>,
) -> Result<ThinkConfig, String> {
    let mut o = ThinkOverrides::default();
    o.show = Some(true);
    if let Some(n) = reasoning_budget_tokens {
        if n == 0 {
            o.think = Some(false);
        } else {
            o.budget = Some(n);
        }
    } else if let Some(e) = reasoning_effort {
        o.effort = Some(Effort::parse(e)?);
    } else {
        o.think = Some(true);
    }
    ThinkConfig::default().apply(&o)
}

fn think_from_anthropic(thinking: Option<&ThinkingBody>) -> Result<ThinkConfig, String> {
    let Some(t) = thinking else {
        return think_from_request(None, None);
    };
    match t.kind.as_deref() {
        Some("disabled") => think_from_request(None, Some(0)),
        Some("enabled") => think_from_request(None, t.budget_tokens.or(Some(1024))),
        Some("adaptive") => think_from_request(Some("high"), None),
        _ => think_from_request(None, t.budget_tokens),
    }
}

fn anthropic_message_body(model: &str, events: &[GenEvent]) -> Value {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut usage = None;
    for ev in events {
        match ev {
            GenEvent::Text(t) => text.push_str(t),
            GenEvent::Reasoning(t) => thinking.push_str(t),
            GenEvent::Usage(u) => usage = Some(u),
            GenEvent::Done(_) => {}
        }
    }
    let mut content = Vec::new();
    if !thinking.is_empty() {
        content.push(json!({"type": "thinking", "thinking": thinking}));
    }
    content.push(json!({"type": "text", "text": text}));
    let usage = usage.map(|u| {
        json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.generated_tokens
        })
    });
    json!({
        "id": format!("msg-{}", completion_id().trim_start_matches("chatcmpl-")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn",
        "usage": usage
    })
}

fn anthropic_stream_chunks(model: &str, events: &[GenEvent]) -> Vec<Event> {
    let id = format!("msg-{}", completion_id().trim_start_matches("chatcmpl-"));
    let mut out = Vec::new();
    let start = json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": null
        }
    });
    out.push(
        Event::default()
            .event("message_start")
            .data(start.to_string()),
    );
    let mut idx = 0u32;
    for ev in events {
        match ev {
            GenEvent::Reasoning(t) if !t.is_empty() => {
                out.push(
                    Event::default().event("content_block_delta").data(
                        json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "thinking_delta", "thinking": t}
                        })
                        .to_string(),
                    ),
                );
            }
            GenEvent::Text(t) if !t.is_empty() => {
                out.push(
                    Event::default().event("content_block_delta").data(
                        json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": t}
                        })
                        .to_string(),
                    ),
                );
            }
            _ => {}
        }
    }
    let _ = idx;
    out.push(
        Event::default().event("message_delta").data(
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}).to_string(),
        ),
    );
    out.push(
        Event::default()
            .event("message_stop")
            .data(json!({"type": "message_stop"}).to_string()),
    );
    out
}

fn messages_from_body(messages: &[IncomingMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: content_text(m.content.as_ref()),
        })
        .filter(|m| !m.content.is_empty() || m.role == "assistant")
        .collect()
}

fn content_text(c: Option<&IncomingContent>) -> String {
    match c {
        Some(IncomingContent::Text(s)) => s.clone(),
        Some(IncomingContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runa_core::ThinkMode;

    #[test]
    fn effort_and_budget_fields() {
        let t = think_from_request(Some("high"), None).unwrap();
        assert!(matches!(t.mode, ThinkMode::Effort(Effort::High)));
        assert!(t.show);
        let t = think_from_request(None, Some(256)).unwrap();
        assert!(matches!(t.mode, ThinkMode::Budget { tokens: 256, .. }));
        let t = think_from_request(None, Some(0)).unwrap();
        assert!(matches!(t.mode, ThinkMode::Off));
    }

    #[test]
    fn messages_join_text_parts() {
        let raw = r#"[{"role":"user","content":[{"type":"text","text":"hi "},{"type":"text","text":"there"}]}]"#;
        let msgs: Vec<IncomingMessage> = serde_json::from_str(raw).unwrap();
        let out = messages_from_body(&msgs);
        assert_eq!(out[0].content, "hi there");
    }

    #[test]
    fn non_stream_includes_reasoning_content() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
        ];
        let v = non_stream_body("m", &events);
        assert_eq!(v["choices"][0]["message"]["content"], "ok");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "plan");
    }

    #[test]
    fn anthropic_message_includes_thinking_block() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
        ];
        let v = anthropic_message_body("m", &events);
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["type"], "thinking");
        assert_eq!(v["content"][0]["thinking"], "plan");
        assert_eq!(v["content"][1]["text"], "ok");
        let t = think_from_anthropic(Some(&ThinkingBody {
            kind: Some("enabled".into()),
            budget_tokens: Some(256),
        }))
        .unwrap();
        assert!(matches!(t.mode, ThinkMode::Budget { tokens: 256, .. }));
    }

    #[test]
    fn stream_chunks_include_reasoning_and_done() {
        let events = vec![
            GenEvent::Reasoning("plan".into()),
            GenEvent::Text("ok".into()),
        ];
        let chunks = stream_chunks("m", &events);
        assert_eq!(chunks.len(), 4, "reasoning, text, finish, [DONE]");
    }
}
