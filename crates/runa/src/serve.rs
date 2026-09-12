//! OpenAI-compatible HTTP server (plan P3.9 / P6.1 / D11).
//!
//! Multi-model LRU pool, `--parallel` in-flight cap, `/v1/embeddings`,
//! `/v1/audio/transcriptions`, and multimodal chat `content` parts.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use futures::stream;
use runa_core::{Effort, ThinkConfig, ThinkOverrides};
use runa_engine::{
    ChatMessage, GenEvent, GenerateRequest, LoadConfig, Mode, Placement, SamplingConfig,
    VisionFrame, VisionSource, load,
};
use runa_fit::{
    Descriptor, FitConfig, HwSpec, PlannerConfig, Reader, check_fit, read_local_prefix,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::SemaphorePermit;
use tokio::sync::{Semaphore, oneshot};

/// CLI bundle for `runa serve` (P6.1).
pub(crate) struct ServeOpts {
    pub models: Vec<(String, PathBuf)>,
    pub host: String,
    pub port: u16,
    pub mode: String,
    pub ctx: u32,
    pub parallel: usize,
    pub max_loaded: Option<usize>,
}

pub(crate) fn cmd_serve(opts: ServeOpts) -> Result<(), String> {
    if opts.models.is_empty() {
        return Err("serve: need at least one model (positional or --models)".into());
    }
    if opts.parallel == 0 {
        return Err("serve: --parallel must be >= 1".into());
    }
    let placement_base = match crate::parse_mode_choice(&opts.mode)? {
        crate::ModeChoice::Fixed(m) => Placement::from_mode(m),
        crate::ModeChoice::Auto => Placement::from_mode(Mode::Cpu),
    };
    let config = LoadConfig {
        n_ctx: opts.ctx,
        ..LoadConfig::default()
    };
    let max_loaded = opts
        .max_loaded
        .unwrap_or_else(|| opts.models.len().min(opts.parallel).max(1));
    let default_id = opts.models[0].0.clone();
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(listen(
        &opts.host,
        opts.port,
        opts.models,
        placement_base,
        opts.mode,
        config,
        default_id,
        opts.parallel,
        max_loaded,
    ))
}

async fn listen(
    host: &str,
    port: u16,
    models: Vec<(String, PathBuf)>,
    placement_base: Placement,
    mode: String,
    config: LoadConfig,
    default_id: String,
    parallel: usize,
    max_loaded: usize,
) -> Result<(), String> {
    let pool = ModelPool::new(models, placement_base, mode, config, max_loaded)?;
    let state = AppState {
        pool: Arc::new(Mutex::new(pool)),
        default_id,
        parallel: Arc::new(Semaphore::new(parallel)),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
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

enum EngineJob {
    Generate {
        req: GenerateRequest,
        resp: oneshot::Sender<Result<Vec<GenEvent>, String>>,
    },
    Embed {
        input: String,
        resp: oneshot::Sender<Result<Vec<f32>, String>>,
    },
}

struct ModelPool {
    specs: HashMap<String, PathBuf>,
    order: Vec<String>,
    engines: HashMap<String, Arc<std::sync::mpsc::Sender<EngineJob>>>,
    lru: VecDeque<String>,
    max_loaded: usize,
    placement_base: Placement,
    mode: String,
    config: LoadConfig,
}

impl ModelPool {
    fn new(
        models: Vec<(String, PathBuf)>,
        placement_base: Placement,
        mode: String,
        config: LoadConfig,
        max_loaded: usize,
    ) -> Result<Self, String> {
        let mut specs = HashMap::new();
        let mut order = Vec::new();
        for (id, path) in models {
            if !path.is_file() {
                return Err(format!("no such model file: {}", path.display()));
            }
            specs.insert(id.clone(), path);
            order.push(id);
        }
        Ok(ModelPool {
            specs,
            order,
            engines: HashMap::new(),
            lru: VecDeque::new(),
            max_loaded: max_loaded.max(1),
            placement_base,
            mode,
            config,
        })
    }

    fn model_ids(&self) -> &[String] {
        &self.order
    }

    fn resolve_id(&self, model: Option<&str>) -> Result<String, String> {
        let id = model
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(self.order.first().ok_or("no models configured")?);
        if self.specs.contains_key(id) {
            Ok(id.to_owned())
        } else {
            Err(format!("model {id} not found"))
        }
    }

    fn touch_lru(&mut self, id: &str) {
        self.lru.retain(|x| x != id);
        self.lru.push_back(id.to_owned());
    }

    fn evict_if_needed(&mut self) {
        while self.engines.len() > self.max_loaded {
            let victim = self
                .lru
                .front()
                .cloned()
                .filter(|id| self.engines.contains_key(id));
            let Some(id) = victim else {
                break;
            };
            self.lru.pop_front();
            if let Some(tx) = self.engines.remove(&id) {
                drop(tx);
                eprintln!("serve: unloaded model {id} (LRU)");
            }
        }
    }

    fn placement_for(&self, path: &Path) -> Result<Placement, String> {
        match crate::parse_mode_choice(&self.mode)? {
            crate::ModeChoice::Auto => match crate::auto_placement(
                path,
                self.config.n_ctx,
                &crate::config::OnUnfit::Error,
                runa_engine::planner_kv_type(self.config.kv_k, self.config.kv_v),
                None,
                None,
            )? {
                crate::AutoPlacement::Local(p) => Ok(p),
                crate::AutoPlacement::Cloud(_) => {
                    Err("unfit: auto mode chose cloud fallback; serve is local-only".into())
                }
            },
            crate::ModeChoice::Fixed(_) => {
                crate::preflight_grow(
                    path,
                    self.config.n_ctx,
                    runa_engine::planner_kv_type(self.config.kv_k, self.config.kv_v),
                    0,
                )?;
                Ok(self.placement_base.clone())
            }
        }
    }

    fn fit_check_no_fit(path: &Path, ctx: u32) -> Result<(), String> {
        let header = read_local_prefix(path).map_err(|e| e.to_string())?;
        let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
        let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
        let (vram, _) = crate::vram_bytes()?;
        let planner = PlannerConfig {
            vram_bytes: vram,
            ram_bytes: crate::ram_bytes(),
            ctx_len: u64::from(ctx),
            kv_type: runa_engine::planner_kv_type(None, None).to_owned(),
            ..PlannerConfig::default()
        };
        let report = check_fit(
            &desc,
            &FitConfig {
                planner,
                gpu_hw: None,
                cpu_hw: HwSpec::cpu(),
                has_mmproj: false,
                media: runa_fit::MediaFit::default(),
            },
        );
        if matches!(report.verdict, runa_fit::Verdict::NoFit) {
            return Err(format!("unfit: model does not fit (ctx={ctx})"));
        }
        Ok(())
    }

    fn ensure_engine(
        &mut self,
        id: &str,
    ) -> Result<Arc<std::sync::mpsc::Sender<EngineJob>>, String> {
        if self.engines.contains_key(id) {
            self.touch_lru(id);
            return Ok(Arc::clone(self.engines.get(id).expect("contains_key")));
        }
        let path = self
            .specs
            .get(id)
            .ok_or_else(|| format!("model {id} not found"))?
            .clone();
        Self::fit_check_no_fit(&path, self.config.n_ctx)?;
        let placement = self.placement_for(&path)?;
        let tx = Arc::new(spawn_engine(path, placement, self.config.clone())?);
        self.engines.insert(id.to_owned(), Arc::clone(&tx));
        self.touch_lru(id);
        self.evict_if_needed();
        Ok(tx)
    }
}

fn spawn_engine(
    path: PathBuf,
    placement: Placement,
    config: LoadConfig,
) -> Result<std::sync::mpsc::Sender<EngineJob>, String> {
    let (tx, rx) = std::sync::mpsc::channel::<EngineJob>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name(format!(
            "runa-engine-{}",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("m")
        ))
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
                match job {
                    EngineJob::Generate { req, resp } => {
                        let out = (|| {
                            if used {
                                loaded.clear_kv();
                            }
                            used = true;
                            let generation = loaded.generate(req).map_err(|e| e.to_string())?;
                            generation
                                .collect::<Result<Vec<_>, _>>()
                                .map_err(|e| e.to_string())
                        })();
                        let _ = resp.send(out);
                    }
                    EngineJob::Embed { input, resp } => {
                        let out = loaded.embed(&input).map_err(|e| e.to_string());
                        let _ = resp.send(out);
                    }
                }
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
    pool: Arc<Mutex<ModelPool>>,
    default_id: String,
    parallel: Arc<Semaphore>,
}

async fn acquire_parallel(st: &AppState) -> Result<SemaphorePermit<'_>, (StatusCode, String)> {
    st.parallel.acquire().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "server shutting down".into(),
        )
    })
}

async fn with_engine(
    pool: &Arc<Mutex<ModelPool>>,
    model_id: &str,
) -> Result<Arc<std::sync::mpsc::Sender<EngineJob>>, (StatusCode, String)> {
    let pool = pool.clone();
    let id = model_id.to_owned();
    tokio::task::spawn_blocking(move || {
        let mut p = pool
            .lock()
            .map_err(|_| "model pool lock poisoned".to_string())?;
        match p.resolve_id(Some(&id)) {
            Ok(resolved) => p.ensure_engine(&resolved),
            Err(e) if e.contains("not found") => Err(e),
            Err(e) => Err(e),
        }
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| {
        if e.starts_with("unfit:") || e.contains("does not fit") {
            (StatusCode::SERVICE_UNAVAILABLE, e)
        } else if e.contains("not found") {
            (StatusCode::NOT_FOUND, e)
        } else {
            (StatusCode::BAD_REQUEST, e)
        }
    })
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn list_models(State(st): State<AppState>) -> Json<Value> {
    let ids = {
        let pool = st.pool.lock().expect("pool");
        pool.model_ids().to_vec()
    };
    let data = ids
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "owned_by": "runa"
            })
        })
        .collect::<Vec<_>>();
    Json(json!({"object": "list", "data": data}))
}

async fn chat_completions(
    State(st): State<AppState>,
    Json(body): Json<ChatCompletionBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    let think = think_from_request(
        body.reasoning_effort.as_deref(),
        body.reasoning_budget_tokens,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let (messages, images, audio_pcm) =
        messages_from_body(&body.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
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
        audio_pcm,
        images,
        speculative: runa_engine::Speculative::default(),
    };
    let model_id = body
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let jobs = with_engine(&st.pool, &model_id).await?;
    if body.stream.unwrap_or(false) {
        let events = generate_events(&jobs, req)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let sse = stream_chunks(&model_id, &events);
        return Ok(Sse::new(stream::iter(
            sse.into_iter().map(Ok::<_, std::convert::Infallible>),
        ))
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let events = generate_events(&jobs, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(non_stream_body(&model_id, &events)).into_response())
}

async fn anthropic_messages(
    State(st): State<AppState>,
    Json(body): Json<MessagesBody>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    let think =
        think_from_anthropic(body.thinking.as_ref()).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mut messages = Vec::new();
    if let Some(sys) = body.system.as_ref() {
        let text = content_text(Some(sys), &mut Vec::new(), &mut None)
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: text,
            });
        }
    }
    let (msgs, images, audio_pcm) =
        messages_from_body(&body.messages).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    messages.extend(msgs);
    if messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages must be non-empty".into()));
    }
    if !images.is_empty() || audio_pcm.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "multimodal content on /v1/messages is not supported; use /v1/chat/completions".into(),
        ));
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
    let model_id = body
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let jobs = with_engine(&st.pool, &model_id).await?;
    if body.stream.unwrap_or(false) {
        let events = generate_events(&jobs, req)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let sse = anthropic_stream_chunks(&model_id, &events);
        return Ok(Sse::new(stream::iter(
            sse.into_iter().map(Ok::<_, std::convert::Infallible>),
        ))
        .keep_alive(KeepAlive::default())
        .into_response());
    }
    let events = generate_events(&jobs, req)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(anthropic_message_body(&model_id, &events)).into_response())
}

#[derive(Debug, Deserialize)]
struct EmbeddingsBody {
    model: Option<String>,
    input: EmbeddingsInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

async fn embeddings(
    State(st): State<AppState>,
    Json(body): Json<EmbeddingsBody>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    let inputs = match body.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    if inputs.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "input must be non-empty".into()));
    }
    let model_id = body
        .model
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| st.default_id.clone());
    let jobs = with_engine(&st.pool, &model_id).await?;
    let mut data = Vec::new();
    let mut total_tokens = 0u32;
    for (index, text) in inputs.iter().enumerate() {
        let vec = embed_vector(&jobs, text)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        total_tokens += text.split_whitespace().count().max(1) as u32;
        data.push(json!({
            "object": "embedding",
            "embedding": vec,
            "index": index
        }));
    }
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    })))
}

async fn audio_transcriptions(
    State(st): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, String)> {
    let _permit = acquire_parallel(&st).await?;
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut whisper_model = "base".to_owned();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("file") => {
                file_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                        .to_vec(),
                );
            }
            Some("model") => {
                whisper_model = field
                    .text()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            }
            _ => {}
        }
    }
    let bytes = file_bytes.ok_or((
        StatusCode::BAD_REQUEST,
        "multipart field `file` is required".into(),
    ))?;
    let kind = runa_media::WhisperKind::parse(&whisper_model)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if runa_media::ensure_whisper_model(kind, false).is_err() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "whisper weights missing for {whisper_model}; run `runa media transcribe` once to pull, or set RUNA_WHISPER=1 in dev"
            ),
        ));
    }
    let path = write_temp_file(&bytes, "audio")?;
    let transcript = runa_media::transcribe_file(
        &path,
        &runa_media::AsrOptions {
            kind,
            pull: false,
            ..runa_media::AsrOptions::default()
        },
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let _ = std::fs::remove_file(&path);
    Ok(Json(json!({"text": transcript.text})))
}

fn write_temp_file(bytes: &[u8], prefix: &str) -> Result<PathBuf, (StatusCode, String)> {
    let path = std::env::temp_dir().join(format!(
        "runa-{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&path, bytes).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(path)
}

async fn generate_events(
    jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    req: GenerateRequest,
) -> Result<Vec<GenEvent>, String> {
    let (resp, rx) = oneshot::channel();
    jobs.send(EngineJob::Generate { req, resp })
        .map_err(|_| "engine thread stopped".to_string())?;
    rx.await.map_err(|e| e.to_string())?
}

async fn embed_vector(
    jobs: &Arc<std::sync::mpsc::Sender<EngineJob>>,
    input: &str,
) -> Result<Vec<f32>, String> {
    let (resp, rx) = oneshot::channel();
    jobs.send(EngineJob::Embed {
        input: input.to_owned(),
        resp,
    })
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
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
    image_url: Option<ImageUrlPart>,
    input_audio: Option<InputAudioPart>,
}

#[derive(Debug, Deserialize)]
struct ImageUrlPart {
    url: String,
}

#[derive(Debug, Deserialize)]
struct InputAudioPart {
    data: String,
    format: Option<String>,
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
    let idx = 0u32;
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

fn messages_from_body(
    messages: &[IncomingMessage],
) -> Result<(Vec<ChatMessage>, Vec<VisionFrame>, Option<Vec<f32>>), String> {
    let mut out = Vec::new();
    let mut images = Vec::new();
    let mut audio_pcm: Option<Vec<f32>> = None;
    for m in messages {
        let text = content_text(m.content.as_ref(), &mut images, &mut audio_pcm)?;
        if !text.is_empty() || m.role == "assistant" {
            out.push(ChatMessage {
                role: m.role.clone(),
                content: text,
            });
        }
    }
    if !images.is_empty() && audio_pcm.is_some() {
        return Err("cannot mix image and audio in one request".into());
    }
    if !images.is_empty() && !cfg!(feature = "mtmd") {
        return Err(
            "image_url / vision content requires rebuilding runa with --features mtmd".into(),
        );
    }
    Ok((out, images, audio_pcm))
}

fn content_text(
    c: Option<&IncomingContent>,
    images: &mut Vec<VisionFrame>,
    audio_pcm: &mut Option<Vec<f32>>,
) -> Result<String, String> {
    match c {
        Some(IncomingContent::Text(s)) => Ok(s.clone()),
        Some(IncomingContent::Parts(parts)) => {
            let mut text = String::new();
            for p in parts {
                let kind = p.kind.as_deref().unwrap_or("text");
                match kind {
                    "text" => {
                        if let Some(t) = p.text.as_deref() {
                            text.push_str(t);
                        }
                    }
                    "image_url" => {
                        let url = p
                            .image_url
                            .as_ref()
                            .ok_or("image_url part missing image_url field")?
                            .url
                            .clone();
                        let path = image_url_to_path(&url)?;
                        images.push(VisionFrame {
                            t_sec: None,
                            source: VisionSource::Path(path),
                        });
                    }
                    "input_audio" => {
                        let part = p
                            .input_audio
                            .as_ref()
                            .ok_or("input_audio part missing input_audio field")?;
                        let pcm = decode_input_audio(part)?;
                        if audio_pcm.is_some() {
                            return Err("only one input_audio part per request".into());
                        }
                        *audio_pcm = Some(pcm);
                    }
                    other => return Err(format!("unsupported content part type: {other}")),
                }
            }
            Ok(text)
        }
        None => Ok(String::new()),
    }
}

fn image_url_to_path(url: &str) -> Result<PathBuf, String> {
    if url.starts_with("data:") {
        let rest = url.strip_prefix("data:").ok_or("malformed data URI")?;
        let (_meta, b64) = rest
            .split_once(',')
            .ok_or("malformed data URI: missing comma")?;
        let bytes = b64_decode(b64.trim())?;
        return write_temp_file(&bytes, "img").map_err(|(_, e)| e);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err("remote image_url is not supported; use a data: URI".into());
    }
    let p = PathBuf::from(url);
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("image_url path not found: {url}"))
    }
}

fn decode_input_audio(part: &InputAudioPart) -> Result<Vec<f32>, String> {
    let bytes = b64_decode(part.data.trim())?;
    let ext = match part.format.as_deref() {
        Some("wav") | None => "wav",
        Some("mp3") => "mp3",
        other => return Err(format!("unsupported input_audio format: {:?}", other)),
    };
    let path = write_temp_file(&bytes, "in-audio").map_err(|(_, e)| e)?;
    let path = {
        let named = path.with_extension(ext);
        std::fs::rename(&path, &named).map_err(|e| e.to_string())?;
        named
    };
    let decoded = runa_media::decode_audio(&path).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&path);
    Ok(decoded.samples)
}

fn b64_decode(input: &str) -> Result<Vec<u8>, String> {
    const T: &[u8; 256] = &{
        let mut t = [255u8; 256];
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[chars[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    let mut n = 0usize;
    for &b in input.as_bytes() {
        if b == b'=' {
            break;
        }
        if b.is_ascii_whitespace() {
            continue;
        }
        let v = T[b as usize];
        if v == 255 {
            return Err("invalid base64".into());
        }
        buf[n] = v;
        n += 1;
        if n == 4 {
            out.push((buf[0] << 2) | (buf[1] >> 4));
            out.push((buf[1] << 4) | (buf[2] >> 2));
            out.push((buf[2] << 6) | buf[3]);
            n = 0;
        }
    }
    if n == 2 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
    } else if n == 3 {
        out.push((buf[0] << 2) | (buf[1] >> 4));
        out.push((buf[1] << 4) | (buf[2] >> 2));
    }
    Ok(out)
}

/// Build model id + path list from CLI paths (stem ids; suffix on collision).
pub(crate) fn model_specs_from_paths(
    paths: Vec<PathBuf>,
) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for path in paths {
        let resolved = if path.is_file() {
            path
        } else {
            crate::resolve_model(path.to_str().unwrap_or(""))?
        };
        let stem = resolved
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("runa")
            .to_owned();
        let n = counts.entry(stem.clone()).or_insert(0);
        let id = if *n == 0 {
            stem.clone()
        } else {
            format!("{stem}-{}", *n)
        };
        *n += 1;
        out.push((id, resolved));
    }
    Ok(out)
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
        let (out, imgs, aud) = messages_from_body(&msgs).unwrap();
        assert_eq!(out[0].content, "hi there");
        assert!(imgs.is_empty());
        assert!(aud.is_none());
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

    #[test]
    fn b64_roundtrip() {
        let raw = b"hello";
        let enc = runa_media::wav_base64(raw);
        assert_eq!(b64_decode(&enc).unwrap(), raw);
    }
}
