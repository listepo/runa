//! Shared engine pool (P9.1): `ModelPool`, `EngineJob`, `spawn_engine`,
//! and the P8.8 `Warmup`, moved verbatim from `serve.rs` so `serve` and
//! `daemon` share one loader. No behavior change.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use runa_core::BackendKind;
use runa_engine::{GenEvent, GenerateRequest, LoadConfig, Placement};
use runa_fit::{
    Descriptor, FitConfig, HwSpec, PlannerConfig, Reader, check_fit, read_local_prefix,
};
use tokio::sync::oneshot;

/// Startup load of the default model, read by `/health` (P8.8) and by the
/// daemon's ready line.
pub(crate) struct Warmup {
    pub model: String,
    /// Per mille, written by llama.cpp's load callback.
    pub progress: Arc<AtomicU32>,
    /// `None` while loading.
    pub state: Mutex<Option<Result<(), String>>>,
}

impl Warmup {
    pub fn new(model: String, progress: Arc<AtomicU32>) -> Self {
        Warmup {
            model,
            progress,
            state: Mutex::new(None),
        }
    }

    pub fn done(&self) -> Option<Result<(), String>> {
        lock(&self.state).clone()
    }
}

pub(crate) fn warm_up(pool: &Mutex<ModelPool>, warm: &Warmup, tag: &str) {
    let t = Instant::now();
    let r = catch_job(|| lock(pool).ensure_engine(&warm.model).map(drop));
    match &r {
        Ok(()) => eprintln!(
            "{tag}: {} ready in {:.1}s",
            warm.model,
            t.elapsed().as_secs_f64()
        ),
        Err(e) => eprintln!("{tag}: loading {} failed: {e}", warm.model),
    }
    *lock(&warm.state) = Some(r);
}

/// `<tag>: loading <id> N%` in 10% steps while the warm-up runs.
pub(crate) async fn report_progress(warm: Arc<Warmup>, tag: &'static str) {
    let mut shown = 0;
    while warm.done().is_none() {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let step = warm.progress.load(Ordering::Relaxed) / 100;
        if step > shown && step < 10 {
            shown = step;
            eprintln!("{tag}: loading {} {}%", warm.model, step * 10);
        }
    }
}

/// A poisoned lock still holds consistent pool state (every mutation is a
/// single insert/remove), so recover it instead of failing every request.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

/// Run one engine/pool step; a panic becomes an error, not a dead thread.
pub(crate) fn catch_job<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|p| Err(format!("internal error: {}", panic_text(&*p))))
}

pub(crate) enum EngineJob {
    Generate {
        req: Box<GenerateRequest>,
        resp: oneshot::Sender<Result<Vec<GenEvent>, String>>,
    },
    Embed {
        input: String,
        resp: oneshot::Sender<Result<Vec<f32>, String>>,
    },
}

pub(crate) struct ModelPool {
    specs: HashMap<String, PathBuf>,
    order: Vec<String>,
    engines: HashMap<String, Arc<std::sync::mpsc::Sender<EngineJob>>>,
    lru: VecDeque<String>,
    max_loaded: usize,
    /// Requested `--backend` (possibly `Auto`; resolved per model path).
    backend: BackendKind,
    placement_base: Placement,
    mode: String,
    config: LoadConfig,
}

impl ModelPool {
    pub(crate) fn new(
        models: Vec<(String, PathBuf)>,
        backend: BackendKind,
        placement_base: Placement,
        mode: String,
        config: LoadConfig,
        max_loaded: usize,
    ) -> Result<Self, String> {
        let mut specs = HashMap::new();
        let mut order = Vec::new();
        for (id, path) in models {
            if !path.is_file() && !path.is_dir() {
                return Err(format!(
                    "no such model file or directory: {}",
                    path.display()
                ));
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
            backend,
            placement_base,
            mode,
            config,
        })
    }

    pub(crate) fn model_ids(&self) -> &[String] {
        &self.order
    }

    pub(crate) fn resolve_id(&self, model: Option<&str>) -> Result<String, String> {
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

    /// Register one more model (file or directory) under a stem id
    /// (daemon on-demand serving of paths it was not started with).
    /// No-op when the id or an equal path is already known. Returns the id.
    pub(crate) fn insert_spec(&mut self, path: &Path) -> Result<String, String> {
        if !path.is_file() && !path.is_dir() {
            return Err(format!(
                "no such model file or directory: {}",
                path.display()
            ));
        }
        if let Some((id, _)) = self.specs.iter().find(|(_, p)| *p == path) {
            return Ok(id.clone());
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("runa")
            .to_owned();
        let mut id = stem.clone();
        let mut n = 1;
        while self.specs.contains_key(&id) {
            n += 1;
            id = format!("{stem}-{n}");
        }
        self.specs.insert(id.clone(), path.to_owned());
        self.order.push(id.clone());
        Ok(id)
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
                &self.config.loras,
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

    fn fit_check_no_fit(path: &Path, ctx: u32, lora_bytes: u64) -> Result<(), String> {
        let header = read_local_prefix(path).map_err(|e| e.to_string())?;
        let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
        let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
        let (vram, _) = crate::vram_bytes()?;
        let planner = PlannerConfig {
            vram_bytes: vram,
            ram_bytes: crate::ram_bytes(),
            ctx_len: u64::from(ctx),
            kv_type: runa_engine::planner_kv_type(None, None).to_owned(),
            lora_bytes,
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

    /// Adapter bytes summed from the configured `--lora` files (P8.5).
    fn lora_bytes(config: &LoadConfig) -> u64 {
        config
            .loras
            .iter()
            .map(|s| runa_fit::mmproj_file_bytes(&s.path))
            .sum()
    }

    pub(crate) fn ensure_engine(
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
        let kind = crate::engine::resolve_requested(self.backend, &path)?;
        // Fit, placement and LoRA are ggml concepts; the mistral backend
        // manages devices and KV itself.
        let (placement, kind) = match kind {
            BackendKind::Gguf => {
                Self::fit_check_no_fit(&path, self.config.n_ctx, Self::lora_bytes(&self.config))?;
                (self.placement_for(&path)?, BackendKind::Gguf)
            }
            BackendKind::Mistral => {
                if !self.config.loras.is_empty() {
                    return Err("--lora needs the gguf backend".into());
                }
                (Placement::cpu(), BackendKind::Mistral)
            }
            BackendKind::Auto => {
                return Err("internal error: backend was not resolved".into());
            }
        };
        let tx = Arc::new(spawn_engine(path, kind, placement, self.config.clone())?);
        self.engines.insert(id.to_owned(), Arc::clone(&tx));
        self.touch_lru(id);
        self.evict_if_needed();
        Ok(tx)
    }
}

pub(crate) fn spawn_engine(
    path: PathBuf,
    kind: BackendKind,
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
        // `runa run` drives the engine on the 8 MiB main thread; match it
        // (the 2 MiB default is tight for llama.cpp's Jinja templates).
        .stack_size(8 << 20)
        .spawn(move || {
            let mut engine =
                match crate::engine::LocalEngine::load(kind, &path, &placement, &config) {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
            let mut used = false;
            while let Ok(job) = rx.recv() {
                match job {
                    EngineJob::Generate { req, resp } => {
                        let out = catch_job(|| {
                            // The ggml backend reuses one context per thread:
                            // drop KV cells between requests (P3.9). The
                            // mistral backend is stateless (no-op there).
                            if used {
                                engine.clear_kv();
                            }
                            used = true;
                            let generation = engine.generate(*req).map_err(|e| e.to_string())?;
                            generation
                                .collect::<Result<Vec<_>, _>>()
                                .map_err(|e| e.to_string())
                        });
                        let _ = resp.send(out);
                    }
                    EngineJob::Embed { input, resp } => {
                        let out = catch_job(|| match &mut engine {
                            crate::engine::LocalEngine::Gguf(loaded) => {
                                loaded.embed(&input).map_err(|e| e.to_string())
                            }
                            #[cfg(feature = "mistralrs")]
                            crate::engine::LocalEngine::Mistral(_) => {
                                Err("mistral backend does not serve /v1/embeddings (gguf only)"
                                    .into())
                            }
                        });
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

/// Resolve `model_id` against the pool and run one generation on its
/// engine thread (blocking load happens on a blocking thread). Shared by
/// the daemon; `serve` keeps its own status-mapped variant.
pub(crate) async fn generate(
    pool: &Arc<Mutex<ModelPool>>,
    model_id: &str,
    req: GenerateRequest,
) -> Result<Vec<GenEvent>, String> {
    let jobs = {
        let pool = Arc::clone(pool);
        let id = model_id.to_owned();
        tokio::task::spawn_blocking(move || lock(&pool).ensure_engine(&id))
            .await
            .map_err(|e| e.to_string())??
    };
    let (resp, rx) = oneshot::channel();
    jobs.send(EngineJob::Generate {
        req: Box::new(req),
        resp,
    })
    .map_err(|_| "engine thread stopped".to_string())?;
    rx.await.map_err(|e| e.to_string())?
}

/// Look up `model` (pool id, or an on-disk path to serve on demand) and
/// return the id to generate with. Inserting a path never unloads models.
pub(crate) fn resolve_or_insert(
    pool: &Mutex<ModelPool>,
    model: Option<&str>,
) -> Result<String, String> {
    let mut pool = lock(pool);
    match pool.resolve_id(model) {
        Ok(id) => Ok(id),
        Err(_) => {
            let raw = model.unwrap_or("").trim();
            let path = Path::new(raw);
            if !raw.is_empty() && path.is_file() {
                pool.insert_spec(path)
            } else {
                Err(format!("model {} not found", model.unwrap_or("")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tiny_gguf(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        path
    }

    #[test]
    fn insert_spec_reuses_path_and_suffixes_collisions() {
        let dir = std::env::temp_dir().join(format!(
            "runa-pool-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Same stem in two dirs → `stem`, `stem-2`.
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let a = tiny_gguf(&dir, "m.gguf");
        let b = tiny_gguf(&sub, "m.gguf");
        let pool = Mutex::new(
            ModelPool::new(
                vec![],
                BackendKind::Gguf,
                Placement::cpu(),
                "cpu".into(),
                LoadConfig {
                    n_ctx: 512,
                    ..LoadConfig::default()
                },
                2,
            )
            .unwrap(),
        );
        let id_a = lock(&pool).insert_spec(&a).unwrap();
        assert_eq!(id_a, "m");
        assert_eq!(lock(&pool).insert_spec(&a).unwrap(), "m");
        assert_eq!(lock(&pool).insert_spec(&b).unwrap(), "m-2");
        assert_eq!(resolve_or_insert(&pool, Some("m")).unwrap(), "m");
        assert!(resolve_or_insert(&pool, Some("nope")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn jobs_survive_panics() {
        let r: Result<(), String> = catch_job(|| panic!("boom"));
        assert_eq!(r, Err("internal error: boom".into()));
    }

    #[test]
    fn pool_rejects_missing_files() {
        let err = ModelPool::new(
            vec![("m".into(), PathBuf::from("/no/such/model.gguf"))],
            BackendKind::Gguf,
            Placement::cpu(),
            "cpu".into(),
            LoadConfig::default(),
            1,
        )
        .err()
        .expect("missing file errors");
        assert!(err.contains("no such model file"), "{err}");
    }
}
