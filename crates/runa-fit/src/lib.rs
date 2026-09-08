//! `runa-fit`: GGUF header reader, hardware probe, estimator, planner,
//! calibration DB (plan P1).
//!
//! The reader core (`gguf`) is zero-copy over `&[u8]` and `no_std`-friendly:
//! it only needs `core`/`alloc` and does no I/O itself. Higher-level modules
//! (remote header fetch P1.2, hardware probe P1.6, estimator/planner
//! P1.8–P1.9, calibration DB P1.10) build on top of it in later tasks.

pub mod compute;
pub mod descriptor;
pub mod ggml_types;
pub mod gguf;
pub mod kv;
pub mod planner;
pub mod speed;
pub mod verdict;

pub use compute::estimate_compute;
pub use descriptor::Descriptor;
pub use ggml_types::{TypeInfo, n_elements, tensor_bytes, type_info};
pub use gguf::{DataType, GGUF_MAGIC, ReadError, Reader, TensorInfo, Value};
pub use kv::{KvEstimate, estimate_kv};
pub use planner::{PlacementPlan, PlannerConfig, plan_placement};
pub use speed::{HwSpec, SpeedEstimate, active_weight_bytes, estimate_speed_single};
pub use verdict::{FitConfig, FitReport, Verdict, check_fit, format_report};
