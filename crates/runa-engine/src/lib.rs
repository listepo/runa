//! runa-engine — llama-cpp-2 wrapper: model load with `Placement`,
//! context params, streaming sampling loop, mtmd (audio/image/video),
//! prompt-cache state save (plan §2).
//!
//! P2.1 delivers [`placement`] (cpu/gpu/hybrid) + [`load`] (params,
//! verdict line, loaded model). Sampling loop (P2.2), prompt cache (P2.8)
//! and multi-GPU devices (P2.9) extend this skeleton.
//! Version pin for llama-cpp-2: docs/versions.md (0.1.133 → llama.cpp b7709).

pub mod embed;
pub mod generate;
pub mod load;
pub mod lora;
mod media;
mod ngram;
pub mod placement;
pub mod prompt_cache;
pub mod sampling;
mod structured;
mod vision;

pub use generate::{
    ChatMessage, GenEvent, GenerateRequest, Generation, StopReason, ToolCall, Usage,
};
pub use load::{EngineError, KvKind, LoadConfig, LoadedModel, load, planner_kv_type};
pub use lora::{DEFAULT_LORA_SCALE, LoraSpec, parse_lora_spec};
pub use ngram::{NgramCache, Speculative, argmax_i32};
pub use placement::{
    FFN_EXPS_REGEX, Mode, Placement, cpu_moe_patterns, parse_device_list, parse_rpc_list,
    parse_tensor_split,
};
pub use prompt_cache::PromptCache;
pub use sampling::SamplingConfig;
pub use structured::schema_to_grammar;
pub use vision::{VisionFrame, VisionSource, format_vision_user_text};

#[cfg(test)]
mod native_feature_tests {
    #[test]
    fn portable_default_build_is_not_native() {
        assert!(
            !cfg!(feature = "native"),
            "CI / release must use the default feature set so ggml runtime-dispatches AVX-512"
        );
    }
}
