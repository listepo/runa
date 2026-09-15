//! runa-core — shared types: `Backend` trait, `Request`/`Event`,
//! `ThinkConfig`, `Mode`, errors (plan §5).
//!
//! Types land here; engine wiring is P2, thinking parse is P3.1.

mod reason;
mod think;

pub use reason::{ReasonFamily, ReasonPiece, ReasoningParser, parse_stream};
pub use think::{
    BUDGET_MESSAGE, BudgetClock, CLOSE_LOGIT_BIAS, DEFAULT_GRACE, EFFORT_BUDGET_HIGH,
    EFFORT_BUDGET_LOW, EFFORT_BUDGET_MEDIUM, Effort, ForceKind, ThinkConfig, ThinkMode,
    ThinkOverrides, effort_system_hint, parse_budget,
};

/// One function call from a model's reply, local or cloud (P8.2 / P8.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Call id, echoed back by the `tool` message that answers it.
    pub id: String,
    /// Function name from the request's `tools`.
    pub name: String,
    /// Arguments as JSON text.
    pub arguments: String,
}
