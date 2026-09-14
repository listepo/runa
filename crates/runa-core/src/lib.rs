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
