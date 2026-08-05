//! Provider-agnostic conversation and request types.
//!
//! Kept separate from `*/wire` so application code never depends on a single
//! backend's JSON shape; adapters translate at the edge.

mod content;
mod conversation;
mod request;
mod response;
mod streaming;

pub use content::{
    ContentBlock, OpaquePayload, ReasoningBlock, TextBlock, ToolOutput, ToolResultBlock,
    ToolUseBlock,
};
pub use conversation::{Conversation, Message, ProviderKind, Role};
pub use request::{RequestOptions, ThinkingEffort, ToolChoice, ToolDef};
pub use response::{Response, StopReason, Usage};
pub use streaming::{Event, EventStream};
