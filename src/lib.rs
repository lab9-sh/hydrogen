//! Unified multi-provider LLM client.
//!
//! One conversation model and request API across Anthropic, OpenAI, and xAI so
//! application code does not fork on provider wire formats. Adapters own the
//! translation; callers stay on shared types.

mod anthropic;
mod client;
mod error;
mod openai;
mod xai;
pub mod types;

pub use anthropic::AnthropicConfig;
pub use client::Client;
pub use error::Error;
pub use openai::OpenAiConfig;
pub use xai::XaiConfig;
pub use types::{
    ContentBlock, Conversation, Event, EventStream, Message, ProviderKind, RequestOptions,
    Response, Role, StopReason, ThinkingEffort, ToolDef, ToolOutput, Usage,
};
