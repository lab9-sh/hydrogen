use serde::{Deserialize, Serialize};

use super::{Message, ProviderKind};

/// Completed assistant turn with normalized stop/usage metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub message: Message,
    pub stop_reason: StopReason,
    pub usage: Usage,
    /// Which backend produced this turn; used to pin the conversation.
    pub(crate) provider: ProviderKind,
}

/// Why generation stopped, collapsed from each provider's vocabulary so
/// tool-loop and token-limit handling stay provider-agnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Refusal,
    Other(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
