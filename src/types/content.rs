use serde::{Deserialize, Serialize};

use super::ProviderKind;

/// Provider wire JSON we do not interpret, kept so multi-turn requests can
/// echo fields (ids, signatures, encrypted reasoning) the backend requires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpaquePayload(pub(crate) serde_json::Value);

/// One unit of message content. Variants cover the shared surface; anything
/// provider-private is carried as [`Reasoning`](ContentBlock::Reasoning) or
/// in `extras` so it still round-trips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Reasoning(ReasoningBlock),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
    /// Original wire item when present (e.g. OpenAI message ids/status).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extras: Option<OpaquePayload>,
}

impl TextBlock {
    /// Construct a plain text block. `extras` is empty (provider wire echo is
    /// only set when parsing a response).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            extras: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
    /// Original wire item when present (call_id, status, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) extras: Option<OpaquePayload>,
}

impl ToolUseBlock {
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
            extras: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub id: String,
    pub output: ToolOutput,
}

impl ToolResultBlock {
    /// Construct a tool-result block for a caller-built user turn.
    pub fn new(id: impl Into<String>, output: ToolOutput) -> Self {
        Self {
            id: id.into(),
            output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutput {
    Text(String),
    Json(serde_json::Value),
    Error(String),
}

/// Thinking/reasoning from the model. Payload stays opaque because each
/// provider signs or encrypts it and rejects hand-edited substitutes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningBlock {
    pub(crate) provider: ProviderKind,
    pub(crate) payload: OpaquePayload,
    pub(crate) summary: Option<String>,
}

impl ReasoningBlock {
    /// Human-readable summary when the provider exposes one; never the full
    /// private chain-of-thought.
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }
}
