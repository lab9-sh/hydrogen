//! Serde shapes for Anthropic's Messages HTTP API.
//!
//! Kept private so public types stay free of Anthropic field names and
//! versioning quirks.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    pub cache_control: CacheControl,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Anthropic `tool_choice` object (`type` + optional name / parallel flag).
#[derive(Debug, Clone, Serialize)]
pub struct WireToolChoice {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

/// Prompt-cache marker; ephemeral matches Anthropic's default caching tier.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
}

impl CacheControl {
    pub const EPHEMERAL: Self = Self { kind: "ephemeral" };
}

#[derive(Debug, Serialize)]
pub struct OutputConfig {
    pub effort: Effort,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Serialize)]
pub struct WireMessage {
    pub role: &'static str,
    pub content: Vec<Value>,
}

/// Function tools and Anthropic-hosted server tools share the `tools` array.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum WireTool {
    Function {
        name: String,
        description: String,
        input_schema: Value,
    },
    /// Hosted web search (`web_search_20250305` + name `web_search`).
    WebSearch {
        #[serde(rename = "type")]
        kind: &'static str,
        name: &'static str,
    },
}

impl WireTool {
    pub fn function(name: String, description: String, input_schema: Value) -> Self {
        Self::Function {
            name,
            description,
            input_schema,
        }
    }

    pub fn web_search() -> Self {
        Self::WebSearch {
            kind: "web_search_20250305",
            name: "web_search",
        }
    }
}

/// Anthropic has two thinking control planes depending on model generation.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Thinking {
    Adaptive { display: Display },
    Enabled { budget_tokens: u32 },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Display {
    Summarized,
}

#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub content: Vec<Value>,
    pub stop_reason: Option<String>,
    pub usage: WireUsage,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct WireUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: StreamMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: Value,
    },
    ContentBlockDelta {
        index: usize,
        delta: Delta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: usize,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        #[serde(default)]
        usage: Option<WireUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ApiError,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct StreamMessageStart {
    pub usage: WireUsage,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    /// Signature fragments must be reassembled for multi-turn thinking.
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<String>,
}
