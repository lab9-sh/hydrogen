//! Serde shapes for xAI's Responses-compatible HTTP API.
//!
//! Mirrors OpenAI Responses closely; kept as its own module so endpoint or
//! field drift does not force a shared type.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    pub reasoning: Reasoning,
    /// Keep conversation state client-side; we resend full history each turn.
    pub store: bool,
    pub include: Vec<&'static str>,
    pub prompt_cache_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct Reasoning {
    pub effort: Effort,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Low,
    Medium,
    High,
}

/// Function tools and hosted tools (e.g. `web_search`) share the `tools` array.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum WireTool {
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
        description: String,
        parameters: Value,
    },
    WebSearch {
        #[serde(rename = "type")]
        kind: &'static str,
    },
}

impl WireTool {
    pub fn function(name: String, description: String, parameters: Value) -> Self {
        Self::Function {
            kind: "function",
            name,
            description,
            parameters,
        }
    }

    pub fn web_search() -> Self {
        Self::WebSearch {
            kind: "web_search",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ResponsesResponse {
    pub output: Vec<Value>,
    pub status: String,
    #[serde(default)]
    pub usage: WireUsage,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct WireUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    /// Subset of `input_tokens` served from prompt cache (automatic caching).
    #[serde(default)]
    pub input_tokens_details: InputTokensDetails,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub kind: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated {
        #[serde(default)]
        #[allow(dead_code)]
        response: Option<Value>,
    },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        #[serde(default)]
        #[allow(dead_code)]
        response: Option<Value>,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: Value,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        #[allow(dead_code)]
        output_index: usize,
        #[allow(dead_code)]
        item: Value,
    },
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        response: ResponsesResponse,
    },
    #[serde(rename = "response.failed")]
    ResponseFailed {
        response: ResponsesResponse,
    },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        response: ResponsesResponse,
    },
    Error {
        error: ApiError,
    },
    #[serde(other)]
    Other,
}
