use serde::{Deserialize, Serialize};

/// Per-request knobs shared across providers. Adapters map these onto each
/// backend's fields (e.g. thinking effort → budget vs adaptive).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestOptions {
    pub model: String,
    pub system: Option<String>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub thinking: Option<ThinkingEffort>,
    /// When true, adapters attach that provider's hosted web-search tool
    /// (Anthropic server tool; OpenAI/xAI Responses `web_search`). Default off.
    /// No provider-specific search knobs or citation surface.
    #[serde(default)]
    pub web_search: bool,
    /// Whether / which tool the model must call. Default [`ToolChoice::Auto`].
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// `None` leaves the provider default untouched. `Some(false)` disables
    /// parallel tool calls when the backend supports it.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Place the prompt-cache breakpoint this many messages from the end,
    /// instead of implicitly at the end of the whole prompt.
    ///
    /// `Some(1)` marks the second-to-last message. That is what a rewriting
    /// loop needs: if the tail message is rebuilt every turn, an implicit
    /// end-of-prompt breakpoint caches a prefix that never matches again, so
    /// every turn is a full cache miss *and* a full cache write. Marking the
    /// last message that will not change keeps the cached prefix growing
    /// monotonically.
    ///
    /// Anthropic only; ignored by backends that cache automatically.
    #[serde(default)]
    pub cache_breakpoint_from_end: Option<usize>,
}

/// Portable control over whether the model may / must / must-not call tools.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides (current default behavior).
    #[default]
    Auto,
    /// Must call some tool.
    Required,
    /// Must call this specific tool.
    Tool(String),
    /// Tools may be visible but not callable.
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Relative reasoning budget. Exact token budgets / modes differ by provider
/// and model; this is the portable control surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingEffort {
    Low,
    Medium,
    High,
}
