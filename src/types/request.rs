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
