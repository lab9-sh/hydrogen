//! Anthropic Messages API adapter.

mod adapter;
mod sse;
mod wire;

pub(crate) use adapter::Adapter;

use crate::Error;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Required by Anthropic on every request; pins request/response semantics.
const API_VERSION: &str = "2023-06-01";

/// Connection settings for Anthropic. Optional `http_client` lets callers share
/// pools, proxies, or timeouts with the rest of the process.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub http_client: Option<reqwest::Client>,
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            http_client: None,
        }
    }

    pub fn from_env() -> Result<Self, Error> {
        std::env::var("ANTHROPIC_API_KEY")
            .map(Self::new)
            .map_err(|_| Error::Auth)
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}
