use std::time::Duration;

use crate::types::ProviderKind;

/// Failures callers may want to branch on (auth vs rate limit vs mismatch)
/// rather than a single opaque transport error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP {status}: {message}")]
    Http { status: u16, message: String },

    #[error("rate limited (retry after {retry_after:?})")]
    RateLimited { retry_after: Option<Duration> },

    #[error("authentication failed")]
    Auth,

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Conversation already holds content only one backend can round-trip.
    #[error("conversation is pinned to {conversation:?} but client is {client:?}")]
    ProviderMismatch {
        conversation: ProviderKind,
        client: ProviderKind,
    },

    #[error("deserialize error: {0}")]
    Deserialize(#[from] serde_json::Error),

    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),
}
