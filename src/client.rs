use crate::anthropic::{self, AnthropicConfig};
use crate::openai::{self, OpenAiConfig};
use crate::xai::{self, XaiConfig};
use crate::types::{Conversation, EventStream, ProviderKind, RequestOptions, Response};
use crate::Error;

/// Provider-backed client. Constructed for one backend so request routing and
/// auth stay fixed for the lifetime of the handle.
pub struct Client {
    inner: Inner,
}

enum Inner {
    Anthropic(anthropic::Adapter),
    Xai(xai::Adapter),
    OpenAi(openai::Adapter),
}

impl Client {
    pub fn anthropic(cfg: AnthropicConfig) -> Self {
        Self {
            inner: Inner::Anthropic(anthropic::Adapter::new(cfg)),
        }
    }

    pub fn xai(cfg: XaiConfig) -> Self {
        Self {
            inner: Inner::Xai(xai::Adapter::new(cfg)),
        }
    }

    pub fn openai(cfg: OpenAiConfig) -> Self {
        Self {
            inner: Inner::OpenAi(openai::Adapter::new(cfg)),
        }
    }

    fn kind(&self) -> ProviderKind {
        match &self.inner {
            Inner::Anthropic(_) => ProviderKind::Anthropic,
            Inner::Xai(_) => ProviderKind::Xai,
            Inner::OpenAi(_) => ProviderKind::OpenAi,
        }
    }

    /// Reject cross-provider use once a conversation has provider-specific
    /// content (reasoning signatures, encrypted blobs) that other backends
    /// cannot accept.
    fn check_pinning(&self, conv: &Conversation) -> Result<(), Error> {
        match conv.provider() {
            Some(p) if p != self.kind() => Err(Error::ProviderMismatch {
                conversation: p,
                client: self.kind(),
            }),
            _ => Ok(()),
        }
    }

    pub async fn send(
        &self,
        conv: &Conversation,
        opts: &RequestOptions,
    ) -> Result<Response, Error> {
        self.check_pinning(conv)?;
        match &self.inner {
            Inner::Anthropic(a) => a.send(conv, opts).await,
            Inner::Xai(a) => a.send(conv, opts).await,
            Inner::OpenAi(a) => a.send(conv, opts).await,
        }
    }

    pub async fn stream(
        &self,
        conv: &Conversation,
        opts: &RequestOptions,
    ) -> Result<EventStream, Error> {
        self.check_pinning(conv)?;
        match &self.inner {
            Inner::Anthropic(a) => a.stream(conv, opts).await,
            Inner::Xai(a) => a.stream(conv, opts).await,
            Inner::OpenAi(a) => a.stream(conv, opts).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Role, StopReason, TextBlock, Usage};

    fn pinned(provider: ProviderKind) -> Conversation {
        let mut conv = Conversation::new();
        conv.push_response(Response {
            message: Message {
                role: Role::Assistant,
                content: vec![crate::types::ContentBlock::Text(TextBlock {
                    text: "hi".into(),
                    extras: None,
                })],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            provider,
        });
        conv
    }

    /// Cross-provider reuse of a pinned conversation would resubmit foreign
    /// reasoning blobs and fail deep in the HTTP layer.
    #[test]
    fn check_pinning_rejects_mismatched_provider() {
        let client = Client::openai(OpenAiConfig::new("sk-test"));
        let conv = pinned(ProviderKind::Anthropic);
        let err = client.check_pinning(&conv).unwrap_err();
        match err {
            Error::ProviderMismatch {
                conversation,
                client: c,
            } => {
                assert_eq!(conversation, ProviderKind::Anthropic);
                assert_eq!(c, ProviderKind::OpenAi);
            }
            other => panic!("expected ProviderMismatch, got {other:?}"),
        }
    }

    #[test]
    fn check_pinning_allows_unpinned_and_matching() {
        let client = Client::xai(XaiConfig::new("xai-test"));
        assert!(client.check_pinning(&Conversation::new()).is_ok());
        assert!(client.check_pinning(&pinned(ProviderKind::Xai)).is_ok());
    }
}
