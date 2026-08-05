use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{ContentBlock, Response, TextBlock, ToolOutput, ToolResultBlock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

/// Which backend produced (or may accept) a conversation's assistant turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    Xai,
    OpenAi,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// Ordered transcript plus soft provider affinity.
///
/// After the first successful assistant response the conversation is pinned so
/// later turns keep using the same wire semantics (tool ids, reasoning blobs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    messages: Vec<Message>,
    provider: Option<ProviderKind>,
    /// Stable across turns so prompt-caching backends can reuse prefixes.
    #[serde(default = "new_cache_key")]
    cache_key: String,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            provider: None,
            cache_key: new_cache_key(),
        }
    }
}

fn new_cache_key() -> String {
    // Time + counter: unique per process without requiring an external RNG.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("lab9-{t:x}-{n:x}")
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild from a message list and optional provider pin.
    ///
    /// Assigns a fresh `cache_key` so restored transcripts do not collide with
    /// an in-flight conversation's prompt-cache entry.
    pub fn from_parts(messages: Vec<Message>, provider: Option<ProviderKind>) -> Self {
        Self {
            messages,
            provider,
            cache_key: new_cache_key(),
        }
    }

    pub(crate) fn cache_key(&self) -> &str {
        &self.cache_key
    }

    /// Mutable access to the transcript for in-place rewriting (e.g. demoting
    /// a fat state block to a thin stub).
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text(TextBlock::new(text))],
        });
    }

    /// Append a caller-constructed turn. Does not change the provider pin.
    pub fn push_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Append a tool result. Coalesces into the last user message when it is
    /// already tool-results-only so multi-tool turns match provider layouts
    /// that expect one user turn with several results.
    pub fn push_tool_result(&mut self, id: impl Into<String>, output: ToolOutput) {
        let block = ContentBlock::ToolResult(ToolResultBlock::new(id, output));
        match self.messages.last_mut() {
            Some(m)
                if m.role == Role::User
                    && m.content
                        .iter()
                        .all(|b| matches!(b, ContentBlock::ToolResult(_))) =>
            {
                m.content.push(block);
            }
            _ => self.messages.push(Message {
                role: Role::User,
                content: vec![block],
            }),
        }
    }

    /// Record an assistant turn and pin the conversation to that provider.
    pub fn push_response(&mut self, resp: Response) -> &Message {
        self.provider = Some(resp.provider);
        self.messages.push(resp.message);
        self.messages.last().expect("just pushed")
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn provider(&self) -> Option<ProviderKind> {
        self.provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StopReason, Usage};

    fn assistant_response(provider: ProviderKind, text: &str) -> Response {
        Response {
            message: Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text(TextBlock {
                    text: text.into(),
                    extras: None,
                })],
            },
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            provider,
        }
    }

    /// Multi-tool turns must land as one user message with several tool_result
    /// blocks; providers reject (or mis-order) interleaved user turns.
    #[test]
    fn push_tool_result_coalesces_into_tool_only_user_turn() {
        let mut conv = Conversation::new();
        conv.push_tool_result("call_1", ToolOutput::Text("a".into()));
        conv.push_tool_result("call_2", ToolOutput::Json(serde_json::json!({"ok": true})));

        assert_eq!(conv.messages().len(), 1);
        let msg = &conv.messages()[0];
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.len(), 2);
        match &msg.content[0] {
            ContentBlock::ToolResult(t) => {
                assert_eq!(t.id, "call_1");
                assert_eq!(t.output, ToolOutput::Text("a".into()));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
        match &msg.content[1] {
            ContentBlock::ToolResult(t) => assert_eq!(t.id, "call_2"),
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn push_tool_result_starts_new_turn_after_text_user_message() {
        let mut conv = Conversation::new();
        conv.push_user("hello");
        conv.push_tool_result("call_1", ToolOutput::Text("out".into()));

        assert_eq!(conv.messages().len(), 2);
        assert!(matches!(
            conv.messages()[1].content.as_slice(),
            [ContentBlock::ToolResult(_)]
        ));
    }

    #[test]
    fn push_tool_result_starts_new_turn_after_assistant() {
        let mut conv = Conversation::new();
        conv.push_user("hello");
        conv.push_response(assistant_response(ProviderKind::Anthropic, "hi"));
        conv.push_tool_result("call_1", ToolOutput::Error("boom".into()));

        assert_eq!(conv.messages().len(), 3);
        assert_eq!(conv.messages()[2].role, Role::User);
    }

    #[test]
    fn push_response_pins_provider() {
        let mut conv = Conversation::new();
        assert_eq!(conv.provider(), None);
        conv.push_response(assistant_response(ProviderKind::OpenAi, "ok"));
        assert_eq!(conv.provider(), Some(ProviderKind::OpenAi));
        // Later responses overwrite the pin (same-provider turns in practice).
        conv.push_response(assistant_response(ProviderKind::OpenAi, "again"));
        assert_eq!(conv.provider(), Some(ProviderKind::OpenAi));
        assert_eq!(conv.messages().len(), 2);
    }

    #[test]
    fn cache_key_is_stable_for_conversation_lifetime() {
        let conv = Conversation::new();
        let k1 = conv.cache_key().to_string();
        let k2 = conv.cache_key().to_string();
        assert_eq!(k1, k2);
        assert!(k1.starts_with("lab9-"));
        // Distinct conversations get distinct keys (prompt-cache isolation).
        let other = Conversation::new();
        assert_ne!(conv.cache_key(), other.cache_key());
    }

    /// Demote the last user message in place (tail-state-block pattern).
    #[test]
    fn messages_mut_can_demote_last_user_message() {
        let mut conv = Conversation::new();
        conv.push_user("FAT board state with lots of detail");
        conv.push_response(assistant_response(ProviderKind::Anthropic, "I play D4"));

        // Demote the fat block that was answered.
        if let Some(last) = conv.messages_mut().first_mut() {
            *last = Message {
                role: Role::User,
                content: vec![ContentBlock::Text(TextBlock::new(
                    "Move 1 — you played D4. White replied Q16.",
                ))],
            };
        }

        assert_eq!(conv.messages().len(), 2);
        match &conv.messages()[0].content[0] {
            ContentBlock::Text(t) => {
                assert!(t.text.starts_with("Move 1"));
                assert!(!t.text.contains("FAT board"));
            }
            other => panic!("expected text, got {other:?}"),
        }
        // Provider pin from push_response must survive demotion.
        assert_eq!(conv.provider(), Some(ProviderKind::Anthropic));
    }

    #[test]
    fn push_message_preserves_provider_pin() {
        let mut conv = Conversation::new();
        conv.push_response(assistant_response(ProviderKind::OpenAi, "hi"));
        assert_eq!(conv.provider(), Some(ProviderKind::OpenAi));

        conv.push_message(Message {
            role: Role::User,
            content: vec![ContentBlock::Text(TextBlock::new("next turn"))],
        });

        assert_eq!(conv.provider(), Some(ProviderKind::OpenAi));
        assert_eq!(conv.messages().len(), 2);
        assert_eq!(conv.messages()[1].role, Role::User);
    }

    #[test]
    fn from_parts_preserves_messages_and_pin_with_fresh_cache_key() {
        let mut source = Conversation::new();
        source.push_user("hello");
        source.push_response(assistant_response(ProviderKind::Xai, "world"));
        let source_key = source.cache_key().to_string();
        let messages = source.messages().to_vec();

        let rebuilt = Conversation::from_parts(messages, Some(ProviderKind::Xai));

        assert_eq!(rebuilt.messages().len(), 2);
        assert_eq!(rebuilt.provider(), Some(ProviderKind::Xai));
        assert_ne!(rebuilt.cache_key(), source_key);
        match &rebuilt.messages()[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "hello"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn text_block_and_tool_result_public_constructors() {
        let text = TextBlock::new("board state");
        assert_eq!(text.text, "board state");
        assert!(text.extras.is_none());

        let result = ToolResultBlock::new("call_1", ToolOutput::Text("ok".into()));
        assert_eq!(result.id, "call_1");
        assert_eq!(result.output, ToolOutput::Text("ok".into()));

        let mut conv = Conversation::new();
        conv.push_message(Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text(text),
                ContentBlock::ToolResult(result),
            ],
        });
        assert_eq!(conv.messages()[0].content.len(), 2);
    }
}
