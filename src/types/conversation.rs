use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{ContentBlock, Response, TextBlock, ToolOutput, ToolResultBlock};
use crate::Error;

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
///
/// An optional **volatile** slot marks at most one prior message (typically a
/// fat environment-state user turn) that may later be demoted in place. This
/// is opt-in: chat consumers that only call `push_user` / `push_tool_result`
/// never set the mark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    messages: Vec<Message>,
    provider: Option<ProviderKind>,
    /// Stable across turns so prompt-caching backends can reuse prefixes.
    #[serde(default = "new_cache_key")]
    cache_key: String,
    /// Index of the single demotable message, if any. Not persisted: resumed
    /// sessions are append-only, so a rewrite-capable mark must not survive load.
    #[serde(default, skip_serializing)]
    volatile: Option<usize>,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            provider: None,
            cache_key: new_cache_key(),
            volatile: None,
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

    pub(crate) fn cache_key(&self) -> &str {
        &self.cache_key
    }

    /// Index of the current volatile message, if any.
    ///
    /// Clears a stale mark if the index is no longer valid (e.g. after a
    /// future truncation API). Callers may treat `None` as "no volatile."
    pub fn volatile_index(&self) -> Option<usize> {
        match self.volatile {
            Some(i) if i < self.messages.len() => Some(i),
            Some(_) => None,
            None => None,
        }
    }

    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Text(TextBlock {
                text: text.into(),
                extras: None,
            })],
        });
    }

    /// Append a user text turn and mark it as the volatile state block.
    ///
    /// Errors if a volatile message already exists — call [`demote_volatile`]
    /// or [`rotate_volatile_user`] first.
    pub fn push_volatile_user(&mut self, text: impl Into<String>) -> Result<(), Error> {
        if self.volatile_index().is_some() {
            return Err(Error::InvalidRequest(
                "volatile message already set; demote or rotate before pushing another".into(),
            ));
        }
        // Clear a stale mark so we never leave an invalid index around.
        self.volatile = None;
        self.push_user(text);
        self.volatile = Some(self.messages.len() - 1);
        Ok(())
    }

    /// Replace the volatile message's content with a short stub, preserving
    /// role and message count. Clears the volatile mark.
    ///
    /// User-only ship: the marked message must be user text. Errors if there
    /// is no volatile message or the kind is not demotable as text.
    pub fn demote_volatile(&mut self, stub: impl Into<String>) -> Result<(), Error> {
        let idx = match self.volatile {
            Some(i) if i < self.messages.len() => i,
            Some(_) => {
                self.volatile = None;
                return Err(Error::InvalidRequest(
                    "volatile index is out of range".into(),
                ));
            }
            None => {
                return Err(Error::InvalidRequest(
                    "no volatile message to demote".into(),
                ));
            }
        };

        let msg = &mut self.messages[idx];
        if msg.role != Role::User {
            return Err(Error::InvalidRequest(
                "volatile message is not a user turn".into(),
            ));
        }
        // User-only demotion: require text content (not tool_result).
        let is_text = msg
            .content
            .first()
            .is_some_and(|b| matches!(b, ContentBlock::Text(_)));
        if !is_text {
            return Err(Error::InvalidRequest(
                "volatile message is not user text; kind-preserving tool_result demotion is not shipped yet"
                    .into(),
            ));
        }

        msg.content = vec![ContentBlock::Text(TextBlock {
            text: stub.into(),
            extras: None,
        })];
        self.volatile = None;
        Ok(())
    }

    /// Demote the previous volatile (if any), then push a new text volatile.
    ///
    /// Infallible sugar for the environment loop: a missing mark is a no-op
    /// for the demote step (stub is ignored).
    pub fn rotate_volatile_user(&mut self, stub: impl Into<String>, fat: impl Into<String>) {
        if self.volatile_index().is_some() {
            self.demote_volatile(stub)
                .expect("volatile_index Some implies demote succeeds for user text");
        } else {
            self.volatile = None;
        }
        self.push_volatile_user(fat)
            .expect("mark cleared; push_volatile_user cannot fail on empty mark");
    }

    /// Append a tool result. Coalesces into the last user message when it is
    /// already tool-results-only so multi-tool turns match provider layouts
    /// that expect one user turn with several results.
    pub fn push_tool_result(&mut self, id: impl Into<String>, output: ToolOutput) {
        let block = ContentBlock::ToolResult(ToolResultBlock {
            id: id.into(),
            output,
        });
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

    #[test]
    fn push_user_does_not_set_volatile_mark() {
        let mut conv = Conversation::new();
        conv.push_user("hello");
        assert_eq!(conv.volatile_index(), None);
    }

    #[test]
    fn single_slot_enforcement_second_push_volatile_fails() {
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT 1").unwrap();
        assert_eq!(conv.volatile_index(), Some(0));
        let err = conv.push_volatile_user("FAT 2").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
        assert_eq!(conv.messages().len(), 1);
        assert_eq!(conv.volatile_index(), Some(0));
    }

    #[test]
    fn demote_is_content_only_and_clears_mark() {
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT board …").unwrap();
        conv.push_response(assistant_response(ProviderKind::Anthropic, "ok"));
        let len_before = conv.messages().len();
        let roles_before: Vec<_> = conv.messages().iter().map(|m| m.role).collect();
        let key_before = conv.cache_key().to_string();

        conv.demote_volatile("Move 1 — you played Q16.").unwrap();

        assert_eq!(conv.messages().len(), len_before);
        let roles_after: Vec<_> = conv.messages().iter().map(|m| m.role).collect();
        assert_eq!(roles_after, roles_before);
        assert_eq!(conv.volatile_index(), None);
        assert_eq!(conv.cache_key(), key_before);
        match &conv.messages()[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "Move 1 — you played Q16."),
            other => panic!("expected demoted text, got {other:?}"),
        }
        // Assistant still present above the demoted user turn.
        assert_eq!(conv.messages()[1].role, Role::Assistant);
    }

    #[test]
    fn demote_without_mark_fails() {
        let mut conv = Conversation::new();
        conv.push_user("stable");
        assert!(matches!(
            conv.demote_volatile("stub"),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn rotate_after_response_rewrites_marked_user_not_assistant() {
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT 1").unwrap();
        conv.push_response(assistant_response(ProviderKind::Anthropic, "I play Q16"));
        let key_before = conv.cache_key().to_string();

        conv.rotate_volatile_user("Move 1 — you played Q16.", "FAT 2");

        assert_eq!(conv.messages().len(), 3);
        assert_eq!(conv.volatile_index(), Some(2), "new mark is the new tail");
        assert_eq!(conv.cache_key(), key_before);
        match &conv.messages()[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "Move 1 — you played Q16."),
            other => panic!("expected demoted stub, got {other:?}"),
        }
        assert_eq!(conv.messages()[1].role, Role::Assistant);
        match &conv.messages()[2].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "FAT 2"),
            other => panic!("expected new fat, got {other:?}"),
        }
    }

    #[test]
    fn rotate_without_prior_volatile_only_pushes() {
        let mut conv = Conversation::new();
        conv.rotate_volatile_user("ignored stub", "FAT first");
        assert_eq!(conv.messages().len(), 1);
        assert_eq!(conv.volatile_index(), Some(0));
        match &conv.messages()[0].content[..] {
            [ContentBlock::Text(t)] => assert_eq!(t.text, "FAT first"),
            other => panic!("expected fat text, got {other:?}"),
        }
    }

    #[test]
    fn demote_rejects_tool_result_kind_for_user_only_ship() {
        let mut conv = Conversation::new();
        conv.push_tool_result("t1", ToolOutput::Text("FAT".into()));
        // Manually mark a tool_result as volatile (internal field) — not
        // reachable via public API today, but demote must refuse corruption.
        conv.volatile = Some(0);
        let err = conv.demote_volatile("stub").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
        assert_eq!(conv.volatile_index(), Some(0));
    }

    #[test]
    fn volatile_skipped_on_serde_round_trip() {
        let mut conv = Conversation::new();
        conv.push_volatile_user("FAT").unwrap();
        let json = serde_json::to_string(&conv).unwrap();
        assert!(!json.contains("volatile"));
        let loaded: Conversation = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.volatile_index(), None);
        assert_eq!(loaded.messages().len(), 1);
    }
}
