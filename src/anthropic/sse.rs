//! Stream Anthropic SSE into agnostic [`Event`]s.
//!
//! Anthropic emits block start/delta/stop rather than a final full payload, so
//! we reassemble content here for [`Event::Done`].

use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

use super::{adapter, wire};
use crate::types::{Event, EventStream, Message, ProviderKind, Response, Role, Usage};
use crate::Error;

pub(crate) fn event_stream<S>(bytes: S) -> EventStream
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    // scan + flatten: one SSE frame can yield multiple Events (or an error).
    let stream = bytes
        .eventsource()
        .scan(Assembler::new(), |asm, item| {
            let out: Vec<Result<Event, Error>> = match item {
                Ok(sse) => match serde_json::from_str::<wire::StreamEvent>(&sse.data) {
                    Ok(ev) => match asm.handle(ev) {
                        Ok(events) => events.into_iter().map(Ok).collect(),
                        Err(e) => vec![Err(e)],
                    },
                    Err(e) => vec![Err(Error::Deserialize(e))],
                },
                Err(e) => vec![Err(sse_error(e))],
            };
            futures_util::future::ready(Some(futures_util::stream::iter(out)))
        })
        .flatten();
    EventStream(Box::pin(stream))
}

fn sse_error(e: EventStreamError<reqwest::Error>) -> Error {
    match e {
        EventStreamError::Transport(e) => Error::Transport(e),
        other => Error::Http {
            status: 0,
            message: format!("SSE framing error: {other}"),
        },
    }
}

enum BlockState {
    Text(String),
    Thinking { thinking: String, signature: String },
    ToolUse { id: String, name: String, json: String },
    /// Hosted server tools (e.g. web_search): accumulate `input_json_delta`
    /// like client tool_use, but finish as opaque wire (no ToolUse* events).
    ServerToolUse { id: String, name: String, json: String },
    Raw(Value),
}

/// Accumulates in-flight blocks so the final message matches a non-stream reply.
pub(super) struct Assembler {
    blocks: Vec<BlockState>,
    usage: Usage,
    stop_reason: Option<String>,
}

impl Assembler {
    pub(super) fn new() -> Self {
        Self {
            blocks: Vec::new(),
            usage: Usage::default(),
            stop_reason: None,
        }
    }

    pub(super) fn handle(&mut self, ev: wire::StreamEvent) -> Result<Vec<Event>, Error> {
        use wire::StreamEvent as W;
        Ok(match ev {
            W::MessageStart { message } => {
                self.usage.input_tokens = message.usage.input_tokens;
                self.usage.cache_creation_input_tokens =
                    message.usage.cache_creation_input_tokens;
                self.usage.cache_read_input_tokens = message.usage.cache_read_input_tokens;
                vec![]
            }
            W::ContentBlockStart { index, content_block } => {
                self.start_block(index, content_block)
            }
            W::ContentBlockDelta { index, delta } => self.apply_delta(index, delta),
            W::ContentBlockStop { .. } => vec![Event::BlockEnd],
            W::MessageDelta { delta, usage } => {
                self.stop_reason = delta.stop_reason;
                if let Some(u) = usage {
                    self.usage.output_tokens = u.output_tokens;
                }
                vec![]
            }
            W::MessageStop => vec![Event::Done(self.finish()?)],
            W::Ping | W::Other => vec![],
            W::Error { error } => {
                return Err(Error::Http {
                    status: 0,
                    message: format!("{}: {}", error.kind, error.message),
                })
            }
        })
    }

    fn start_block(&mut self, index: usize, block: Value) -> Vec<Event> {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or_default();
        let (state, events) = match kind {
            "text" => (BlockState::Text(String::new()), vec![]),
            "thinking" => (
                BlockState::Thinking {
                    thinking: String::new(),
                    signature: String::new(),
                },
                vec![],
            ),
            "tool_use" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                let name = block.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
                let ev = Event::ToolUseStart {
                    id: id.clone(),
                    name: name.clone(),
                };
                (BlockState::ToolUse { id, name, json: String::new() }, vec![ev])
            }
            // Official streaming sends empty input at start and fragments via
            // input_json_delta — same as tool_use. Must reassemble for multi-turn.
            "server_tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                (
                    BlockState::ServerToolUse {
                        id,
                        name,
                        json: String::new(),
                    },
                    vec![],
                )
            }
            _ => (BlockState::Raw(block), vec![]),
        };
        // Indices are sparse if earlier blocks were skipped; pad so deltas land
        // on the right slot.
        while self.blocks.len() < index {
            self.blocks.push(BlockState::Raw(Value::Null));
        }
        self.blocks.push(state);
        events
    }

    fn apply_delta(&mut self, index: usize, delta: wire::Delta) -> Vec<Event> {
        use wire::Delta as D;
        let Some(state) = self.blocks.get_mut(index) else {
            return vec![];
        };
        match (state, delta) {
            (BlockState::Text(buf), D::TextDelta { text }) => {
                buf.push_str(&text);
                vec![Event::TextDelta(text)]
            }
            (BlockState::Thinking { thinking, .. }, D::ThinkingDelta { thinking: t }) => {
                thinking.push_str(&t);
                vec![Event::ReasoningDelta(t)]
            }
            // Signature is not shown to users but must be complete for round-trip.
            (BlockState::Thinking { signature, .. }, D::SignatureDelta { signature: s }) => {
                signature.push_str(&s);
                vec![]
            }
            (BlockState::ToolUse { id, json, .. }, D::InputJsonDelta { partial_json }) => {
                json.push_str(&partial_json);
                vec![Event::ToolInputDelta {
                    id: id.clone(),
                    json_fragment: partial_json,
                }]
            }
            // Server tool input is not a client tool call — accumulate silently.
            (BlockState::ServerToolUse { json, .. }, D::InputJsonDelta { partial_json }) => {
                json.push_str(&partial_json);
                vec![]
            }
            _ => vec![],
        }
    }

    fn finish(&mut self) -> Result<Response, Error> {
        let mut content = Vec::with_capacity(self.blocks.len());
        for state in self.blocks.drain(..) {
            let wire_block = match state {
                BlockState::Text(text) => json!({ "type": "text", "text": text }),
                BlockState::Thinking { thinking, signature } => {
                    json!({ "type": "thinking", "thinking": thinking, "signature": signature })
                }
                BlockState::ToolUse { id, name, json: input } => {
                    let input: Value = if input.is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&input)?
                    };
                    json!({ "type": "tool_use", "id": id, "name": name, "input": input })
                }
                BlockState::ServerToolUse { id, name, json: input } => {
                    let input: Value = if input.is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&input)?
                    };
                    json!({
                        "type": "server_tool_use",
                        "id": id,
                        "name": name,
                        "input": input,
                    })
                }
                BlockState::Raw(Value::Null) => continue,
                BlockState::Raw(v) => v,
            };
            content.push(adapter::wire_block_to_agnostic(wire_block)?);
        }
        Ok(Response {
            message: Message {
                role: Role::Assistant,
                content,
            },
            stop_reason: adapter::map_stop_reason(self.stop_reason.as_deref()),
            usage: self.usage,
            provider: ProviderKind::Anthropic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, StopReason};
    use serde_json::json;

    fn feed(asm: &mut Assembler, events: Vec<wire::StreamEvent>) -> Vec<Event> {
        let mut out = Vec::new();
        for ev in events {
            out.extend(asm.handle(ev).unwrap());
        }
        out
    }

    /// Signature fragments arrive separately from thinking text. Incomplete
    /// reassembly only fails on the *next* turn with an opaque API 400.
    #[test]
    fn assembles_thinking_signature_and_text_into_done() {
        let mut asm = Assembler::new();
        let events = feed(
            &mut asm,
            vec![
                wire::StreamEvent::MessageStart {
                    message: wire::StreamMessageStart {
                        usage: wire::WireUsage {
                            input_tokens: 12,
                            output_tokens: 0,
                            ..Default::default()
                        },
                    },
                },
                wire::StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: json!({ "type": "thinking" }),
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::ThinkingDelta {
                        thinking: "step ".into(),
                    },
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::ThinkingDelta {
                        thinking: "one".into(),
                    },
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::SignatureDelta {
                        signature: "sig-".into(),
                    },
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::SignatureDelta {
                        signature: "final".into(),
                    },
                },
                wire::StreamEvent::ContentBlockStop { index: 0 },
                wire::StreamEvent::ContentBlockStart {
                    index: 1,
                    content_block: json!({ "type": "text" }),
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 1,
                    delta: wire::Delta::TextDelta {
                        text: "hello".into(),
                    },
                },
                wire::StreamEvent::ContentBlockStop { index: 1 },
                wire::StreamEvent::MessageDelta {
                    delta: wire::MessageDeltaBody {
                        stop_reason: Some("end_turn".into()),
                    },
                    usage: Some(wire::WireUsage {
                        input_tokens: 0,
                        output_tokens: 9,
                        ..Default::default()
                    }),
                },
                wire::StreamEvent::MessageStop,
            ],
        );

        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::TextDelta(t) => Some(t.as_str()),
                Event::ReasoningDelta(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, ["step ", "one", "hello"]);

        let Event::Done(resp) = events.last().unwrap() else {
            panic!("expected Done");
        };
        assert_eq!(resp.provider, ProviderKind::Anthropic);
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.input_tokens, 12);
        assert_eq!(resp.usage.output_tokens, 9);
        assert_eq!(resp.message.content.len(), 2);

        match &resp.message.content[0] {
            ContentBlock::Reasoning(r) => {
                assert_eq!(r.summary.as_deref(), Some("step one"));
                assert_eq!(r.payload.0["signature"], "sig-final");
                assert_eq!(r.payload.0["thinking"], "step one");
            }
            other => panic!("expected thinking, got {other:?}"),
        }
        match &resp.message.content[1] {
            ContentBlock::Text(t) => assert_eq!(t.text, "hello"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn assembles_fragmented_tool_input_json() {
        let mut asm = Assembler::new();
        let events = feed(
            &mut asm,
            vec![
                wire::StreamEvent::MessageStart {
                    message: wire::StreamMessageStart {
                        usage: wire::WireUsage::default(),
                    },
                },
                wire::StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: json!({
                        "type": "tool_use",
                        "id": "toolu_9",
                        "name": "search",
                    }),
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::InputJsonDelta {
                        partial_json: r#"{"q":""#.into(),
                    },
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::InputJsonDelta {
                        partial_json: r#"hi"}"#.into(),
                    },
                },
                wire::StreamEvent::ContentBlockStop { index: 0 },
                wire::StreamEvent::MessageDelta {
                    delta: wire::MessageDeltaBody {
                        stop_reason: Some("tool_use".into()),
                    },
                    usage: None,
                },
                wire::StreamEvent::MessageStop,
            ],
        );

        assert!(matches!(
            &events[0],
            Event::ToolUseStart { id, name }
                if id == "toolu_9" && name == "search"
        ));
        assert!(matches!(
            &events[1],
            Event::ToolInputDelta { id, json_fragment }
                if id == "toolu_9" && json_fragment == r#"{"q":""#
        ));

        let Event::Done(resp) = events.last().unwrap() else {
            panic!("expected Done");
        };
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.message.content[0] {
            ContentBlock::ToolUse(t) => {
                assert_eq!(t.id, "toolu_9");
                assert_eq!(t.name, "search");
                assert_eq!(t.input, json!({"q": "hi"}));
            }
            other => panic!("expected tool_use, got {other:?}"),
        }
    }

    #[test]
    fn stream_error_event_surfaces_as_http_error() {
        let mut asm = Assembler::new();
        let err = asm
            .handle(wire::StreamEvent::Error {
                error: wire::ApiError {
                    kind: "overloaded_error".into(),
                    message: "try again".into(),
                },
            })
            .unwrap_err();
        match err {
            Error::Http { message, .. } => {
                assert!(message.contains("overloaded_error"));
                assert!(message.contains("try again"));
            }
            other => panic!("expected Http, got {other:?}"),
        }
    }

    /// Anthropic streams server_tool_use input via input_json_delta (same as
    /// client tool_use). Dropping those fragments leaves multi-turn history
    /// without the search query; web_search_tool_result encrypted_content must
    /// also survive Event::Done for the next request.
    #[test]
    fn stream_assembles_web_search_server_artifacts_for_multi_turn() {
        use crate::types::{Conversation, RequestOptions};
        use crate::anthropic::adapter::{block_to_wire, build_request};

        let mut asm = Assembler::new();
        // Mirrors the official streaming sequence: text → server_tool_use
        // (fragmented input) → web_search_tool_result → final text.
        let events = feed(
            &mut asm,
            vec![
                wire::StreamEvent::MessageStart {
                    message: wire::StreamMessageStart {
                        usage: wire::WireUsage {
                            input_tokens: 100,
                            output_tokens: 0,
                            ..Default::default()
                        },
                    },
                },
                wire::StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: json!({ "type": "text", "text": "" }),
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: wire::Delta::TextDelta {
                        text: "I'll search for that.".into(),
                    },
                },
                wire::StreamEvent::ContentBlockStop { index: 0 },
                wire::StreamEvent::ContentBlockStart {
                    index: 1,
                    content_block: json!({
                        "type": "server_tool_use",
                        "id": "srvtoolu_xyz789",
                        "name": "web_search",
                    }),
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 1,
                    delta: wire::Delta::InputJsonDelta {
                        partial_json: r#"{"query":"latest quantum"#.into(),
                    },
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 1,
                    delta: wire::Delta::InputJsonDelta {
                        partial_json: r#" computing breakthroughs 2025"}"#.into(),
                    },
                },
                wire::StreamEvent::ContentBlockStop { index: 1 },
                wire::StreamEvent::ContentBlockStart {
                    index: 2,
                    content_block: json!({
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_xyz789",
                        "content": [{
                            "type": "web_search_result",
                            "title": "Quantum Computing Breakthroughs in 2025",
                            "url": "https://example.com",
                            "encrypted_content": "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5",
                            "page_age": "April 30, 2025"
                        }],
                    }),
                },
                wire::StreamEvent::ContentBlockStop { index: 2 },
                wire::StreamEvent::ContentBlockStart {
                    index: 3,
                    content_block: json!({ "type": "text", "text": "" }),
                },
                wire::StreamEvent::ContentBlockDelta {
                    index: 3,
                    delta: wire::Delta::TextDelta {
                        text: "Based on search results...".into(),
                    },
                },
                wire::StreamEvent::ContentBlockStop { index: 3 },
                wire::StreamEvent::MessageDelta {
                    delta: wire::MessageDeltaBody {
                        stop_reason: Some("end_turn".into()),
                    },
                    usage: Some(wire::WireUsage {
                        input_tokens: 0,
                        output_tokens: 40,
                        ..Default::default()
                    }),
                },
                wire::StreamEvent::MessageStop,
            ],
        );

        // Hosted search must not surface as client ToolUse* stream events.
        assert!(
            !events.iter().any(|e| matches!(
                e,
                Event::ToolUseStart { .. } | Event::ToolInputDelta { .. }
            )),
            "server_tool_use must not emit client tool events"
        );

        let Event::Done(resp) = events.last().unwrap() else {
            panic!("expected Done");
        };
        assert_eq!(resp.provider, ProviderKind::Anthropic);
        assert_eq!(resp.message.content.len(), 4);

        // server_tool_use: full input from accumulated deltas, opaque Reasoning.
        let server_use = match &resp.message.content[1] {
            ContentBlock::Reasoning(r) => {
                assert_eq!(r.payload.0["type"], "server_tool_use");
                assert_eq!(r.payload.0["id"], "srvtoolu_xyz789");
                assert_eq!(r.payload.0["name"], "web_search");
                assert_eq!(
                    r.payload.0["input"],
                    json!({"query": "latest quantum computing breakthroughs 2025"})
                );
                r.payload.0.clone()
            }
            other => panic!("expected opaque server_tool_use, got {other:?}"),
        };

        // web_search_tool_result: encrypted_content intact.
        let search_result = match &resp.message.content[2] {
            ContentBlock::Reasoning(r) => {
                assert_eq!(r.payload.0["type"], "web_search_tool_result");
                assert_eq!(
                    r.payload.0["content"][0]["encrypted_content"],
                    "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5"
                );
                r.payload.0.clone()
            }
            other => panic!("expected opaque web_search_tool_result, got {other:?}"),
        };

        // Wire echo is byte-faithful for both blocks.
        assert_eq!(block_to_wire(&resp.message.content[1]), server_use);
        assert_eq!(block_to_wire(&resp.message.content[2]), search_result);

        // Full stream → conversation → next request path used in production.
        let mut conv = Conversation::new();
        conv.push_user("What are the latest quantum computing breakthroughs?");
        conv.push_response(resp.clone());
        conv.push_user("go deeper");

        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                web_search: true,
                ..Default::default()
            },
            true,
        );

        assert_eq!(body.messages[1].role, "assistant");
        assert_eq!(body.messages[1].content[1], server_use);
        assert_eq!(body.messages[1].content[2], search_result);
        assert_eq!(
            body.messages[1].content[1]["input"]["query"],
            "latest quantum computing breakthroughs 2025"
        );
        assert_eq!(
            body.messages[1].content[2]["content"][0]["encrypted_content"],
            "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5"
        );
    }
}
