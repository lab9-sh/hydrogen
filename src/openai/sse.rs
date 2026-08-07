//! Stream OpenAI Responses SSE into agnostic [`Event`]s.
//!
//! Deltas are emitted live; [`Event::Done`] uses the final `response.completed`
//! payload (not client reassembly) so content matches the non-stream path.

use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::{Stream, StreamExt};
use serde_json::Value;

use super::{adapter, wire};
use crate::types::{Event, EventStream};
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

/// Tracks open items only for routing deltas by `output_index`.
enum ItemState {
    Message { text: String },
    Reasoning { summary: String },
    FunctionCall {
        call_id: String,
        #[allow(dead_code)]
        name: String,
        arguments: String,
    },
    #[allow(dead_code)]
    Raw(Value),
}

pub(super) struct Assembler {
    items: Vec<ItemState>,
}

impl Assembler {
    pub(super) fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    pub(super) fn handle(&mut self, ev: wire::StreamEvent) -> Result<Vec<Event>, Error> {
        use wire::StreamEvent as W;
        Ok(match ev {
            W::OutputItemAdded { output_index, item } => self.start_item(output_index, item),
            W::OutputTextDelta { output_index, delta } => self.apply_text_delta(output_index, delta),
            W::ReasoningSummaryTextDelta { output_index, delta } => {
                self.apply_reasoning_delta(output_index, delta)
            }
            W::FunctionCallArgumentsDelta { output_index, delta } => {
                self.apply_function_args_delta(output_index, delta)
            }
            W::OutputItemDone { .. } => vec![Event::BlockEnd],
            W::ResponseCompleted { response } => vec![Event::Done(adapter::parse_response(response)?)],
            W::ResponseFailed { response } => {
                return Err(Error::Http {
                    status: 0,
                    message: format!("response failed: {}", response.status),
                })
            }
            // Incomplete still yields a usable partial response for the tool loop.
            W::ResponseIncomplete { response } => {
                vec![Event::Done(adapter::parse_response(response)?)]
            }
            W::Error { error } => {
                return Err(Error::Http {
                    status: 0,
                    message: error.message,
                })
            }
            W::ResponseCreated { .. } | W::ResponseInProgress { .. } | W::Other => vec![],
        })
    }

    fn start_item(&mut self, index: usize, item: Value) -> Vec<Event> {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
        let (state, events) = match kind {
            "message" => (ItemState::Message { text: String::new() }, vec![]),
            "reasoning" => (ItemState::Reasoning { summary: String::new() }, vec![]),
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let ev = Event::ToolUseStart {
                    id: call_id.clone(),
                    name: name.clone(),
                };
                (
                    ItemState::FunctionCall {
                        call_id,
                        name,
                        arguments: String::new(),
                    },
                    vec![ev],
                )
            }
            _ => (ItemState::Raw(item), vec![]),
        };
        // output_index may skip slots; pad or overwrite so deltas hit the right item.
        while self.items.len() < index {
            self.items.push(ItemState::Raw(Value::Null));
        }
        if index == self.items.len() {
            self.items.push(state);
        } else {
            self.items[index] = state;
        }
        events
    }

    fn apply_text_delta(&mut self, index: usize, delta: String) -> Vec<Event> {
        let Some(state) = self.items.get_mut(index) else {
            return vec![];
        };
        match state {
            ItemState::Message { text } => {
                text.push_str(&delta);
                vec![Event::TextDelta(delta)]
            }
            _ => vec![],
        }
    }

    fn apply_reasoning_delta(&mut self, index: usize, delta: String) -> Vec<Event> {
        let Some(state) = self.items.get_mut(index) else {
            return vec![];
        };
        match state {
            ItemState::Reasoning { summary } => {
                summary.push_str(&delta);
                vec![Event::ReasoningDelta(delta)]
            }
            _ => vec![],
        }
    }

    fn apply_function_args_delta(&mut self, index: usize, delta: String) -> Vec<Event> {
        let Some(state) = self.items.get_mut(index) else {
            return vec![];
        };
        match state {
            ItemState::FunctionCall {
                call_id,
                arguments,
                ..
            } => {
                arguments.push_str(&delta);
                vec![Event::ToolInputDelta {
                    id: call_id.clone(),
                    json_fragment: delta,
                }]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, ProviderKind, StopReason};
    use serde_json::json;

    fn feed(asm: &mut Assembler, events: Vec<wire::StreamEvent>) -> Vec<Event> {
        let mut out = Vec::new();
        for ev in events {
            out.extend(asm.handle(ev).unwrap());
        }
        out
    }

    /// Live deltas must keep `output_index` aligned with item starts; Done must
    /// come from the completed payload (not our partial buffers) so stream and
    /// non-stream content stay consistent.
    #[test]
    fn stream_emits_deltas_and_done_from_completed_payload() {
        let mut asm = Assembler::new();
        let completed = wire::ResponsesResponse {
            output: vec![
                json!({
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": "hello world" }],
                }),
                json!({
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "ping",
                    "arguments": "{\"n\":1}",
                    "status": "completed",
                }),
            ],
            status: "completed".into(),
            usage: wire::WireUsage {
                input_tokens: 1,
                output_tokens: 2,
            ..Default::default()
            },
            incomplete_details: None,
        };

        let events = feed(
            &mut asm,
            vec![
                wire::StreamEvent::OutputItemAdded {
                    output_index: 0,
                    item: json!({ "type": "message" }),
                },
                wire::StreamEvent::OutputTextDelta {
                    output_index: 0,
                    delta: "hello ".into(),
                },
                wire::StreamEvent::OutputTextDelta {
                    output_index: 0,
                    delta: "world".into(),
                },
                wire::StreamEvent::OutputItemDone {
                    output_index: 0,
                    item: json!({ "type": "message" }),
                },
                wire::StreamEvent::OutputItemAdded {
                    output_index: 1,
                    item: json!({
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "ping",
                    }),
                },
                wire::StreamEvent::FunctionCallArgumentsDelta {
                    output_index: 1,
                    delta: r#"{"n":"#.into(),
                },
                wire::StreamEvent::FunctionCallArgumentsDelta {
                    output_index: 1,
                    delta: "1}".into(),
                },
                wire::StreamEvent::OutputItemDone {
                    output_index: 1,
                    item: json!({ "type": "function_call" }),
                },
                wire::StreamEvent::ResponseCompleted {
                    response: completed,
                },
            ],
        );

        assert!(matches!(&events[0], Event::TextDelta(t) if t == "hello "));
        assert!(matches!(&events[1], Event::TextDelta(t) if t == "world"));
        assert!(matches!(events[2], Event::BlockEnd));
        assert!(matches!(
            &events[3],
            Event::ToolUseStart { id, name } if id == "call_1" && name == "ping"
        ));
        assert!(matches!(
            &events[4],
            Event::ToolInputDelta { id, json_fragment }
                if id == "call_1" && json_fragment == r#"{"n":"#
        ));

        let Event::Done(resp) = events.last().unwrap() else {
            panic!("expected Done");
        };
        assert_eq!(resp.provider, ProviderKind::OpenAi);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.message.content.len(), 2);
        match &resp.message.content[0] {
            ContentBlock::Text(t) => assert_eq!(t.text, "hello world"),
            other => panic!("expected text from completed payload, got {other:?}"),
        }
        match &resp.message.content[1] {
            ContentBlock::ToolUse(t) => assert_eq!(t.input, json!({"n": 1})),
            other => panic!("expected tool use, got {other:?}"),
        }
    }

    #[test]
    fn response_failed_is_an_error_not_done() {
        let mut asm = Assembler::new();
        let err = asm
            .handle(wire::StreamEvent::ResponseFailed {
                response: wire::ResponsesResponse {
                    output: vec![],
                    status: "failed".into(),
                    usage: wire::WireUsage::default(),
                    incomplete_details: None,
                },
            })
            .unwrap_err();
        assert!(matches!(err, Error::Http { .. }));
    }
}
