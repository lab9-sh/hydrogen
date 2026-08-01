//! Stream xAI Responses SSE into agnostic [`Event`]s.
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
