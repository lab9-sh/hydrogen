use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;

use super::Response;
use crate::Error;

/// Incremental stream events. Shape is fixed so UIs can render without
/// knowing which backend's SSE schema is in use.
#[derive(Debug)]
pub enum Event {
    TextDelta(String),
    ReasoningDelta(String),
    ToolUseStart { id: String, name: String },
    ToolInputDelta { id: String, json_fragment: String },
    BlockEnd,
    /// Terminal event carrying the assembled [`Response`] for `push_response`.
    Done(Response),
}

/// Type-erased event stream so each adapter can own its SSE state machine.
pub struct EventStream(pub(crate) Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>);

impl Stream for EventStream {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0.as_mut().poll_next(cx)
    }
}
