use std::time::Duration;

use serde_json::{json, Value};

use super::{sse, wire, OpenAiConfig};
use crate::types::{
    ContentBlock, Conversation, EventStream, Message, OpaquePayload, ProviderKind, ReasoningBlock,
    RequestOptions, Response, Role, StopReason, TextBlock, ThinkingEffort, ToolOutput,
    ToolUseBlock, Usage,
};
use crate::Error;

// High default so long tool/reasoning turns are not truncated by accident when
// the caller omits max_tokens.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 64_000;

pub(crate) struct Adapter {
    cfg: OpenAiConfig,
    http: reqwest::Client,
}

impl Adapter {
    pub fn new(cfg: OpenAiConfig) -> Self {
        let http = cfg.http_client.clone().unwrap_or_default();
        Self { cfg, http }
    }

    fn request(&self, conv: &Conversation, opts: &RequestOptions, stream: bool) -> reqwest::RequestBuilder {
        let body = build_request(conv, opts, stream);
        self.http
            .post(format!("{}/v1/responses", self.cfg.base_url))
            .header("Authorization", format!("Bearer {}", self.cfg.api_key))
            .json(&body)
    }

    pub async fn send(&self, conv: &Conversation, opts: &RequestOptions) -> Result<Response, Error> {
        let resp = self.request(conv, opts, false).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let retry_after = parse_retry_after(resp.headers());
            let body = resp.text().await.unwrap_or_default();
            return Err(map_error(status.as_u16(), &body, retry_after));
        }
        let wire: wire::ResponsesResponse = serde_json::from_str(&resp.text().await?)?;
        parse_response(wire)
    }

    pub async fn stream(&self, conv: &Conversation, opts: &RequestOptions) -> Result<EventStream, Error> {
        let resp = self.request(conv, opts, true).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let retry_after = parse_retry_after(resp.headers());
            let body = resp.text().await.unwrap_or_default();
            return Err(map_error(status.as_u16(), &body, retry_after));
        }
        Ok(sse::event_stream(resp.bytes_stream()))
    }
}

fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get("retry-after")?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

pub(super) fn map_error(status: u16, body: &str, retry_after: Option<Duration>) -> Error {
    let message = serde_json::from_str::<wire::ErrorEnvelope>(body)
        .map(|e| e.error.message)
        .unwrap_or_else(|_| body.to_string());
    match status {
        401 | 403 => Error::Auth,
        429 => Error::RateLimited { retry_after },
        400 => Error::InvalidRequest(message),
        _ => Error::Http { status, message },
    }
}

pub(super) fn build_request(
    conv: &Conversation,
    opts: &RequestOptions,
    stream: bool,
) -> wire::ResponsesRequest {
    let effort = opts.thinking.unwrap_or(ThinkingEffort::Medium);
    let reasoning_effort = match effort {
        ThinkingEffort::Low => wire::Effort::Low,
        ThinkingEffort::Medium => wire::Effort::Medium,
        ThinkingEffort::High => wire::Effort::High,
    };

    wire::ResponsesRequest {
        model: opts.model.clone(),
        input: conversation_to_input(conv),
        instructions: opts.system.clone(),
        tools: opts
            .tools
            .iter()
            .map(|t| wire::WireTool {
                kind: "function",
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            })
            .collect(),
        temperature: opts.temperature,
        max_output_tokens: Some(opts.max_tokens.unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)),
        reasoning: wire::Reasoning {
            effort: reasoning_effort,
        },
        // Client owns history; store:true would create server-side state we
        // never consult.
        store: false,
        // Required so multi-turn can resubmit encrypted reasoning items.
        include: vec!["reasoning.encrypted_content"],
        prompt_cache_key: conv.cache_key().to_string(),
        stream: stream.then_some(true),
    }
}

/// Flatten messages into Responses `input` items (role messages, function
/// calls, and function outputs are separate top-level items).
fn conversation_to_input(conv: &Conversation) -> Vec<Value> {
    conv.messages()
        .iter()
        .flat_map(message_to_input)
        .collect()
}

fn message_to_input(msg: &Message) -> Vec<Value> {
    match msg.role {
        Role::User => msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(t) => Some(json!({
                    "role": "user",
                    "content": t.text,
                })),
                ContentBlock::ToolResult(t) => {
                    let output = match &t.output {
                        ToolOutput::Text(s) => s.clone(),
                        ToolOutput::Json(v) => v.to_string(),
                        ToolOutput::Error(s) => s.clone(),
                    };
                    Some(json!({
                        "type": "function_call_output",
                        "call_id": t.id,
                        "output": output,
                    }))
                }
                _ => None,
            })
            .collect(),
        Role::Assistant => msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult(_) => None,
                _ => Some(block_to_input(block)),
            })
            .collect(),
    }
}

pub(super) fn block_to_input(block: &ContentBlock) -> Value {
    match block {
        // Prefer stored wire form so ids/status/encrypted fields survive.
        ContentBlock::Text(t) => t
            .extras
            .as_ref()
            .map(|e| e.0.clone())
            .unwrap_or_else(|| {
                json!({
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": t.text }],
                })
            }),
        ContentBlock::ToolUse(t) => t
            .extras
            .as_ref()
            .map(|e| e.0.clone())
            .unwrap_or_else(|| {
                json!({
                    "type": "function_call",
                    "call_id": t.id,
                    "name": t.name,
                    "arguments": serde_json::to_string(&t.input).unwrap_or_else(|_| "{}".into()),
                    "status": "completed",
                })
            }),
        ContentBlock::Reasoning(r) => r.payload.0.clone(),
        ContentBlock::ToolResult(_) => {
            debug_assert!(false, "tool results belong in user messages");
            json!(null)
        }
    }
}

pub(super) fn parse_response(wire: wire::ResponsesResponse) -> Result<Response, Error> {
    let stop_reason = map_stop_reason(&wire);
    let usage = Usage {
        input_tokens: wire.usage.input_tokens,
        output_tokens: wire.usage.output_tokens,
    };
    let content = wire
        .output
        .into_iter()
        .map(wire_item_to_agnostic)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Response {
        message: Message {
            role: Role::Assistant,
            content,
        },
        stop_reason,
        usage,
        provider: ProviderKind::OpenAi,
    })
}

pub(super) fn wire_item_to_agnostic(item: Value) -> Result<ContentBlock, Error> {
    let kind = item
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    match kind.as_str() {
        "message" => {
            let text = item
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| {
                            if p.get("type").and_then(Value::as_str) == Some("output_text") {
                                p.get("text").and_then(Value::as_str).map(String::from)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            // Keep full item in extras for faithful multi-turn resubmission.
            Ok(ContentBlock::Text(TextBlock {
                text,
                extras: Some(OpaquePayload(item)),
            }))
        }
        "function_call" => {
            let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or("{}");
            let input = serde_json::from_str(arguments).unwrap_or(Value::Null);
            Ok(ContentBlock::ToolUse(ToolUseBlock {
                id: str_field(&item, "call_id"),
                name: str_field(&item, "name"),
                input,
                extras: Some(OpaquePayload(item)),
            }))
        }
        "reasoning" => {
            let summary = item
                .get("summary")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|s| s.get("text"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from);
            Ok(ContentBlock::Reasoning(reasoning(item, summary)))
        }
        // Preserve unknown item types rather than dropping them mid-history.
        _ => Ok(ContentBlock::Reasoning(reasoning(item, None))),
    }
}

fn reasoning(item: Value, summary: Option<String>) -> ReasoningBlock {
    ReasoningBlock {
        provider: ProviderKind::OpenAi,
        payload: OpaquePayload(item),
        summary,
    }
}

fn str_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(super) fn map_stop_reason(wire: &wire::ResponsesResponse) -> StopReason {
    // Responses does not always set a stop_reason; function_call items imply
    // the model wants tools run before continuing.
    let has_tool_call = wire
        .output
        .iter()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"));

    if has_tool_call {
        return StopReason::ToolUse;
    }

    match wire.status.as_str() {
        "completed" => StopReason::EndTurn,
        "incomplete" => match wire.incomplete_details.as_ref().map(|d| d.reason.as_str()) {
            Some("max_output_tokens") | Some("max_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::Refusal,
            Some(other) => StopReason::Other(other.to_string()),
            None => StopReason::Other("incomplete".into()),
        },
        other => StopReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encrypted reasoning must be resubmitted byte-for-byte. Rebuilding from a
    /// summary alone fails on the next turn with opaque provider errors.
    #[test]
    fn reasoning_item_round_trips_via_payload() {
        let wire = json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc-blob",
            "summary": [{ "type": "summary_text", "text": "brief" }],
        });
        let block = wire_item_to_agnostic(wire.clone()).unwrap();
        match &block {
            ContentBlock::Reasoning(r) => {
                assert_eq!(r.provider, ProviderKind::OpenAi);
                assert_eq!(r.summary.as_deref(), Some("brief"));
                assert_eq!(r.payload.0, wire);
            }
            other => panic!("expected reasoning, got {other:?}"),
        }
        assert_eq!(block_to_input(&block), wire);
    }

    #[test]
    fn message_and_function_call_prefer_stored_extras_on_resubmit() {
        let msg_wire = json!({
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": "hi" }],
        });
        let fc_wire = json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{\"q\":\"x\"}",
            "status": "completed",
        });

        let msg = wire_item_to_agnostic(msg_wire.clone()).unwrap();
        let fc = wire_item_to_agnostic(fc_wire.clone()).unwrap();

        match &msg {
            ContentBlock::Text(t) => assert_eq!(t.text, "hi"),
            other => panic!("expected text, got {other:?}"),
        }
        match &fc {
            ContentBlock::ToolUse(t) => {
                assert_eq!(t.id, "call_1");
                assert_eq!(t.input, json!({"q": "x"}));
            }
            other => panic!("expected tool use, got {other:?}"),
        }

        // Round-trip must use extras, not a reconstructed skeleton missing ids.
        assert_eq!(block_to_input(&msg), msg_wire);
        assert_eq!(block_to_input(&fc), fc_wire);
    }

    /// Tool loop layout is easy to get wrong: function_call_output is a top-level
    /// input item (not nested under a user role message).
    #[test]
    fn multi_turn_input_flattens_tool_results_and_preserves_reasoning() {
        let mut conv = Conversation::new();
        conv.push_user("search something");

        let resp = parse_response(wire::ResponsesResponse {
            output: vec![
                json!({
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "enc",
                    "summary": [],
                }),
                json!({
                    "type": "function_call",
                    "call_id": "call_9",
                    "name": "search",
                    "arguments": "{\"q\":\"rust\"}",
                    "status": "completed",
                }),
            ],
            status: "completed".into(),
            usage: wire::WireUsage {
                input_tokens: 3,
                output_tokens: 7,
            },
            incomplete_details: None,
        })
        .unwrap();

        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.provider, ProviderKind::OpenAi);
        conv.push_response(resp);
        conv.push_tool_result("call_9", ToolOutput::Text("docs".into()));

        let body = build_request(
            &conv,
            &RequestOptions {
                model: "gpt-5".into(),
                ..Default::default()
            },
            false,
        );

        // user text, reasoning, function_call, function_call_output
        assert_eq!(body.input.len(), 4);
        assert_eq!(body.input[0]["role"], "user");
        assert_eq!(body.input[1]["type"], "reasoning");
        assert_eq!(body.input[1]["encrypted_content"], "enc");
        assert_eq!(body.input[2]["type"], "function_call");
        assert_eq!(body.input[2]["call_id"], "call_9");
        assert_eq!(body.input[3]["type"], "function_call_output");
        assert_eq!(body.input[3]["call_id"], "call_9");
        assert_eq!(body.input[3]["output"], "docs");

        assert!(!body.store);
        assert!(body.include.contains(&"reasoning.encrypted_content"));
        assert_eq!(body.prompt_cache_key, conv.cache_key());
    }

    #[test]
    fn stop_reason_incomplete_max_tokens() {
        let wire = wire::ResponsesResponse {
            output: vec![json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "partial" }],
            })],
            status: "incomplete".into(),
            usage: wire::WireUsage::default(),
            incomplete_details: Some(wire::IncompleteDetails {
                reason: "max_output_tokens".into(),
            }),
        };
        assert_eq!(map_stop_reason(&wire), StopReason::MaxTokens);
    }

    #[test]
    fn function_call_forces_tool_use_even_when_status_completed() {
        let wire = wire::ResponsesResponse {
            output: vec![json!({
                "type": "function_call",
                "call_id": "c",
                "name": "f",
                "arguments": "{}",
            })],
            status: "completed".into(),
            usage: wire::WireUsage::default(),
            incomplete_details: None,
        };
        assert_eq!(map_stop_reason(&wire), StopReason::ToolUse);
    }

    #[test]
    fn tool_result_error_becomes_function_call_output_string() {
        // Via conversation path so we exercise message_to_input, not just block_to_input.
        let mut conv = Conversation::new();
        conv.push_tool_result(
            "call_err",
            ToolOutput::Error("failed".into()),
        );
        // Also cover Json stringify on the same path with a second result.
        conv.push_tool_result(
            "call_json",
            ToolOutput::Json(json!({"a": 1})),
        );
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "gpt-5".into(),
                ..Default::default()
            },
            false,
        );
        assert_eq!(body.input[0]["output"], "failed");
        assert_eq!(body.input[1]["output"], "{\"a\":1}");
    }
}
