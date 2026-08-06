use std::time::Duration;

use serde_json::{json, Value};

use super::{sse, wire, XaiConfig};
use crate::types::{
    ContentBlock, Conversation, EventStream, Message, OpaquePayload, ProviderKind, ReasoningBlock,
    RequestOptions, Response, Role, StopReason, TextBlock, ThinkingEffort, ToolChoice, ToolOutput,
    ToolUseBlock, Usage,
};
use crate::Error;

// High default so long tool/reasoning turns are not truncated by accident when
// the caller omits max_tokens.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 64_000;

pub(crate) struct Adapter {
    cfg: XaiConfig,
    http: reqwest::Client,
}

impl Adapter {
    pub fn new(cfg: XaiConfig) -> Self {
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

    let mut tools: Vec<wire::WireTool> = opts
        .tools
        .iter()
        .map(|t| {
            wire::WireTool::function(
                t.name.clone(),
                t.description.clone(),
                t.input_schema.clone(),
            )
        })
        .collect();
    if opts.web_search {
        tools.push(wire::WireTool::web_search());
    }

    wire::ResponsesRequest {
        model: opts.model.clone(),
        input: conversation_to_input(conv),
        instructions: opts.system.clone(),
        tools,
        tool_choice: map_tool_choice(&opts.tool_choice),
        parallel_tool_calls: opts.parallel_tool_calls,
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

/// Map portable tool_choice onto Responses API values. `Auto` omits the field
/// (provider default). Specific tools use `{ "type": "function", "name": … }`.
pub(super) fn map_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => None,
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Tool(name) => Some(json!({
            "type": "function",
            "name": name,
        })),
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
    let usage = map_usage(wire.usage);
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
        provider: ProviderKind::Xai,
    })
}

/// Map Responses usage onto the Anthropic-shaped [`Usage`] fields.
///
/// xAI bills `input_tokens` as the full prompt (cached subset included) and
/// reports cache hits in `input_tokens_details.cached_tokens`. Split them so
/// `input_tokens` is the uncached remainder and `cache_read_input_tokens` is
/// the hit — then [`Usage::total_input_tokens`] stays additive.
pub(super) fn map_usage(wire: wire::WireUsage) -> Usage {
    let cached = wire.input_tokens_details.cached_tokens.min(wire.input_tokens);
    Usage {
        input_tokens: wire.input_tokens - cached,
        output_tokens: wire.output_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cached,
    }
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
        provider: ProviderKind::Xai,
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

    /// xAI mirrors OpenAI Responses closely; lock the few places drift hurts:
    /// provider pin + encrypted reasoning resubmit on multi-turn tool loops.
    #[test]
    fn multi_turn_preserves_encrypted_reasoning_and_pins_xai() {
        let mut conv = Conversation::new();
        conv.push_user("think then call");

        let resp = parse_response(wire::ResponsesResponse {
            output: vec![
                json!({
                    "type": "reasoning",
                    "id": "rs_x",
                    "encrypted_content": "xai-enc",
                    "summary": [{ "type": "summary_text", "text": "plan" }],
                }),
                json!({
                    "type": "function_call",
                    "call_id": "call_x",
                    "name": "tool",
                    "arguments": "{}",
                    "status": "completed",
                }),
            ],
            status: "completed".into(),
            usage: wire::WireUsage::default(),
            incomplete_details: None,
        })
        .unwrap();

        assert_eq!(resp.provider, ProviderKind::Xai);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.message.content[0] {
            ContentBlock::Reasoning(r) => {
                assert_eq!(r.provider, ProviderKind::Xai);
                assert_eq!(r.summary.as_deref(), Some("plan"));
            }
            other => panic!("expected reasoning, got {other:?}"),
        }

        conv.push_response(resp);
        assert_eq!(conv.provider(), Some(ProviderKind::Xai));
        conv.push_tool_result("call_x", ToolOutput::Text("ok".into()));

        let body = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                ..Default::default()
            },
            true,
        );

        assert_eq!(body.stream, Some(true));
        assert_eq!(body.input[1]["encrypted_content"], "xai-enc");
        assert_eq!(body.input[2]["type"], "function_call");
        assert_eq!(body.input[3]["type"], "function_call_output");
        assert_eq!(body.include, vec!["reasoning.encrypted_content"]);
    }

    /// Hosted web_search must appear as type-only tool, next to function tools.
    #[test]
    fn web_search_emits_hosted_tool_alongside_function_tools() {
        use crate::types::ToolDef;

        let conv = Conversation::new();
        let lookup = ToolDef {
            name: "lookup".into(),
            description: "d".into(),
            input_schema: json!({"type": "object"}),
        };

        let off = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                tools: vec![lookup.clone()],
                web_search: false,
                ..Default::default()
            },
            false,
        );
        let off_json = serde_json::to_value(&off.tools).unwrap();
        assert_eq!(off_json.as_array().unwrap().len(), 1);
        assert_eq!(off_json[0]["type"], "function");
        assert_eq!(off_json[0]["name"], "lookup");

        let on = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                tools: vec![lookup],
                web_search: true,
                ..Default::default()
            },
            false,
        );
        let on_json = serde_json::to_value(&on.tools).unwrap();
        assert_eq!(on_json.as_array().unwrap().len(), 2);
        assert_eq!(on_json[0]["type"], "function");
        assert_eq!(on_json[1]["type"], "web_search");
        assert!(on_json[1].get("name").is_none());
        assert!(on_json[1].get("parameters").is_none());
    }

    #[test]
    fn web_search_off_emits_no_search_tool_when_tools_empty() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                web_search: false,
                ..Default::default()
            },
            false,
        );
        assert!(body.tools.is_empty());
    }

    /// Default Auto + unset parallel omits both fields (provider defaults).
    #[test]
    fn tool_choice_auto_default_omits_fields() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                tools: vec![crate::types::ToolDef {
                    name: "play_move".into(),
                    description: "d".into(),
                    input_schema: json!({"type": "object"}),
                }],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
                ..Default::default()
            },
            false,
        );
        assert!(body.tool_choice.is_none());
        assert!(body.parallel_tool_calls.is_none());
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("tool_choice").is_none());
        assert!(json.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn tool_choice_tool_with_parallel_disabled() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                tool_choice: ToolChoice::Tool("play_move".into()),
                parallel_tool_calls: Some(false),
                ..Default::default()
            },
            false,
        );
        assert_eq!(
            body.tool_choice,
            Some(json!({ "type": "function", "name": "play_move" }))
        );
        assert_eq!(body.parallel_tool_calls, Some(false));

        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tool_choice"]["type"], "function");
        assert_eq!(json["tool_choice"]["name"], "play_move");
        assert_eq!(json["parallel_tool_calls"], false);
    }

    #[test]
    fn tool_choice_required_and_none() {
        let conv = Conversation::new();
        let required = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                tool_choice: ToolChoice::Required,
                ..Default::default()
            },
            false,
        );
        assert_eq!(required.tool_choice, Some(json!("required")));

        let none = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                tool_choice: ToolChoice::None,
                parallel_tool_calls: Some(true),
                ..Default::default()
            },
            false,
        );
        assert_eq!(none.tool_choice, Some(json!("none")));
        assert_eq!(none.parallel_tool_calls, Some(true));
    }

    /// web_search_call items must reappear in the next request input unchanged.
    #[test]
    fn multi_turn_resubmits_web_search_call_item() {
        let mut conv = Conversation::new();
        conv.push_user("latest news?");

        let search_call = json!({
            "type": "web_search_call",
            "id": "ws_xai_1",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "latest news about AI"
            }
        });
        let message = json!({
            "type": "message",
            "id": "msg_x",
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": "Here is the news." }],
        });

        let resp = parse_response(wire::ResponsesResponse {
            output: vec![search_call.clone(), message],
            status: "completed".into(),
            usage: wire::WireUsage::default(),
            incomplete_details: None,
        })
        .unwrap();

        assert_eq!(resp.provider, ProviderKind::Xai);
        assert!(matches!(
            &resp.message.content[0],
            ContentBlock::Reasoning(r) if r.provider == ProviderKind::Xai
        ));
        assert_eq!(block_to_input(&resp.message.content[0]), search_call);

        conv.push_response(resp);
        conv.push_user("more?");

        let body = build_request(
            &conv,
            &RequestOptions {
                model: "grok-4".into(),
                web_search: true,
                ..Default::default()
            },
            false,
        );

        assert_eq!(body.input[1], search_call);
        assert_eq!(body.input[1]["type"], "web_search_call");
        assert_eq!(body.input[1]["id"], "ws_xai_1");
        let tools_json = serde_json::to_value(&body.tools).unwrap();
        assert_eq!(tools_json[0]["type"], "web_search");
    }

    /// xAI reports full input_tokens with cached as a subset; split so totals
    /// stay additive and match Anthropic-shaped telemetry.
    #[test]
    fn map_usage_splits_cached_tokens_out_of_input() {
        let usage = map_usage(wire::WireUsage {
            input_tokens: 1192,
            output_tokens: 174,
            input_tokens_details: wire::InputTokensDetails {
                cached_tokens: 256,
            },
        });
        assert_eq!(usage.input_tokens, 936);
        assert_eq!(usage.cache_read_input_tokens, 256);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.output_tokens, 174);
        assert_eq!(usage.total_input_tokens(), 1192);
    }

    #[test]
    fn map_usage_caps_cached_at_input_total() {
        let usage = map_usage(wire::WireUsage {
            input_tokens: 100,
            output_tokens: 1,
            input_tokens_details: wire::InputTokensDetails {
                cached_tokens: 999,
            },
        });
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 100);
        assert_eq!(usage.total_input_tokens(), 100);
    }

    #[test]
    fn parse_response_surfaces_cached_tokens() {
        let resp = parse_response(wire::ResponsesResponse {
            output: vec![json!({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{ "type": "output_text", "text": "ok" }],
            })],
            status: "completed".into(),
            usage: wire::WireUsage {
                input_tokens: 500,
                output_tokens: 10,
                input_tokens_details: wire::InputTokensDetails {
                    cached_tokens: 400,
                },
            },
            incomplete_details: None,
        })
        .unwrap();
        assert_eq!(resp.usage.input_tokens, 100);
        assert_eq!(resp.usage.cache_read_input_tokens, 400);
        assert_eq!(resp.usage.total_input_tokens(), 500);
    }
}
