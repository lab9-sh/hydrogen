use std::time::Duration;

use serde_json::{json, Value};

use super::{sse, wire, AnthropicConfig, API_VERSION};
use crate::types::{
    ContentBlock, Conversation, EventStream, Message, OpaquePayload, ProviderKind, ReasoningBlock,
    RequestOptions, Response, Role, StopReason, TextBlock, ThinkingEffort, ToolChoice, ToolOutput,
    ToolUseBlock, Usage,
};
use crate::Error;

// High default so long tool/thinking turns are not truncated by accident when
// the caller omits max_tokens.
const DEFAULT_MAX_TOKENS: u32 = 64_000;

pub(crate) struct Adapter {
    cfg: AnthropicConfig,
    http: reqwest::Client,
}

impl Adapter {
    pub fn new(cfg: AnthropicConfig) -> Self {
        let http = cfg.http_client.clone().unwrap_or_default();
        Self { cfg, http }
    }

    fn request(&self, conv: &Conversation, opts: &RequestOptions, stream: bool) -> reqwest::RequestBuilder {
        let body = build_request(conv, opts, stream);
        self.http
            .post(format!("{}/v1/messages", self.cfg.base_url))
            .header("x-api-key", &self.cfg.api_key)
            .header("anthropic-version", API_VERSION)
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
        let wire: wire::MessagesResponse = serde_json::from_str(&resp.text().await?)?;
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
) -> wire::MessagesRequest {
    let effort = opts.thinking.unwrap_or(ThinkingEffort::Medium);
    let (thinking, output_config) = translate_thinking(&opts.model, effort);

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

    wire::MessagesRequest {
        model: opts.model.clone(),
        max_tokens: opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages: conv.messages().iter().map(message_to_wire).collect(),
        system: opts.system.clone(),
        tools,
        tool_choice: map_tool_choice(&opts.tool_choice, opts.parallel_tool_calls),
        temperature: opts.temperature,
        thinking: Some(thinking),
        output_config,
        cache_control: wire::CacheControl::EPHEMERAL,
        stream: stream.then_some(true),
    }
}

/// Map portable tool_choice + optional parallel control onto Anthropic's
/// `tool_choice` object. `Auto` with unset parallel omits the field entirely
/// (provider default). When `parallel_tool_calls` is set, Anthropic folds it
/// into `disable_parallel_tool_use` on the tool_choice object.
pub(super) fn map_tool_choice(
    choice: &ToolChoice,
    parallel_tool_calls: Option<bool>,
) -> Option<wire::WireToolChoice> {
    let disable_parallel = parallel_tool_calls.map(|p| !p);

    // Omit when Auto + no parallel override — preserves historical wire shape.
    if matches!(choice, ToolChoice::Auto) && disable_parallel.is_none() {
        return None;
    }

    let (kind, name) = match choice {
        ToolChoice::Auto => ("auto", None),
        ToolChoice::Required => ("any", None),
        ToolChoice::Tool(n) => ("tool", Some(n.clone())),
        ToolChoice::None => ("none", None),
    };

    Some(wire::WireToolChoice {
        kind,
        name,
        disable_parallel_tool_use: disable_parallel,
    })
}

/// Map portable effort onto the thinking API the model actually supports.
fn translate_thinking(
    model: &str,
    effort: ThinkingEffort,
) -> (wire::Thinking, Option<wire::OutputConfig>) {
    if uses_extended_thinking(model) {
        let budget_tokens = extended_thinking_budget(effort);
        (wire::Thinking::Enabled { budget_tokens }, None)
    } else {
        let effort = match effort {
            ThinkingEffort::Low => wire::Effort::Low,
            ThinkingEffort::Medium => wire::Effort::Medium,
            ThinkingEffort::High => wire::Effort::High,
        };
        let thinking = wire::Thinking::Adaptive {
            display: wire::Display::Summarized,
        };
        (thinking, Some(wire::OutputConfig { effort }))
    }
}

// Haiku 4.5 still uses budget-token extended thinking; newer models use
// adaptive thinking + output_config.effort instead.
fn uses_extended_thinking(model: &str) -> bool {
    model.contains("haiku-4-5")
}

fn extended_thinking_budget(effort: ThinkingEffort) -> u32 {
    match effort {
        ThinkingEffort::Low => 4_096,
        ThinkingEffort::Medium => 16_384,
        ThinkingEffort::High => 32_768,
    }
}

fn message_to_wire(msg: &Message) -> wire::WireMessage {
    wire::WireMessage {
        role: match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        },
        content: msg.content.iter().map(block_to_wire).collect(),
    }
}

pub(super) fn block_to_wire(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text(t) => json!({ "type": "text", "text": t.text }),
        ContentBlock::ToolUse(t) => json!({
            "type": "tool_use",
            "id": t.id,
            "name": t.name,
            "input": t.input,
        }),
        ContentBlock::ToolResult(t) => {
            let (content, is_error) = match &t.output {
                ToolOutput::Text(s) => (Value::String(s.clone()), false),
                ToolOutput::Json(v) => (Value::String(v.to_string()), false),
                ToolOutput::Error(s) => (Value::String(s.clone()), true),
            };
            let mut obj = json!({
                "type": "tool_result",
                "tool_use_id": t.id,
                "content": content,
            });
            if is_error {
                obj["is_error"] = Value::Bool(true);
            }
            obj
        }
        // Echo the original thinking/redacted block; Anthropic verifies signatures.
        ContentBlock::Reasoning(r) => r.payload.0.clone(),
    }
}

pub(super) fn parse_response(wire: wire::MessagesResponse) -> Result<Response, Error> {
    let content = wire
        .content
        .into_iter()
        .map(wire_block_to_agnostic)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Response {
        message: Message {
            role: Role::Assistant,
            content,
        },
        stop_reason: map_stop_reason(wire.stop_reason.as_deref()),
        usage: Usage {
            input_tokens: wire.usage.input_tokens,
            output_tokens: wire.usage.output_tokens,
        },
        provider: ProviderKind::Anthropic,
    })
}

pub(super) fn wire_block_to_agnostic(block: Value) -> Result<ContentBlock, Error> {
    let kind = block
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    match kind.as_str() {
        "text" => Ok(ContentBlock::Text(TextBlock {
            text: block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            extras: None,
        })),
        "tool_use" => Ok(ContentBlock::ToolUse(ToolUseBlock::new(
            str_field(&block, "id"),
            str_field(&block, "name"),
            block.get("input").cloned().unwrap_or(Value::Null),
        ))),
        "thinking" => {
            let summary = block
                .get("thinking")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from);
            Ok(ContentBlock::Reasoning(reasoning(block, summary)))
        }
        // Opaque to clients but still must be returned on the next turn.
        "redacted_thinking" => Ok(ContentBlock::Reasoning(reasoning(block, None))),
        // Unknown block types: preserve rather than drop so multi-turn works.
        _ => Ok(ContentBlock::Reasoning(reasoning(block, None))),
    }
}

fn reasoning(block: Value, summary: Option<String>) -> ReasoningBlock {
    ReasoningBlock {
        provider: ProviderKind::Anthropic,
        payload: OpaquePayload(block),
        summary,
    }
}

fn str_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(super) fn map_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("end_turn") => StopReason::EndTurn,
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("refusal") => StopReason::Refusal,
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::Other(String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ToolDef, ToolResultBlock};

    fn opts(model: &str) -> RequestOptions {
        RequestOptions {
            model: model.into(),
            thinking: Some(ThinkingEffort::High),
            ..Default::default()
        }
    }

    /// Anthropic verifies thinking signatures on the next turn. Dropping or
    /// reshaping the block is a multi-turn footgun that only surfaces as a
    /// cryptic 400 from the API.
    #[test]
    fn thinking_block_round_trips_verbatim_including_signature() {
        let wire = json!({
            "type": "thinking",
            "thinking": "plan the steps",
            "signature": "sig-abc-123",
        });
        let block = wire_block_to_agnostic(wire.clone()).unwrap();
        match &block {
            ContentBlock::Reasoning(r) => {
                assert_eq!(r.provider, ProviderKind::Anthropic);
                assert_eq!(r.summary.as_deref(), Some("plan the steps"));
                assert_eq!(r.payload.0, wire);
            }
            other => panic!("expected reasoning, got {other:?}"),
        }
        assert_eq!(block_to_wire(&block), wire);
    }

    #[test]
    fn redacted_thinking_is_preserved_opaque() {
        let wire = json!({
            "type": "redacted_thinking",
            "data": "encrypted-blob",
        });
        let block = wire_block_to_agnostic(wire.clone()).unwrap();
        assert!(matches!(&block, ContentBlock::Reasoning(r) if r.summary.is_none()));
        assert_eq!(block_to_wire(&block), wire);
    }

    #[test]
    fn unknown_block_types_are_preserved_not_dropped() {
        let wire = json!({ "type": "server_tool_use", "id": "x", "name": "web" });
        let block = wire_block_to_agnostic(wire.clone()).unwrap();
        assert_eq!(block_to_wire(&block), wire);
    }

    #[test]
    fn tool_result_error_sets_is_error_flag() {
        let block = ContentBlock::ToolResult(ToolResultBlock {
            id: "toolu_1".into(),
            output: ToolOutput::Error("nope".into()),
        });
        let wire = block_to_wire(&block);
        assert_eq!(wire["type"], "tool_result");
        assert_eq!(wire["tool_use_id"], "toolu_1");
        assert_eq!(wire["content"], "nope");
        assert_eq!(wire["is_error"], true);
    }

    #[test]
    fn tool_result_json_is_stringified_without_error_flag() {
        let block = ContentBlock::ToolResult(ToolResultBlock {
            id: "toolu_2".into(),
            output: ToolOutput::Json(json!({"n": 1})),
        });
        let wire = block_to_wire(&block);
        assert_eq!(wire["content"], "{\"n\":1}");
        assert!(wire.get("is_error").is_none());
    }

    /// Haiku 4.5 still speaks budget-token extended thinking; sending adaptive
    /// + output_config.effort is a silent protocol mismatch.
    #[test]
    fn haiku_4_5_uses_extended_thinking_budget() {
        let conv = Conversation::new();
        let body = build_request(&conv, &opts("claude-haiku-4-5-20251001"), false);
        match body.thinking {
            Some(wire::Thinking::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 32_768); // High
            }
            other => panic!("expected Enabled thinking, got {other:?}"),
        }
        assert!(body.output_config.is_none());
    }

    #[test]
    fn non_haiku_uses_adaptive_thinking_and_effort() {
        let conv = Conversation::new();
        let body = build_request(&conv, &opts("claude-sonnet-4-20250514"), false);
        assert!(matches!(
            body.thinking,
            Some(wire::Thinking::Adaptive {
                display: wire::Display::Summarized
            })
        ));
        assert!(matches!(
            body.output_config,
            Some(wire::OutputConfig {
                effort: wire::Effort::High
            })
        ));
    }

    #[test]
    fn multi_turn_request_echoes_thinking_and_coalesced_tool_results() {
        let mut conv = Conversation::new();
        conv.push_user("use the tool");

        let resp = parse_response(wire::MessagesResponse {
            content: vec![
                json!({
                    "type": "thinking",
                    "thinking": "need tool",
                    "signature": "sig-1",
                }),
                json!({
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": { "q": "x" },
                }),
            ],
            stop_reason: Some("tool_use".into()),
            usage: wire::WireUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        })
        .unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        conv.push_response(resp);
        conv.push_tool_result("toolu_1", ToolOutput::Text("answer".into()));
        conv.push_tool_result("toolu_2", ToolOutput::Text("more".into()));

        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tools: vec![ToolDef {
                    name: "lookup".into(),
                    description: "d".into(),
                    input_schema: json!({"type": "object"}),
                }],
                ..Default::default()
            },
            false,
        );

        // user, assistant (thinking+tool), user (two results)
        assert_eq!(body.messages.len(), 3);
        assert_eq!(body.messages[1].role, "assistant");
        assert_eq!(
            body.messages[1].content[0],
            json!({
                "type": "thinking",
                "thinking": "need tool",
                "signature": "sig-1",
            })
        );
        assert_eq!(body.messages[2].role, "user");
        assert_eq!(body.messages[2].content.len(), 2);
        assert_eq!(body.messages[2].content[0]["tool_use_id"], "toolu_1");
        assert_eq!(body.messages[2].content[1]["tool_use_id"], "toolu_2");
    }

    #[test]
    fn map_error_classifies_auth_rate_limit_and_invalid() {
        assert!(matches!(map_error(401, "{}", None), Error::Auth));
        assert!(matches!(
            map_error(429, r#"{"error":{"type":"rate","message":"slow"}}"#, None),
            Error::RateLimited { .. }
        ));
        match map_error(
            400,
            r#"{"error":{"type":"invalid_request_error","message":"bad schema"}}"#,
            None,
        ) {
            Error::InvalidRequest(m) => assert_eq!(m, "bad schema"),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    /// Hosted web search must serialize as Anthropic's server tool, not a
    /// client function tool, and sit alongside any caller ToolDefs.
    #[test]
    fn web_search_emits_hosted_tool_alongside_function_tools() {
        let conv = Conversation::new();
        let lookup = ToolDef {
            name: "lookup".into(),
            description: "d".into(),
            input_schema: json!({"type": "object"}),
        };

        let off = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tools: vec![lookup.clone()],
                web_search: false,
                ..Default::default()
            },
            false,
        );
        let off_json = serde_json::to_value(&off.tools).unwrap();
        assert_eq!(off_json.as_array().unwrap().len(), 1);
        assert_eq!(off_json[0]["name"], "lookup");
        assert!(off_json[0].get("type").is_none());

        let on = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tools: vec![lookup],
                web_search: true,
                ..Default::default()
            },
            false,
        );
        let on_json = serde_json::to_value(&on.tools).unwrap();
        assert_eq!(on_json.as_array().unwrap().len(), 2);
        assert_eq!(on_json[0]["name"], "lookup");
        assert_eq!(on_json[1]["type"], "web_search_20250305");
        assert_eq!(on_json[1]["name"], "web_search");
        // No client-style input_schema on the hosted tool.
        assert!(on_json[1].get("input_schema").is_none());
    }

    #[test]
    fn web_search_off_emits_no_search_tool_when_tools_empty() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                web_search: false,
                ..Default::default()
            },
            false,
        );
        assert!(body.tools.is_empty());
    }

    /// Default Auto + unset parallel omits tool_choice so existing callers
    /// keep the historical wire shape (provider default = auto).
    #[test]
    fn tool_choice_auto_default_omits_field() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tools: vec![ToolDef {
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
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("tool_choice").is_none());
    }

    #[test]
    fn tool_choice_tool_with_parallel_disabled() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tools: vec![ToolDef {
                    name: "play_move".into(),
                    description: "d".into(),
                    input_schema: json!({"type": "object"}),
                }],
                tool_choice: ToolChoice::Tool("play_move".into()),
                parallel_tool_calls: Some(false),
                ..Default::default()
            },
            false,
        );
        let tc = body.tool_choice.as_ref().expect("tool_choice present");
        assert_eq!(tc.kind, "tool");
        assert_eq!(tc.name.as_deref(), Some("play_move"));
        assert_eq!(tc.disable_parallel_tool_use, Some(true));

        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tool_choice"]["type"], "tool");
        assert_eq!(json["tool_choice"]["name"], "play_move");
        assert_eq!(json["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn tool_choice_required_maps_to_any() {
        let conv = Conversation::new();
        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tool_choice: ToolChoice::Required,
                ..Default::default()
            },
            false,
        );
        let tc = body.tool_choice.expect("tool_choice present");
        assert_eq!(tc.kind, "any");
        assert!(tc.name.is_none());
        assert!(tc.disable_parallel_tool_use.is_none());
    }

    #[test]
    fn tool_choice_none_and_auto_with_parallel_override() {
        let conv = Conversation::new();
        let none_body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tool_choice: ToolChoice::None,
                ..Default::default()
            },
            false,
        );
        assert_eq!(none_body.tool_choice.as_ref().map(|t| t.kind), Some("none"));

        // Auto + parallel override still emits tool_choice so disable_parallel lands.
        let auto_parallel = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: Some(false),
                ..Default::default()
            },
            false,
        );
        let tc = auto_parallel.tool_choice.expect("tool_choice present");
        assert_eq!(tc.kind, "auto");
        assert_eq!(tc.disable_parallel_tool_use, Some(true));
    }

    /// Anthropic requires server_tool_use + web_search_tool_result (with
    /// encrypted_content) echoed byte-faithfully on the next turn.
    #[test]
    fn multi_turn_resubmits_web_search_server_artifacts() {
        let mut conv = Conversation::new();
        conv.push_user("latest news?");

        let server_tool_use = json!({
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": { "query": "latest news" },
        });
        let search_result = json!({
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{
                "type": "web_search_result",
                "url": "https://example.com",
                "title": "Example",
                "encrypted_content": "enc-search-blob-xyz",
                "page_age": "April 30, 2025"
            }],
        });

        let resp = parse_response(wire::MessagesResponse {
            content: vec![
                server_tool_use.clone(),
                search_result.clone(),
                json!({ "type": "text", "text": "Here is the news." }),
            ],
            stop_reason: Some("end_turn".into()),
            usage: wire::WireUsage::default(),
        })
        .unwrap();

        // Opaque reasoning path — not client tool_use / tool_result.
        assert!(matches!(
            &resp.message.content[0],
            ContentBlock::Reasoning(_)
        ));
        assert!(matches!(
            &resp.message.content[1],
            ContentBlock::Reasoning(_)
        ));

        // Round-trip each block through block_to_wire before multi-turn build.
        assert_eq!(block_to_wire(&resp.message.content[0]), server_tool_use);
        assert_eq!(block_to_wire(&resp.message.content[1]), search_result);

        conv.push_response(resp);
        conv.push_user("tell me more");

        let body = build_request(
            &conv,
            &RequestOptions {
                model: "claude-sonnet-4-20250514".into(),
                web_search: true,
                ..Default::default()
            },
            false,
        );

        assert_eq!(body.messages.len(), 3);
        assert_eq!(body.messages[1].role, "assistant");
        assert_eq!(body.messages[1].content[0], server_tool_use);
        assert_eq!(body.messages[1].content[1], search_result);
        // encrypted_content must survive intact.
        assert_eq!(
            body.messages[1].content[1]["content"][0]["encrypted_content"],
            "enc-search-blob-xyz"
        );
        let tools_json = serde_json::to_value(&body.tools).unwrap();
        assert_eq!(tools_json[0]["type"], "web_search_20250305");
    }
}
