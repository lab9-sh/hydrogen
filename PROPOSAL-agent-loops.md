# Proposal: hydrogen support for long-running agent loops

Status: **proposed** (not implemented on `main`)
Motivating consumer: [indium](../indium) — human-vs-LLM Go (19×19, ~300 moves, one model turn per move)

This document captures API surface that was briefly implemented on hydrogen
(`5fe2f4f`, `38f05eb`) and then reverted in favor of keeping the append-only
conversation model intact until the design is deliberate. Usage cache telemetry
(originally bundled with the breakpoint work, refined in `54698ae`) **stays** on
main — it is portable, additive, and required to observe caching at all.

## Motivation

hydrogen models a conversation as an **append-only** transcript the caller never
edits. That is the right shape for chat (carbon, simple tool loops). It does not
work for a long-running *environment* loop, where the same state is re-presented
every turn in an updated form and the transcript would otherwise accumulate
hundreds of superseded copies.

The concrete driver is a Go game. Each turn the model needs the board plus
derived analysis (groups in atari, liberty counts, ko, captures, scratchpad).
Rendered, that block is roughly 700–900 tokens. A full game is 250–350 moves.

| approach | context at move 250 | correctness |
|---|---|---|
| board in every turn | ~200k tokens | 249 of 250 boards are stale and wrong |
| board only in the tail, rebuilt each turn | ~12k tokens (user side) | exactly one board, always current |

The second approach — the **tail state block** — is what this proposal enables.

### The tail state block pattern

One “fat” block (board + analysis + scratchpad) always sits at the end of the
transcript. Once the model has responded, it is **demoted in place** to a
one-line stub, and a freshly rendered fat block is appended.

```
turn 47 sends:  [system, thin_1, a_1, …, thin_46, a_46, FAT_47]
                                                          ^ ~900 tok

turn 48 sends:  [system, thin_1, a_1, …, thin_46, a_46, thin_47, a_47, FAT_48]
                └──────────── unchanged, cache hit ──────┘ └── reprocessed ──┘
```

Demotion replaces rather than deletes, so the assistant turn above the block is
still answering something that exists. Because the edit is always at a fixed
depth from the end, the stable prefix can grow monotonically — *if* the cache
breakpoint is placed correctly (see feature 3).

---

## Summary

| # | Feature | Priority | Blocking? |
|---|---|---|---|
| 1 | [Mutable / constructible transcripts](#1-mutable--constructible-transcripts) | high | yes — pattern is impossible without it |
| 2 | [`tool_choice` and parallel-call control](#2-tool_choice-and-parallel-call-control) | high | no, but affects wire correctness |
| 3 | [Explicit prompt-cache breakpoints](#3-explicit-prompt-cache-breakpoints) | high | yes for Anthropic cost — demotion without this is *worse* than no caching |
| — | Usage cache fields + cross-provider mapping | done | already on main (`Usage::cache_*`, `total_input_tokens`, OpenAI/xAI `map_usage`) |

---

## 1. Mutable / constructible transcripts

### Problem

`Conversation` is append-only *and* cannot be rebuilt from outside the crate.

- `Conversation.messages` is private; `messages()` returns `&[Message]` only —
  no `messages_mut`, `retain`, `truncate`, or `drain`.
- Rebuilding a filtered copy fails: assistant turns only append via
  `push_response(Response)`, and `Response.provider` is `pub(crate)`, so
  downstream code cannot construct a `Response`.
- User-side content is partly unreachable: `TextBlock.extras` is `pub(crate)`
  with no public constructor, so a struct literal will not compile downstream.
  `ToolUseBlock::new` exists; `TextBlock` / `ToolResultBlock` do not.

Net effect: a consumer cannot edit, filter, or reconstruct a transcript by any
means the public API provides.

### Proposed API

```rust
impl Conversation {
    /// Mutable access to the transcript for in-place rewriting.
    pub fn messages_mut(&mut self) -> &mut Vec<Message>;

    /// Append a caller-constructed turn. Does not change the provider pin.
    pub fn push_message(&mut self, msg: Message);

    /// Rebuild from parts, preserving the provider pin. Assigns a fresh
    /// `cache_key` so restored transcripts do not collide with an in-flight
    /// conversation's prompt-cache entry.
    pub fn from_parts(messages: Vec<Message>, provider: Option<ProviderKind>) -> Self;
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self;   // extras: None
}

impl ToolResultBlock {
    pub fn new(id: impl Into<String>, output: ToolOutput) -> Self;
}
```

`messages_mut` is the smallest hole that unblocks demotion; `from_parts` is
worth having for save/load. Both are needed because `Conversation` serializes
with private fields — restoring from disk today goes through serde, which works
but is not an intentional API.

### Client example (tail demotion)

```rust
use hydrogen::types::{ContentBlock, Message, Role, TextBlock, ToolOutput};
use hydrogen::{Client, Conversation, RequestOptions, StopReason, ThinkingEffort};

fn fat_state_block(game: &Game) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text(TextBlock::new(/* board + analysis */))],
    }
}

fn thin_record(n: u32, played: Move, reply: Move) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text(TextBlock::new(format!(
            "Move {n} — you played {played}. White replied {reply}."
        )))],
    }
}

async fn play_one_move(
    client: &Client,
    conv: &mut Conversation,
    game: &mut Game,
    opts: &RequestOptions,
) -> Result<(), hydrogen::Error> {
    // Demote previous fat block, then append the current one.
    if let Some(last) = conv.messages_mut().last_mut() {
        *last = thin_record(game.move_number(), game.last_own_move(), game.last_reply());
    }
    conv.push_message(fat_state_block(game));

    for _ in 0..3 {
        let resp = client.send(conv, opts).await?;
        let tool_call = resp.message.content.iter().find_map(|b| match b {
            ContentBlock::ToolUse(t) if t.name == "play_move" => Some(t.clone()),
            _ => None,
        });
        conv.push_response(resp);

        let Some(call) = tool_call else { continue };
        let point: String = call.input["point"].as_str().unwrap_or_default().into();

        match game.try_play(&point) {
            Ok(()) => {
                conv.push_tool_result(&call.id, ToolOutput::Text("accepted".into()));
                return Ok(());
            }
            Err(why) => {
                conv.push_tool_result(&call.id, ToolOutput::Error(format!("{point}: {why}")));
            }
        }
    }

    game.pass();
    Ok(())
}
```

### Why it is required

Without demotion every turn’s board stays forever. At ~900 tokens/board that is
~200k of context by endgame, of which all but the last board is stale. The
failure mode is not only cost — it is a model that has seen hundreds of
contradictory boards.

Front-truncation is not a substitute: it invalidates the cached prefix from the
first surviving message and throws away the model’s own commentary.

### Impact on “one conversation model”

This is the load-bearing design trade-off.

| Promise | Effect of `messages_mut` |
|---|---|
| One shared type across providers | **intact** — still `Conversation` / `Message` / `ContentBlock` |
| One `Client::send(conv, opts)` entry point | **intact** |
| Hydrogen owns multi-turn invariants (append-only, valid wire shape) | **relaxed** — callers can orphan tool results, strip signed reasoning mid-loop, rewrite under a cache breakpoint |

The brief implementation chose the wide hole (`&mut Vec<Message>`). That is the
smallest *code* change and the largest *semantic* change: the conversation model
stays one type, but stops being a closed lifecycle.

#### Alternatives to consider before shipping

1. **Narrow demotion API (preferred if we want to keep ownership)**  
   ```rust
   conv.replace_user_content(index, …);
   // or higher-level:
   conv.demote_tail(stub);
   conv.append_state(fat);
   ```
   Keeps tool_result identity, forbids casual middle-history rewrites, and
   documents the pattern hydrogen is willing to support.

2. **Ephemeral context, append-only history**  
   `client.send_with_context(&conv, &opts, ContextBlock { … })` each turn.  
   Conversation stays chat-shaped; environment state never lives in the
   transcript. Harder to express “fat board rides in the unanswered
   `tool_result`” continuous tool loops.

3. **Rebuild every turn via `from_parts` only**  
   Environment is source of truth; no in-place edit. Still needs constructors
   and (for Anthropic) breakpoints; loses stable object identity unless
   `cache_key` is reattached carefully.

4. **Raw `messages_mut` (what was shipped, then reverted)**  
   Maximum flexibility for consumers (strip old reasoning, collapse retries).
   Shifts wire-invariant burden entirely to the app. Acceptable only with
   strong docs and examples of the failure modes below.

### Implementation notes

- `push_message` must not touch the provider pin — only `push_response` does.
- `ReasoningBlock` should stay non-constructible (provider-signed). Rebuilt
  transcripts that drop reasoning are fine *between* turns; **not** mid
  tool-loop on the turn being continued (Anthropic rejects a tool result whose
  preceding assistant turn lost its thinking block).
- Demote message *content*; never drop messages that a later tool_result
  depends on.

### Lessons from the indium POC (when this was implemented)

1. **Demoting the wrong block kind orphans tool results.** After a successful
   `play_move`, the transcript ends `[…, assistant(tool_use), user(tool_result)]`.
   Replacing `last_mut()` with a plain text message leaves the tool_use
   unanswered and Anthropic rejects the next request. Demotion must preserve
   block kind: a `tool_result` stays a `tool_result` with the same id, stub as
   output.

2. **Assistant turns dominate growth.** Demotion only controls the user half.
   With thinking on, measured growth was ~427 tok/move — almost all retained
   assistant output. Stripping old `ReasoningBlock`s via `messages_mut` works
   and needs no extra API, but fights the cache breakpoint (see feature 3).

3. **Carbon broke at compile time**, not at runtime: it builds `RequestOptions`
   with an exhaustive struct literal. Any additive fields need either
   `..Default::default()` at call sites or a careful major-version story.
   (Feature 2 and 3 share this concern.)

---

## 2. `tool_choice` and parallel-call control

### Problem

`RequestOptions` exposes `tools` but no way to require a call or disable
parallel calls. For a game loop:

1. The model can answer with prose instead of `play_move`.
2. Nothing prevents `play_move` + `update_notes` in one turn; the loop then has
   to invent ordering.

All three backends already support both knobs:

| provider | forced call | disable parallel |
|---|---|---|
| Anthropic | `tool_choice: {"type": "tool", "name": …}` | `tool_choice.disable_parallel_tool_use` |
| OpenAI (Responses) | `tool_choice: {"type": "function", "name": …}` | `parallel_tool_calls: false` |
| xAI | OpenAI-compatible | OpenAI-compatible |

### Proposed API

```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,           // model decides (current behavior)
    Required,       // must call some tool
    Tool(String),   // must call this specific tool
    None,           // tools visible but not callable
}

pub struct RequestOptions {
    // ...existing fields...
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// `None` leaves the provider default untouched.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
}
```

`parallel_tool_calls` is `Option<bool>` deliberately: `RequestOptions` derives
`Default`, and a bare `bool` would default to `false`, silently changing
behavior for every existing caller.

### Mapping notes

- **Anthropic:** fold parallel control into `disable_parallel_tool_use` on the
  `tool_choice` object. `Auto` + unset parallel should omit the field entirely
  (preserve historical wire shape).
- **OpenAI / xAI:** `tool_choice` as string or `{type, name}`; separate
  `parallel_tool_calls` field.
- Default `Auto` / `None` must leave the wire identical to today.

### Lesson from indium

Forced `ToolChoice::Tool("play_move")` (and even `Required` / Anthropic `any`)
**silently disables extended thinking** on Claude: the API accepts the request
but returns only `[tool_use]` with zero thinking tokens. Under `Auto`, the same
model called `play_move` every turn in a 32-call sample *and* produced
reasoning.

Keep the portable surface — it is correct and useful — but do **not** document
forced tool choice as the default recipe for agent loops that also want
thinking. Prefer `Auto` + `parallel_tool_calls: Some(false)` for move turns,
and reserve `Tool(…)` for cases where a call is mandatory and reasoning is not.

This feature does **not** weaken the conversation model; it strengthens the
shared request API.

---

## 3. Explicit prompt-cache breakpoints

### Problem

hydrogen’s Anthropic adapter marks the **whole prompt** with a top-level
`cache_control: ephemeral`. That is correct for pure append-only chat: each turn
extends a prefix that still matches.

It is **wrong** for demotion. Divergence between turn *n* and turn *n+1* is
exactly at the fat carrier (it becomes a stub). The stored cache entry is the
entire previous prompt, which is no longer a prefix of the next one → 100% miss
**and** a full paid cache write every turn. Measured on indium: demotion without
placement control served 1–6% of input tokens and was strictly worse than
disabling caching.

OpenAI / xAI cache automatically by prefix; they ignore explicit breakpoints.
The portable knob must be meaningful on Anthropic and a no-op elsewhere (or
eventually auto-derived).

### Proposed API

```rust
pub struct RequestOptions {
    // ...existing fields...
    /// Place the prompt-cache breakpoint this many messages from the end,
    /// instead of implicitly at the end of the whole prompt.
    ///
    /// `Some(1)` marks the second-to-last message — the invariant a rewriting
    /// loop needs: the tail is volatile, everything before it is not.
    ///
    /// Anthropic only; ignored by backends that cache automatically.
    /// When set, the whole-prompt marker is dropped (keeping both would cache
    /// the volatile tail too).
    #[serde(default)]
    pub cache_breakpoint_from_end: Option<usize>,
}
```

Adapter behavior (Anthropic):

1. Resolve index `len.saturating_sub(n + 1)` when in range; else fall back to
   whole-prompt marker.
2. Attach `cache_control: { type: ephemeral }` on the last content block of that
   message.
3. Omit top-level `cache_control` when an explicit breakpoint is active.

### Measured outcome (indium, 30-move games)

| | cache served | `cache_rd` at move 30 | uncached / turn |
|---|---|---|---|
| top-level marker only + demotion | 1–6% | 0 | grows to ~21k |
| `cache_breakpoint_from_end: Some(1)` | **~65%** | 4332, growing | **flat ~900–1300** |

### Portability concern

This is the only proposed field that is honestly provider-shaped. Options:

| approach | pro | con |
|---|---|---|
| Explicit `cache_breakpoint_from_end` (shipped then reverted) | Clear, testable, documents the pattern | Anthropic-only semantics on a shared struct |
| Auto: hash messages, breakpoint at first changed index | Shared API stays clean | Stateful, easy to get wrong with retries/streaming |
| Portable “volatile tail length” concept adapters ignore when N/A | Names the intent, not the vendor | Still a special case |

Recommendation: if feature 1 ships as raw `messages_mut`, ship this too (or
automatic placement) in the **same** change — demotion without breakpoints is a
footgun. If feature 1 ships as a narrow demotion API, hydrogen can place the
breakpoint internally and keep `RequestOptions` free of Anthropic knobs.

### Interaction with stripping old reasoning

`--keep-reasoning N` (consumer-side strip via `messages_mut`) rewrites assistant
turns *at or before* the breakpoint, so the cached prefix never stabilizes.
Measured: breakpoint alone beat breakpoint + strip on effective cost. Treat
stripping and caching as substitutes; document that rewriting history under the
breakpoint freezes `cache_read`.

---

## Already on main: usage cache telemetry

Not part of this proposal’s open work — retained when the agent-loop commits
were reverted:

- `Usage::cache_creation_input_tokens`
- `Usage::cache_read_input_tokens`
- `Usage::total_input_tokens()` — needed because with caching on, `input_tokens`
  alone is the *uncached remainder* (can read as `2` against a multi-k cached
  prefix)
- OpenAI / xAI adapters split `input_tokens_details.cached_tokens` out of the
  full input total so the portable shape matches Anthropic

Agent-loop consumers (and carbon, if it wants accurate context display) should
use `total_input_tokens()` for sizing and the cache fields for hit-rate
telemetry.

---

## Non-goals

- **A built-in truncation or summarization helper.** Policy belongs to the
  consumer; the crate only needs to stop preventing it (or own demotion as a
  first-class op).
- **Making `ReasoningBlock` constructible.** Payloads are provider-signed.
- **Relaxing provider pinning.** Cross-provider transcript reuse is a hazard.
- **A local tokenizer.** Three providers, three vocabularies.
- **Usage accounting on `Conversation`.** Per-`Response` `Usage` is enough;
  cumulative totals stay in the consumer.
- **Retry policy for transient failures.** Typed errors already suffice.
- **A second conversation type** (“chat” vs “agent”). Prefer one type with a
  carefully opened lifecycle over forking the model.

---

## Suggested ship order

1. **Decide the mutability shape** (narrow demotion API vs `messages_mut`) —
   this decides whether breakpoints are public or internal.
2. **Ship mutability + Anthropic breakpoint placement together** so demotion
   never lands without a viable cache story.
3. **Ship `ToolChoice` / `parallel_tool_calls`** independently anytime — pure
   portable surface; document the thinking interaction.
4. **Fix downstream exhaustive `RequestOptions` literals** (carbon) with
   `..Default::default()` before or in the same release.

## Acceptance criteria (when implemented)

- [ ] Tail-state demotion works without orphaning tool_use / tool_result pairs
- [ ] Anthropic: with volatile tail + breakpoint, cache read grows monotonically
      over ≥30 turns; uncached input stays roughly flat
- [ ] OpenAI / xAI: new options are no-ops or map correctly; no wire regression
      when defaults are used
- [ ] Default `RequestOptions` produces the same wire shape as today for all
      three providers
- [ ] carbon (or any exhaustive struct literal) builds after the additive fields
- [ ] Docs state: forced tool choice may suppress thinking; do not rewrite
      messages under the active cache breakpoint if you care about hits
