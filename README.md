# hydrogen

Unified multi-provider LLM client for Rust. One conversation model and request API across **Anthropic**, **OpenAI**, and **xAI**.

## Install

```toml
[dependencies]
hydrogen = { git = "https://github.com/lab9-sh/hydrogen" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

## Quick start

```rust
use hydrogen::{AnthropicConfig, Client, Conversation, RequestOptions};

#[tokio::main]
async fn main() -> Result<(), hydrogen::Error> {
    let client = Client::anthropic(AnthropicConfig::from_env()?);

    let mut conv = Conversation::new();
    conv.push_user("Say hello in one sentence.");

    let opts = RequestOptions {
        model: "claude-sonnet-4-20250514".into(),
        ..Default::default()
    };

    let resp = client.send(&conv, &opts).await?;
    conv.push_response(resp);
    Ok(())
}
```

### Other providers

```rust
// OpenAI — OPENAI_API_KEY
let client = Client::openai(OpenAiConfig::from_env()?);

// xAI — XAI_API_KEY
let client = Client::xai(XaiConfig::from_env()?);
```

### Streaming

```rust
use futures_util::StreamExt;
use hydrogen::Event;

let mut stream = client.stream(&conv, &opts).await?;
while let Some(event) = stream.next().await {
    match event? {
        Event::TextDelta(t) => print!("{t}"),
        Event::Done(resp) => {
            conv.push_response(resp);
        }
        _ => {}
    }
}
```

## Design notes

- **Shared types** — `Conversation`, `Message`, `ContentBlock`, tools, thinking effort, and usage are provider-agnostic.
- **Provider pinning** — after the first successful response, a conversation is pinned to that provider; mixing clients returns `Error::ProviderMismatch`.
- **Optional HTTP client** — pass a custom `reqwest::Client` via each `*Config` if you need proxies, timeouts, or shared pools.
- **Volatile fat blocks (opt-in)** — environment loops can mark one user message with `push_volatile_user` / `rotate_volatile_user` and demote it later with `demote_volatile`. Append-only chat callers never set the mark. On Anthropic, when a mark is set, hydrogen places a block-level cache breakpoint on the last content block of the message *before* the volatile and omits top-level `cache_control`; pure-append conversations keep always-on top-level caching. OpenAI/xAI keep `prompt_cache_key` only. See [PROPOSAL-volatile-fat-blocks.md](PROPOSAL-volatile-fat-blocks.md).

## License

Licensed under either of:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option.
