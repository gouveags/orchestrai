# orchestrai

very fast ai orchestrator sdk

## Scope

The first version is intentionally small:

- a provider abstraction for model calls
- direct HTTP adapters for OpenAI, Anthropic Claude, and AWS Bedrock
- full-response and streaming model calls
- a simple loop that lets models request tools, executes those tools, and calls
  the model again until it returns a final answer

This is not a graph framework.

## Development

```sh
cargo run
cargo test
```

## Rust Agent API

Use `create_agent` when you want the provider-backed tool loop without wiring the
loop internals yourself:

```rust
use orchestrai::{
    AgentConfig, ToolRegistry, create_agent, providers::OpenAiProvider,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?);
let agent = create_agent(
    AgentConfig::new(provider, "gpt-4.1-mini")
        .with_instructions("You are concise and practical.")
        .with_tools(ToolRegistry::new())
        .with_max_tool_rounds(4)
        .with_max_tokens(1024),
);

let output = agent.run("Write a haiku about small APIs.").await?;
println!("{}", output.final_message);
# Ok(())
# }
```

The agent surface intentionally exposes only stable concepts: provider, model,
instructions, tools, max tool rounds, and max tokens. `Agent::run_stream` emits
the existing `LoopEvent` stream while preserving the same provider-backed loop
behavior.

## Python Bindings

Python packaging is scaffolded with `pyo3` and `maturin` behind the `python`
feature. The initial Python surface is intentionally tiny and currently supports
an OpenAI-backed smoke path without Python-defined tools:

```python
import orchestrai

agent = orchestrai.create_agent(
    model="gpt-4.1-mini",
    instructions="You are concise and practical.",
    max_tool_rounds=0,
)
print(agent.run("Say hi from Python."))
```

`create_agent` reads `OPENAI_API_KEY` unless `api_key=` is provided. Python
provider interop, Python-defined tools, streaming callbacks, and publishing
automation are left for follow-up PRs.

## Provider Credentials

Provider adapters expect credentials from the caller:

- OpenAI: `OpenAiProvider::new(openai_api_key)`
- Anthropic: `AnthropicProvider::new(anthropic_api_key)`
- Bedrock: `BedrockProvider::new(region, AwsCredentials::new(...))`

`BedrockProvider::from_env(region)` reads `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`.
