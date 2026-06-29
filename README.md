# orchestrai

very fast ai orchestrator sdk

## Scope

The first version is intentionally small:

- a provider abstraction for model calls
- direct HTTP adapters for OpenAI, Anthropic Claude, and AWS Bedrock
- full-response and streaming model calls
- a simple loop that lets models request tools, executes those tools, and calls
  the model again until it returns a final answer
- a root-confined local filesystem environment with basic read, write, list, and
  search tools

This is not a graph framework.

## Local Filesystem Tools

Agent-accessible files are scoped to an explicit environment root. The default
implementation is `LocalEnvironment`, which only accepts relative paths under
that root and refuses root-escape attempts.

```rust
use std::sync::Arc;

use orchestrai::{
    LocalEnvironment, ToolRegistry, register_filesystem_tools,
};

let environment = Arc::new(LocalEnvironment::new("./workspace")?);
let mut tools = ToolRegistry::new();
register_filesystem_tools(&mut tools, environment);
```

The registered tools are `fs_read`, `fs_write`, `fs_list`, and `fs_search`.
Normal file errors, such as a missing file, are returned as tool result content
so the model can recover. Configuration and root-confinement failures are hard
errors.

## Development

```sh
cargo run
cargo test
```

## Provider Credentials

Provider adapters expect credentials from the caller:

- OpenAI: `OpenAiProvider::new(openai_api_key)`
- Anthropic: `AnthropicProvider::new(anthropic_api_key)`
- Bedrock: `BedrockProvider::new(region, AwsCredentials::new(...))`

`BedrockProvider::from_env(region)` reads `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`.
