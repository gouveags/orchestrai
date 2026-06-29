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

## Provider Credentials

Provider adapters expect credentials from the caller:

- OpenAI: `OpenAiProvider::new(openai_api_key)`
- Anthropic: `AnthropicProvider::new(anthropic_api_key)`
- Bedrock: `BedrockProvider::new(region, AwsCredentials::new(...))`

`BedrockProvider::from_env(region)` reads `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`.
