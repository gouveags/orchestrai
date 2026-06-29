# orchestrai

very fast ai orchestrator sdk

## Scope

The first version is intentionally small:

- a provider abstraction for model calls
- direct HTTP adapters for OpenAI, Anthropic Claude, and AWS Bedrock
- full-response and streaming model calls
- a simple loop that lets models request tools, executes those tools, and calls
  the model again until it returns a final answer
- built-in planning tools for creating, updating, and reading a simple plan
- explicit summary context injection without destructive message history
  mutation

This is not a graph framework.

## Planning

`PlanToolSet` registers three generic tools into any `ToolRegistry`:

- `plan.create` replaces the current plan with ordered pending items
- `plan.update` changes one item by id
- `plan.read` returns the current plan as JSON

The toolset is intentionally small and in-memory. Storage belongs behind a
future runtime/store interface, not in the first generic tool contract.

## Summarization Policy

Summaries are explicit derived context. `SummaryPolicy` prepares model input by
prepending a system message that contains a caller-provided
`ConversationSummary`; it does not delete, rewrite, or compact the original
messages.

`AgentLoopConfig::with_summary_policy(...)` applies the policy when building
provider requests. `LoopOutput.messages` remains the real transcript, and
`LoopOutput.injected_summary` reports the summary context used for the call.

Tools can return `ToolError::result_content(...)` when the model should see the
error and recover in the next loop turn. Unknown tools, hard execution failures,
provider failures, invalid streamed tool arguments, and too many tool rounds
fail the loop directly.

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
