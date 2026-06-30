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
- a root-confined local filesystem environment with basic read, write, list, and
  search tools

This is not a graph framework.

## Planning

`PlanToolSet` registers three generic tools into any `ToolRegistry`:

- `plan_create` replaces the current plan with ordered pending items
- `plan_update` changes one item by id
- `plan_read` returns the current plan as JSON

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

## Artifact Tools

Artifacts are generated outputs intended to outlive a single tool call, such as
reports, charts, summaries, or exported data. `LocalArtifactStore` provides a
small local implementation, and `register_artifact_tools` exposes three generic
tools:

- `artifact_publish` writes a text artifact and returns metadata
- `artifact_list` lists artifacts published through the store
- `artifact_read` reads a text artifact by id

```rust
use std::sync::Arc;

use orchestrai::{
    LocalArtifactStore, ToolRegistry, register_artifact_tools,
};

# fn example() -> Result<(), Box<dyn std::error::Error>> {
let artifacts = Arc::new(LocalArtifactStore::new("./artifacts")?);
let mut tools = ToolRegistry::new();
register_artifact_tools(&mut tools, artifacts);
# Ok(())
# }
```

The local store rejects absolute paths and parent-directory escapes as hard
errors. Remote artifact backends can implement `ArtifactStore` without changing
the agent-facing tool contract.

## Development

```sh
cargo run
cargo test
```

## Agent Family Examples

The agent-family examples are Python scripts that show how to build several
role-selected agents with one shared harness:

- `examples/agent_family_data.py`
- `examples/agent_family_simulation.py`
- `examples/agent_family_records.py`
- `examples/agent_family_knowledge.py` (brief, wiki, and memory modes)
- `examples/agent_family_vehicle.py`
- `examples/agent_family_computing.py`
- `examples/agent_family_real_llm.py` (runs every role, or a selected subset)

They compose synthetic public prompts, role-selected capability bundles,
run-state injection, planning and filesystem tools, usage accounting, and
dynamic sub-agent mounting through the common `agent_run` tool. Domain tools are
intentionally mocked with matching names and permission/exclusion behavior, so
the examples validate orchestration shape without implying access to private
systems.

```sh
maturin develop
python examples/agent_family_data.py
python examples/agent_family_real_llm.py data simulation
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

## Observability and Usage Controls

Telemetry is opt-in and sink-based. Core events describe run lifecycle, model
calls, tool calls, and provider-reported token usage without recording raw
prompts, tool arguments, or tool result content.

```rust
use orchestrai::{TelemetryConfig, TelemetryEvent, TelemetrySink};

#[derive(Clone)]
struct Logger;

impl TelemetrySink for Logger {
    fn record(&self, event: TelemetryEvent) {
        println!("{event:?}");
    }
}

let telemetry = TelemetryConfig::new().with_sink(Logger);
```

Run identity and event storage are also opt-in. `InMemoryRunStore` is useful for
tests and local applications; production hosts can implement `RunStore` to write
sanitized lifecycle events elsewhere. Run-store events intentionally omit raw
prompts, tool arguments, and tool result content.

```rust
use orchestrai::{AgentConfig, InMemoryRunStore};

let run_store = InMemoryRunStore::new();
let config = AgentConfig::new(provider, "gpt-4.1-mini")
    .with_run_store(run_store.clone());
```

Usage accounting is also built into the base agent loop. `UsageMeter` can be
shared across agents, `LoopOutput::usage_snapshot()` reports the per-run delta,
and `UsageLimits` fails closed before provider calls when an existing budget is
already exhausted.

`UsageSnapshot.model_calls` counts logical orchestrai model requests made by the
loop: one `ModelProvider::complete` or `ModelProvider::stream` invocation. It
does not count provider-internal retry or fallback attempts inside an adapter.
The loop reserves model and tool calls before awaiting provider or tool work so
shared meters enforce call limits across concurrent runs.

```rust
use orchestrai::{UsageLimits, UsageMeter};

let meter = UsageMeter::default();
let limits = UsageLimits::default().with_max_total_tokens(10_000);
```

## Python Bindings

Python packaging is scaffolded with `pyo3` and `maturin` behind the `python`
feature. The Python surface supports provider-backed agents, Python-defined
tools, built-in planning/filesystem tool registration, capability prompts,
selected run-state injection, structured outputs, and streaming callbacks:

```python
import orchestrai

agent = orchestrai.create_agent(
    model="gpt-4.1-mini",
    instructions="You are concise and practical.",
    max_tool_rounds=0,
)
print(agent.run("Say hi from Python."))
```

`create_agent` reads `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` unless `api_key=`
is provided. Bedrock remains Rust-only for now. Publishing automation is left
for a follow-up PR.

## Provider Credentials

Provider adapters expect credentials from the caller:

- OpenAI: `OpenAiProvider::new(openai_api_key)`
- Anthropic: `AnthropicProvider::new(anthropic_api_key)`
- Bedrock: `BedrockProvider::new(region, AwsCredentials::new(...))`

`BedrockProvider::from_env(region)` reads `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`.
