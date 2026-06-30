# OrchestrAI

OrchestrAI is a small, fast AI orchestration SDK for building agents that call
LLMs, use tools, stream responses, work with files, and keep lightweight run
visibility.

It exists because most agent frameworks make simple workflows feel like graph
engineering. OrchestrAI starts from the common case: call a model, let it use
tools, keep going until it has an answer, and make the important runtime knobs
easy to understand.

The project is Rust-first, with Python bindings for teams that want to use the
same agent runtime from Python applications.

## What You Can Build

- Chat-style agents that call tools until they return a final answer.
- Streaming agents that emit token deltas and tool events as they run.
- Agents with provider-backed model aliases and fallback models.
- Role-based agents that load different prompts and tools at runtime.
- Local file and artifact workflows for reports, summaries, generated data, and
  workspace files.
- Agents with planning, summarization, usage limits, telemetry, and sanitized
  run history.
- Python applications that use the same core agent API.

OrchestrAI is not trying to be a graph framework. If your workflow is mostly
"model call, maybe tools, maybe another model call", the API should stay small.

## Major Changes In This Version

- `create_agent` is the main user-facing API for Rust and Python.
- OpenAI, Anthropic Claude, and AWS Bedrock providers are available.
- Python bindings can create provider-backed agents, register Python tools, and
  run text, full-result, or streaming calls.
- Model aliases and fallback policies let users ask for names like
  `team-regular` while routing to concrete provider models.
- Run state can be injected into prompts with an explicit allowlist.
- Capability bundles let users activate different prompts and tools by role.
- A default sub-agent tool can expose child agents through one common interface.
- Planning tools, filesystem tools, and artifact tools are built in.
- Usage accounting, run limits, token limits, telemetry, and sanitized run
  events are part of the base agent loop.
- Public Python examples now show a shared "agent family" setup without
  duplicating prompts, tools, and state for every role.

## Install

Rust:

```toml
[dependencies]
orchestrai = { git = "https://github.com/gouveags/orchestrai" }
```

Python development install:

```sh
maturin develop
```

The Python package is scaffolded but not published yet.

## Create An Agent

Rust:

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

Python:

```python
import orchestrai

agent = orchestrai.create_agent(
    provider="openai",
    model="gpt-4.1-mini",
    instructions="You are concise and practical.",
    max_tool_rounds=4,
)

response = agent.run("Say hi from Python.")
print(response)
```

Python `create_agent` reads `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` unless
`api_key=` is passed directly. Bedrock reads standard AWS credentials and
`AWS_REGION` or `AWS_DEFAULT_REGION`.

Use `agent.run_full(...)` when you want the run id, final message, transcript,
tool results, and usage.

## Stream Responses

Rust:

```rust
let output = agent
    .run_stream("Explain the plan.", |event| async move {
        println!("{event:?}");
    })
    .await?;
```

Python:

```python
def on_delta(text: str) -> None:
    print(text, end="", flush=True)

def on_event(event: dict) -> None:
    if event["type"] == "tool_started":
        print(f"\nusing {event['name']}")

result = agent.stream("Explain the plan.", on_delta=on_delta, on_event=on_event)
print(result.usage)
```

Streaming preserves the same instructions, state, tools, model routing, usage
accounting, and run tracking as full-response calls.

## Add Tools

Tools are normal functions with a JSON schema. Recoverable tool errors can be
returned to the model so it can correct itself on the next loop turn; hard
errors fail the run.

Rust:

```rust
use orchestrai::{FnTool, ToolDefinition};
use serde_json::json;

let lookup = FnTool::new(
    ToolDefinition::new(
        "lookup_order",
        "Look up an order by id.",
        json!({
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }),
    ),
    |arguments| {
        Box::pin(async move {
            Ok(format!(r#"{{"order":{arguments}}}"#))
        })
    },
);

let agent = create_agent(
    AgentConfig::new(provider, "gpt-4.1-mini").with_tool(lookup),
);
```

Python:

```python
tools = orchestrai.tools()
tools.register(
    name="lookup_order",
    description="Look up an order by id.",
    handler=lambda args: {"order": args["order_id"], "status": "ready"},
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)

agent = orchestrai.create_agent(
    model="gpt-4.1-mini",
    tools=tools,
)
```

## Use Providers And Fallback Models

Use a provider directly when one model is enough:

```rust
use orchestrai::providers::{AnthropicProvider, OpenAiProvider};

let openai = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?);
let anthropic = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY")?);
```

Use a routed provider when you want user-facing model names and fallback
behavior:

```rust
use std::sync::Arc;

use orchestrai::{
    FallbackPolicy, ModelCatalog, ProviderModel, ProviderRegistry,
    RoutedModelProvider,
};

let provider = RoutedModelProvider::new(
    ProviderRegistry::new()
        .with_provider("anthropic", Arc::new(anthropic))
        .with_provider("openai", Arc::new(openai)),
    ModelCatalog::new().with_alias(
        "team-regular",
        [
            ProviderModel::new("anthropic", "claude-sonnet-4-6"),
            ProviderModel::new("openai", "gpt-4.1"),
        ],
    ),
)
.with_fallback_policy(FallbackPolicy::transient_provider_errors());

let agent = create_agent(AgentConfig::new(provider, "team-regular"));
```

Fallbacks are explicit. Hard provider errors surface directly.

Python can use the same idea with plain dictionaries:

```python
import os
import orchestrai

agent = orchestrai.create_agent(
    model="team-regular",
    providers={
        "anthropic": {"api_key": os.environ["ANTHROPIC_API_KEY"]},
        "openai": {"api_key": os.environ["OPENAI_API_KEY"]},
    },
    models={
        "team-regular": [
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            {"provider": "openai", "model": "gpt-4.1-mini"},
        ],
    },
    fallback="transient",
)
```

When `providers=` and `models=` are set, that routed configuration is the
source of truth and the single-provider `provider=` option is not used.

## Pass Runtime State

Run state is structured data provided per run. You decide which keys are allowed
to enter the model instructions.

```rust
use orchestrai::{RunOptions, RunState, StateInstructionPolicy};
use serde_json::json;

let agent = create_agent(
    AgentConfig::new(provider, "team-regular")
        .with_run_state_instructions(StateInstructionPolicy::selected([
            "tenant_name",
            "plan_tier",
        ])),
);

let output = agent
    .run_with_options(
        RunOptions::new("Draft the customer reply.").with_state(
            RunState::from_json(json!({
                "tenant_name": "Acme",
                "plan_tier": "pro",
                "private_note": "not sent to the model"
            }))?,
        ),
    )
    .await?;
```

This keeps runtime inputs useful without letting every field become prompt
context by accident.

Python:

```python
agent = orchestrai.create_agent(
    model="team-regular",
    state_keys=["tenant_name", "plan_tier"],
)

result = agent.run(
    "Draft the customer reply.",
    state={
        "tenant_name": "Acme",
        "plan_tier": "pro",
        "private_note": "not sent to the model",
    },
)
```

## Load Prompts And Tools By Role

Capability bundles let you define a base agent once and activate extra prompts
or tools per run.

```rust
use orchestrai::{CapabilityBundle, CapabilityBundleSet, CapabilitySelection};

let bundles = CapabilityBundleSet::new()
    .with_default(CapabilityBundle::new("base").with_prompt("Follow the house style."))
    .with_bundle(
        "data",
        CapabilityBundle::new("data").with_prompt("You can inspect datasets."),
    );

let agent = create_agent(
    AgentConfig::new(provider, "team-regular").with_capability_bundles(bundles),
);

let output = agent
    .run_with_capabilities(
        "Summarize the latest file.",
        CapabilitySelection::new(["data"]),
    )
    .await?;
```

This is the main pattern for role-based agents and permission-selected tools.

## Mount Sub-Agents

OrchestrAI includes a default sub-agent tool so a parent agent can call child
agents through one common `agent_run`-style interface. Use it when you want
separate roles behind one parent agent without hand-writing a new tool for each
child.

Sub-agent mounts can be limited by permission so a runtime role only sees the
children it is allowed to call.

## Built-In Tools

### Planning

Register planning tools when the model should maintain a simple plan:

- `plan_create`
- `plan_update`
- `plan_read`

```rust
use orchestrai::{PlanToolSet, ToolRegistry};

let mut tools = ToolRegistry::new();
PlanToolSet::new().register(&mut tools);
```

### Filesystem

Filesystem tools are scoped to a root directory:

- `fs_read`
- `fs_write`
- `fs_list`
- `fs_search`

```rust
use std::sync::Arc;

use orchestrai::{
    LocalEnvironment, ToolRegistry, register_filesystem_tools,
};

let environment = Arc::new(LocalEnvironment::new("./workspace")?);
let mut tools = ToolRegistry::new();
register_filesystem_tools(&mut tools, environment);
```

Missing files are returned as tool-visible errors so the model can recover.
Root escapes and invalid configuration fail the run.

### Artifacts

Artifacts are generated outputs that should be listed or read later:

- `artifact_publish`
- `artifact_list`
- `artifact_read`

```rust
use std::sync::Arc;

use orchestrai::{
    LocalArtifactStore, ToolRegistry, register_artifact_tools,
};

let artifacts = Arc::new(LocalArtifactStore::new("./artifacts")?);
let mut tools = ToolRegistry::new();
register_artifact_tools(&mut tools, artifacts);
```

The local artifact store rejects absolute paths, parent-directory escapes, and
existing symlinks that point outside the artifact root.

Python tool registries can also enable the built-ins:

```python
tools = orchestrai.tools()
tools.planning()
tools.filesystem("./workspace")
tools.artifacts("./artifacts")

print([tool["name"] for tool in tools.definitions()])
```

## Summarize Without Losing Messages

Summaries are added as explicit derived context. They do not delete, rewrite, or
compact the transcript.

Today, summary policies are available on the lower-level `AgentLoopConfig`.
Use this when you need direct loop control and want OrchestrAI to prepend a
summary before model calls while preserving the real transcript.

```rust
use orchestrai::{
    AgentLoopConfig, ConversationSummary, SummaryPolicy,
};

let loop_config = AgentLoopConfig::new("team-regular").with_summary_policy(
    SummaryPolicy::always(ConversationSummary::new(
        "The user prefers short answers.",
    )),
);
```

`LoopOutput.messages` remains the real transcript. `LoopOutput.injected_summary`
reports the summary used for the request.

## Observe Runs And Control Usage

Observability is off by default. Enable only the pieces you want.

Telemetry emits lifecycle, model, tool, and usage events:

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

Run stores give each run an id and sanitized lifecycle history:

```rust
use orchestrai::InMemoryRunStore;

let run_store = InMemoryRunStore::new();
let agent = create_agent(
    AgentConfig::new(provider, "team-regular")
        .with_run_store(run_store.clone()),
);
```

Run events intentionally omit raw prompts, tool arguments, and tool result
content.

Usage meters can be shared across agents:

```rust
use orchestrai::{UsageLimits, UsageMeter};

let meter = UsageMeter::default();
let limits = UsageLimits::default()
    .with_max_runs(100)
    .with_max_total_tokens(1_000_000);

let agent = create_agent(
    AgentConfig::new(provider, "team-regular")
        .with_usage_meter(meter.clone())
        .with_usage_limits(limits),
);
```

Limits fail closed before new expensive work starts.

Python:

```python
agent = orchestrai.create_agent(
    model="gpt-4.1-mini",
    usage_limits={"max_runs": 100, "max_total_tokens": 1_000_000},
    track_runs=True,
)

result = agent.run_full("Say hi.")
print(result.usage)
print(agent.usage())
print(agent.run_events())
```

## Run The Examples

The Python examples show how to build several role-selected agents from one
shared setup:

- `examples/agent_family_data.py`
- `examples/agent_family_simulation.py`
- `examples/agent_family_records.py`
- `examples/agent_family_knowledge.py`
- `examples/agent_family_vehicle.py`
- `examples/agent_family_computing.py`
- `examples/agent_family_real_llm.py`

```sh
maturin develop
python examples/agent_family_data.py
python examples/agent_family_real_llm.py data simulation
```

The examples use synthetic public prompts and mock domain tools. They are meant
to demonstrate orchestration shape, permissions, state injection, built-in
tools, usage tracking, and sub-agent mounting.

## Provider Credentials

- OpenAI Rust provider: `OpenAiProvider::new(openai_api_key)`
- Anthropic Rust provider: `AnthropicProvider::new(anthropic_api_key)`
- Bedrock Rust provider:
  `BedrockProvider::new(region, AwsCredentials::new(...))`
- Bedrock from environment:
  `BedrockProvider::from_env(region)`

`BedrockProvider::from_env(region)` reads:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- optional `AWS_SESSION_TOKEN`

## Development

Common local checks:

```sh
cargo fmt --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo check --features python
cargo check --tests --features python
cargo check --examples
python -m py_compile scripts/hello_agent.py examples/agent_family_support/__init__.py examples/agent_family_*.py
```
