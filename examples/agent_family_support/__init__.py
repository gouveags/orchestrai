from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import orchestrai


DEFAULT_PROMPT = (
    "Hello. Please list the tools you have available and explain what you can "
    "help with. If your instructions prevent naming exact internal tool names, "
    "follow those instructions and describe the capabilities at a high level."
)

BASE_PROMPT = """
You are part of a public orchestrai example suite.

Operate with these rules:
- use the active role and permissions injected in run state;
- prefer a short plan before multi-step work;
- use tools only when they materially help;
- treat mocked tools as examples of integration shape, not real external systems;
- never imply access to private company data, private repositories, or proprietary
  production systems.
""".strip()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    exclude_for_subagent: bool = False
    permission: str | None = None


@dataclass(frozen=True)
class RoleDefinition:
    id: str
    title: str
    bundle: str
    prompt: str
    tools: tuple[ToolSpec, ...]
    subagents: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExampleRunConfig:
    role: str
    is_subagent: bool = False
    permissions: frozenset[str] | None = None

    @property
    def active_permissions(self) -> frozenset[str]:
        return self.permissions or DEFAULT_PERMISSIONS[self.role]


def tool(name: str, description: str) -> ToolSpec:
    return ToolSpec(name=name, description=description)


def subagent_excluded(name: str, description: str) -> ToolSpec:
    return ToolSpec(name=name, description=description, exclude_for_subagent=True)


def permissioned(name: str, description: str, permission: str) -> ToolSpec:
    return ToolSpec(name=name, description=description, permission=permission)


SHARED_TOOLS = (
    tool("run_code", "Run analysis code in the configured sandbox."),
    tool("artifact", "Publish or retrieve generated artifacts."),
    tool("memory", "Search or update durable memories."),
    tool("workspace", "Inspect the sandbox workspace."),
    tool("run_command", "Run shell commands in the sandbox."),
    subagent_excluded("ask_user", "Ask for structured clarification."),
    subagent_excluded("task", "Create or update reusable tasks."),
    subagent_excluded("schedule", "Create or update recurring schedules."),
    subagent_excluded("report_bug", "Report product or tool issues."),
)

DATA_TOOLS = (
    tool("discovery", "Discover datasets, tables, files, and available metrics."),
    tool("query_metrics", "Retrieve metric values for selected entities."),
    tool("read_dataset_file", "Read structured data files."),
    permissioned("define_metric", "Define reusable derived metrics.", "metric_definition"),
    permissioned("database_query", "Run read-only SQL queries.", "database_query"),
    *SHARED_TOOLS,
)

SIMULATION_TOOLS = (
    tool("simulation_config", "List, clone, and edit simulation configurations."),
    tool("simulation_run", "Run a configured simulation study."),
    tool("simulation_results", "Load scalar and time-series simulation results."),
    tool("simulation_docs", "Retrieve simulation reference documentation."),
    subagent_excluded("data_agent", "Delegate data lookup to the data analysis role."),
    *SHARED_TOOLS,
)

RECORDS_TOOLS = (
    tool("record_collection", "Browse engineering record collections."),
    tool("record_item", "Retrieve individual engineering records."),
    tool("record_attachment", "Inspect record attachments and exported files."),
    tool("record_compare", "Compare structured records across runs or revisions."),
    *SHARED_TOOLS,
)

KNOWLEDGE_TOOLS = (
    tool("list_context_threads", "List source threads with new context."),
    tool("dump_thread_messages", "Dump a thread transcript to a workspace file."),
    tool("complete_knowledge_run", "Mark a knowledge maintenance run complete."),
    *SHARED_TOOLS,
)

MEMORY_TOOLS = (
    tool("dump_thread_messages", "Dump a thread transcript to a workspace file."),
    tool("complete_memory_run", "Mark a memory extraction run complete."),
    tool("memory", "Create, search, update, and vote on durable memories."),
    subagent_excluded("report_bug", "Report product or tool issues."),
    tool("workspace", "Inspect the sandbox workspace."),
    tool("run_code", "Run helper code in the sandbox."),
)

VEHICLE_TOOLS = (
    tool("vehicle", "Manage vehicle model assets."),
    tool("maneuver", "Manage maneuver definitions."),
    tool("road", "Manage road or track assets."),
    tool("simulation", "Run vehicle dynamics simulations."),
    tool("output_map", "Configure simulation output channels."),
    *SHARED_TOOLS,
)

COMPUTING_TOOLS = (
    tool("detect_toolboxes", "Detect installed numerical-computing toolboxes."),
    tool("check_script", "Check script files for issues."),
    tool("run_script", "Run scripts in the configured workspace."),
    tool("run_test_file", "Run test files in the configured workspace."),
    tool("model_read", "Read model content."),
    tool("model_edit", "Edit model content."),
    tool("model_test", "Run model tests."),
    *SHARED_TOOLS,
)

ROLES: dict[str, RoleDefinition] = {
    "data": RoleDefinition(
        id="data",
        title="Data analysis",
        bundle="data",
        prompt=(
            "You are a public example data analysis agent. Help users discover "
            "datasets, inspect metrics, compare entities, build visual evidence, "
            "and explain uncertainty."
        ),
        tools=DATA_TOOLS,
        subagents=("records", "simulation"),
    ),
    "simulation": RoleDefinition(
        id="simulation",
        title="Simulation studies",
        bundle="simulation",
        prompt=(
            "You are a public example simulation agent. Help users manage "
            "simulation configurations, run studies, compare results, and "
            "correlate outputs with supplied reference data."
        ),
        tools=SIMULATION_TOOLS,
        subagents=("data",),
    ),
    "records": RoleDefinition(
        id="records",
        title="Engineering records",
        bundle="records",
        prompt=(
            "You are a public example engineering records agent. Help users "
            "retrieve structured records, compare revisions, summarize notes, "
            "and connect record data to analysis outputs."
        ),
        tools=RECORDS_TOOLS,
        subagents=("data",),
    ),
    "knowledge_brief": RoleDefinition(
        id="knowledge_brief",
        title="Knowledge brief",
        bundle="knowledge_brief",
        prompt=(
            "You are a public example knowledge-brief agent. Synthesize recent "
            "thread activity into a concise project brief with decisions, "
            "findings, open questions, and evidence links."
        ),
        tools=KNOWLEDGE_TOOLS,
        subagents=("data", "records"),
    ),
    "knowledge_wiki": RoleDefinition(
        id="knowledge_wiki",
        title="Knowledge wiki",
        bundle="knowledge_wiki",
        prompt=(
            "You are a public example knowledge-wiki agent. Fold reusable "
            "knowledge from source threads into durable wiki-style markdown "
            "pages while avoiding duplication and drift."
        ),
        tools=KNOWLEDGE_TOOLS,
        subagents=("data", "knowledge_brief"),
    ),
    "memory": RoleDefinition(
        id="memory",
        title="Memory maintenance",
        bundle="memory",
        prompt=(
            "You are a public example memory-maintenance agent. Read a source "
            "conversation, extract only durable reusable preferences or lessons, "
            "and produce a short summary for future retrieval."
        ),
        tools=MEMORY_TOOLS,
    ),
    "vehicle": RoleDefinition(
        id="vehicle",
        title="Vehicle simulation",
        bundle="vehicle",
        prompt=(
            "You are a public example vehicle simulation agent. Help configure "
            "vehicle models, maneuvers, roads, solver runs, and post-processing "
            "of generated result files."
        ),
        tools=VEHICLE_TOOLS,
    ),
    "computing": RoleDefinition(
        id="computing",
        title="Numerical computing",
        bundle="computing",
        prompt=(
            "You are a public example numerical-computing agent. Help users "
            "write scripts, run tests, inspect model files, and organize "
            "workspace artifacts in a controlled runtime."
        ),
        tools=COMPUTING_TOOLS,
    ),
}

DEFAULT_PERMISSIONS = {
    "data": frozenset(("metric_definition", "database_query")),
    "simulation": frozenset(("simulation_agent", "data_agent")),
    "records": frozenset(("records_agent", "data_agent")),
    "knowledge_brief": frozenset(("knowledge_agent",)),
    "knowledge_wiki": frozenset(("knowledge_agent",)),
    "memory": frozenset(("memory_agent",)),
    "vehicle": frozenset(("vehicle_agent",)),
    "computing": frozenset(("computing_agent",)),
}

STATE_KEYS = (
    "agent_role",
    "is_subagent",
    "permissions",
    "runtime",
    "sandbox_root",
    "example_project",
)


def base_prompts(role: RoleDefinition, run: ExampleRunConfig) -> list[str]:
    runtime = (
        "# Runtime Selection\n"
        f"agent_role={role.id}\n"
        f"is_subagent={str(run.is_subagent).lower()}\n"
        f"permissions={sorted(run.active_permissions)}"
    )
    return [BASE_PROMPT, runtime]


def role_prompts(role: RoleDefinition) -> dict[str, list[str]]:
    return {role.bundle: [role.prompt]}


def build_state(role: RoleDefinition, run: ExampleRunConfig) -> dict:
    return {
        "agent_role": role.id,
        "is_subagent": run.is_subagent,
        "permissions": sorted(run.active_permissions),
        "runtime": "local",
        "sandbox_root": str(workspace_root(role.id)),
        "example_project": {
            "id": "public-example",
            "name": "Public Orchestrai Example",
        },
        "private_runtime_secret": "not rendered because state policy excludes this key",
    }


def workspace_root(role_id: str) -> Path:
    root = Path(tempfile.gettempdir()) / "orchestrai-agent-family-examples" / role_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_tools(role: RoleDefinition, run: ExampleRunConfig) -> orchestrai.ToolRegistry:
    registry = orchestrai.tools()
    registry.planning()
    registry.filesystem(str(workspace_root(role.id)))
    registry.artifacts(str(workspace_root(role.id) / "artifacts"))

    for spec in role.tools:
        if spec.exclude_for_subagent and run.is_subagent:
            continue
        if spec.permission and spec.permission not in run.active_permissions:
            continue
        if spec.name == "artifact":
            continue
        register_mock_tool(registry, spec)

    if not run.is_subagent and role.subagents:
        register_agent_run_tool(registry, role.subagents)

    return registry


def register_mock_tool(registry: orchestrai.ToolRegistry, spec: ToolSpec) -> None:
    def handler(arguments: dict, name: str = spec.name) -> dict:
        return {
            "tool": name,
            "status": "mocked",
            "arguments": arguments,
            "porting_note": (
                "This public example exercises orchestration shape only. "
                "Replace the mock with an application adapter in real deployments."
            ),
        }

    registry.register(
        spec.name,
        f"{spec.description}\n\nMocked in this public orchestrai example.",
        handler,
        {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "query": {"type": "string"},
                "id": {"type": "string"},
            },
            "additionalProperties": True,
        },
    )


def register_agent_run_tool(registry: orchestrai.ToolRegistry, agents: Iterable[str]) -> None:
    agent_ids = list(agents)

    def handler(arguments: dict) -> dict:
        return {
            "agent": arguments.get("agent"),
            "status": "mocked_subagent_completed",
            "input": arguments.get("input"),
            "state": arguments.get("state", {}),
            "excluded_tools": arguments.get("exclude_tools", []),
        }

    registry.register(
        "agent_run",
        "Run one mounted sub-agent through the common interface.",
        handler,
        {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": agent_ids},
                "input": {"type": "string"},
                "state": {"type": "object"},
                "exclude_tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["agent", "input"],
            "additionalProperties": False,
        },
    )


def build_agent(
    role_name: str,
    *,
    provider: str,
    model: str,
    max_tokens: int = 450,
    max_tool_rounds: int = 1,
) -> tuple[object, RoleDefinition, ExampleRunConfig]:
    role = ROLES[role_name]
    run = ExampleRunConfig(role=role_name)
    agent = orchestrai.create_agent(
        provider=provider,
        model=model,
        instructions="You are the shared orchestrai public example harness.",
        tools=build_tools(role, run),
        prompts=base_prompts(role, run),
        bundles=role_prompts(role),
        state_keys=list(STATE_KEYS),
        model_modes={
            "fast": os.getenv("ORCHESTRAI_FAST_MODEL", model),
            "regular": os.getenv("ORCHESTRAI_REGULAR_MODEL", model),
            "max": os.getenv("ORCHESTRAI_MAX_MODEL", model),
        },
        usage_limits={"max_total_tokens": 20_000},
        track_runs=True,
        max_tokens=max_tokens,
        max_tool_rounds=max_tool_rounds,
    )
    return agent, role, run


def run_role(
    role_name: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    stream: bool = False,
) -> dict:
    model = model or os.getenv("ORCHESTRAI_REAL_LLM_MODEL", "claude-sonnet-4-6")
    agent, role, run = build_agent(role_name, provider=provider, model=model)
    state = build_state(role, run)

    if stream:
        result = agent.stream(
            prompt,
            lambda delta: print(delta, end="", flush=True),
            state=state,
            capabilities=[role.bundle],
            model_mode="regular",
        )
        return result.to_dict()

    result = agent.run_full(
        prompt,
        state=state,
        capabilities=[role.bundle],
        model_mode="regular",
    )
    return result.to_dict()


def print_role_output(role_name: str, output: dict) -> None:
    print(f"===== {role_name} =====")
    print(output["final_message"])
    print(f"usage: {json.dumps(output['usage'], sort_keys=True)}")


def run_cli(default_roles: Iterable[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=os.getenv("ORCHESTRAI_PROVIDER", "anthropic"))
    parser.add_argument("--model", default=os.getenv("ORCHESTRAI_REAL_LLM_MODEL"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("roles", nargs="*", choices=sorted(ROLES))
    args = parser.parse_args()

    for role_name in args.roles or list(default_roles):
        output = run_role(
            role_name,
            provider=args.provider,
            model=args.model,
            prompt=args.prompt,
            stream=args.stream,
        )
        if args.stream:
            print()
        print_role_output(role_name, output)
