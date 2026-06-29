use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream;
use orchestrai::{
    AgentConfig, AgentLoop, AgentLoopConfig, DefaultSubAgentTool, FnTool, LoopError, Message,
    ModelProvider, ModelRequest, ModelResponse, ModelStream, Tool, ToolCall, ToolDefinition,
    ToolError, ToolRegistry,
    provider::{ModelStreamEvent, ProviderResult},
    subagents::{SubAgentDefinition, SubAgentMountPermissions, SubAgentRegistry},
};
use serde_json::{Value, json};

const DEFAULT_SUB_AGENT_TOOL_NAME: &str = "agent_run";

#[tokio::test]
async fn default_sub_agent_tool_runs_named_agent_with_scoped_payload_inside_parent_loop() {
    let child_requests = Arc::new(Mutex::new(Vec::new()));
    let child_provider = FakeProvider::sequence(
        vec![text_response(
            "child summary: Rust ownership prevents shared mutation",
        )],
        Arc::clone(&child_requests),
    );
    let mut sub_agents = SubAgentRegistry::new();
    sub_agents.register(
        SubAgentDefinition::new(
            "researcher",
            AgentConfig::new(child_provider, "fake-child-model")
                .with_instructions("Summarize only the scoped payload."),
        )
        .with_capabilities(["research", "summarize"])
        .with_mount_permissions(SubAgentMountPermissions::runtime_mountable()),
    );
    let parent_requests = Arc::new(Mutex::new(Vec::new()));
    let parent_provider = FakeProvider::sequence(
        vec![
            tool_response(
                "call_subagent",
                DEFAULT_SUB_AGENT_TOOL_NAME,
                json!({
                    "agent": "researcher",
                    "input": "Summarize the ownership note.",
                    "state": {
                        "topic": "rust ownership",
                        "source": "scoped parent state"
                    }
                }),
            ),
            text_response("parent final after child"),
        ],
        Arc::clone(&parent_requests),
    );
    let mut tools = ToolRegistry::new();
    tools.register(DefaultSubAgentTool::new(sub_agents));
    let loop_runner = AgentLoop::new(
        parent_provider,
        tools,
        AgentLoopConfig::new("fake-parent-model").with_max_tool_rounds(1),
    );

    let output = loop_runner
        .run(vec![Message::user(
            "Ask the researcher, but do not leak parent-only context.",
        )])
        .await
        .unwrap();

    assert_eq!(output.final_message, "parent final after child");
    assert_eq!(output.tool_results[0].name, DEFAULT_SUB_AGENT_TOOL_NAME);
    let tool_content: Value = serde_json::from_str(&output.tool_results[0].content).unwrap();
    assert_eq!(
        tool_content["final_message"],
        "child summary: Rust ownership prevents shared mutation"
    );
    assert_eq!(tool_content["agent"], "researcher");

    let child_requests = child_requests.lock().unwrap();
    assert_eq!(child_requests.len(), 1);
    assert_eq!(child_requests[0].model, "fake-child-model");
    assert!(
        child_requests[0]
            .messages
            .iter()
            .any(|message| message.content.contains("Summarize the ownership note."))
    );
    assert!(
        child_requests[0]
            .messages
            .iter()
            .any(|message| message.content.contains("rust ownership"))
    );
    assert!(
        child_requests[0]
            .messages
            .iter()
            .all(|message| !message.content.contains("parent-only context"))
    );
}

#[tokio::test]
async fn runtime_mount_permissions_limit_which_sub_agents_are_exposed_to_the_parent() {
    let mut sub_agents = SubAgentRegistry::new();
    sub_agents.register(
        SubAgentDefinition::new(
            "researcher",
            AgentConfig::new(
                FakeProvider::sequence(vec![text_response("ok")], Arc::new(Mutex::new(vec![]))),
                "fake-child-model",
            ),
        )
        .with_capabilities(["research"])
        .with_mount_permissions(SubAgentMountPermissions::runtime_mountable()),
    );
    sub_agents.register(
        SubAgentDefinition::new(
            "billing_auditor",
            AgentConfig::new(
                FakeProvider::sequence(
                    vec![text_response("forbidden")],
                    Arc::new(Mutex::new(vec![])),
                ),
                "fake-private-model",
            ),
        )
        .with_capabilities(["billing", "private"])
        .with_mount_permissions(SubAgentMountPermissions::not_runtime_mountable()),
    );

    let tool = DefaultSubAgentTool::new(sub_agents);
    let definition = tool.definition();

    assert_eq!(definition.name, DEFAULT_SUB_AGENT_TOOL_NAME);
    assert!(definition.description.contains("research"));
    assert_json_array_contains(
        &definition.input_schema,
        &["properties", "agent", "enum"],
        "researcher",
    );
    assert_json_array_excludes(
        &definition.input_schema,
        &["properties", "agent", "enum"],
        "billing_auditor",
    );
}

#[tokio::test]
async fn sub_agent_recoverable_tool_errors_are_returned_as_parent_visible_tool_content() {
    let mut child_tools = ToolRegistry::new();
    child_tools.register(FnTool::new(
        ToolDefinition::new(
            "case_file.read",
            "Read a case file that may be absent.",
            json!({
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"]
            }),
        ),
        |_arguments| {
            Box::pin(async {
                Err(ToolError::result_content(
                    r#"{"error":"case file missing","recoverable":true}"#,
                ))
            })
        },
    ));
    let child_provider = FakeProvider::sequence(
        vec![
            tool_response("child_tool", "case_file.read", json!({"id": "case-123"})),
            text_response("child recovered with a caveat"),
        ],
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut sub_agents = SubAgentRegistry::new();
    sub_agents.register(
        SubAgentDefinition::new(
            "case_reviewer",
            AgentConfig::new(child_provider, "fake-child-model").with_tools(child_tools),
        )
        .with_capabilities(["case_review"])
        .with_mount_permissions(SubAgentMountPermissions::runtime_mountable()),
    );
    let parent_provider = FakeProvider::sequence(
        vec![
            tool_response(
                "call_subagent",
                DEFAULT_SUB_AGENT_TOOL_NAME,
                json!({
                    "agent": "case_reviewer",
                    "input": "Review case-123",
                    "state": {}
                }),
            ),
            text_response("parent recovered too"),
        ],
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut tools = ToolRegistry::new();
    tools.register(DefaultSubAgentTool::new(sub_agents));
    let loop_runner = AgentLoop::new(
        parent_provider,
        tools,
        AgentLoopConfig::new("fake-parent-model").with_max_tool_rounds(1),
    );

    let output = loop_runner
        .run(vec![Message::user("review it")])
        .await
        .unwrap();

    assert_eq!(output.final_message, "parent recovered too");
    assert!(!output.tool_results[0].is_error);
    let tool_content: Value = serde_json::from_str(&output.tool_results[0].content).unwrap();
    assert_eq!(
        tool_content["final_message"],
        "child recovered with a caveat"
    );
    assert_eq!(tool_content["tool_results"][0]["is_error"], true);
    assert_eq!(
        tool_content["tool_results"][0]["content"],
        r#"{"error":"case file missing","recoverable":true}"#
    );
}

#[tokio::test]
async fn missing_or_forbidden_sub_agent_mounts_fail_the_parent_loop_hard() {
    let mut sub_agents = SubAgentRegistry::new();
    let private_requests = Arc::new(Mutex::new(Vec::new()));
    sub_agents.register(
        SubAgentDefinition::new(
            "private_reviewer",
            AgentConfig::new(
                FakeProvider::sequence(
                    vec![text_response("forbidden")],
                    Arc::clone(&private_requests),
                ),
                "fake-private-model",
            ),
        )
        .with_mount_permissions(SubAgentMountPermissions::not_runtime_mountable()),
    );

    let missing_error = run_parent_once_with_sub_agent_call(
        sub_agents.clone(),
        json!({
            "agent": "unknown_reviewer",
            "input": "try unknown",
            "state": {}
        }),
    )
    .await
    .unwrap_err();
    assert_unknown_sub_agent_mount_error(missing_error, "unknown_reviewer");

    let forbidden_error = run_parent_once_with_sub_agent_call(
        sub_agents,
        json!({
            "agent": "private_reviewer",
            "input": "try private",
            "state": {}
        }),
    )
    .await
    .unwrap_err();
    assert_forbidden_sub_agent_mount_error(forbidden_error, "private_reviewer");
    assert!(
        private_requests.lock().unwrap().is_empty(),
        "forbidden sub-agent provider must not be called"
    );
}

#[tokio::test]
async fn non_object_sub_agent_state_fails_before_calling_child_provider() {
    let child_requests = Arc::new(Mutex::new(Vec::new()));
    let mut sub_agents = SubAgentRegistry::new();
    sub_agents.register(
        SubAgentDefinition::new(
            "researcher",
            AgentConfig::new(
                FakeProvider::sequence(
                    vec![text_response("should not run")],
                    Arc::clone(&child_requests),
                ),
                "fake-child-model",
            ),
        )
        .with_mount_permissions(SubAgentMountPermissions::runtime_mountable()),
    );

    let error = run_parent_once_with_sub_agent_call(
        sub_agents,
        json!({
            "agent": "researcher",
            "input": "try malformed state",
            "state": ["not", "an", "object"]
        }),
    )
    .await
    .unwrap_err();

    match error {
        LoopError::Tool(ToolError::Execution(message)) => {
            assert!(message.contains("researcher"), "{message}");
            assert!(message.contains("state must be an object"), "{message}");
        }
        other => panic!("expected malformed sub-agent state failure, got {other:?}"),
    }
    assert!(child_requests.lock().unwrap().is_empty());
}

async fn run_parent_once_with_sub_agent_call(
    sub_agents: SubAgentRegistry,
    arguments: Value,
) -> Result<orchestrai::LoopOutput, LoopError> {
    let parent_provider = FakeProvider::sequence(
        vec![tool_response(
            "call_subagent",
            DEFAULT_SUB_AGENT_TOOL_NAME,
            arguments,
        )],
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut tools = ToolRegistry::new();
    tools.register(DefaultSubAgentTool::new(sub_agents));
    AgentLoop::new(
        parent_provider,
        tools,
        AgentLoopConfig::new("fake-parent-model").with_max_tool_rounds(1),
    )
    .run(vec![Message::user("mount a sub-agent")])
    .await
}

fn assert_unknown_sub_agent_mount_error(error: LoopError, expected_agent: &str) {
    match error {
        LoopError::Tool(ToolError::NotFound(name)) => {
            assert_eq!(name, expected_agent);
        }
        other => panic!("expected unknown sub-agent mount failure, got {other:?}"),
    }
}

fn assert_forbidden_sub_agent_mount_error(error: LoopError, expected_agent: &str) {
    match error {
        LoopError::Tool(ToolError::Execution(message)) => {
            assert!(message.contains(expected_agent), "{message}");
            assert!(message.contains("not runtime mountable"), "{message}");
        }
        other => panic!("expected forbidden sub-agent mount failure, got {other:?}"),
    }
}

fn text_response(message: impl Into<String>) -> ModelResponse {
    ModelResponse {
        message: message.into(),
        tool_calls: Vec::new(),
        usage: None,
    }
}

fn tool_response(id: &str, name: &str, arguments: Value) -> ModelResponse {
    ModelResponse {
        message: String::new(),
        tool_calls: vec![ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments,
        }],
        usage: None,
    }
}

fn assert_json_array_contains(schema: &Value, path: &[&str], expected: &str) {
    let values = json_path_array(schema, path);
    assert!(
        values.iter().any(|value| value == expected),
        "expected {expected:?} in {values:?}"
    );
}

fn assert_json_array_excludes(schema: &Value, path: &[&str], forbidden: &str) {
    let values = json_path_array(schema, path);
    assert!(
        values.iter().all(|value| value != forbidden),
        "did not expect {forbidden:?} in {values:?}"
    );
}

fn json_path_array(schema: &Value, path: &[&str]) -> Vec<String> {
    let mut cursor = schema;
    for segment in path {
        cursor = cursor
            .get(*segment)
            .unwrap_or_else(|| panic!("missing schema segment {segment:?} in {schema}"));
    }
    cursor
        .as_array()
        .unwrap_or_else(|| panic!("schema path {path:?} was not an array in {schema}"))
        .iter()
        .map(|value| value.as_str().unwrap().to_owned())
        .collect()
}

struct FakeProvider {
    responses: Mutex<Vec<ModelResponse>>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

impl FakeProvider {
    fn sequence(
        mut responses: Vec<ModelResponse>,
        requests: Arc<Mutex<Vec<ModelRequest>>>,
    ) -> Self {
        responses.reverse();
        Self {
            responses: Mutex::new(responses),
            requests,
        }
    }
}

#[async_trait]
impl ModelProvider for FakeProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        Ok(self.responses.lock().unwrap().pop().unwrap())
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        Ok(Box::pin(stream::iter(vec![Ok(ModelStreamEvent::Done)])))
    }
}
