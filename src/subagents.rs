use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    agent::{Agent, AgentConfig},
    loop_runner::{LoopError, LoopOutput},
    provider::ModelProvider,
    tool::{Tool, ToolError, ToolResult},
    types::{Message, ToolDefinition},
};

pub const DEFAULT_SUB_AGENT_TOOL_NAME: &str = "agent_run";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubAgentMountPermissions {
    runtime_mountable: bool,
}

impl SubAgentMountPermissions {
    pub fn runtime_mountable() -> Self {
        Self {
            runtime_mountable: true,
        }
    }

    pub fn not_runtime_mountable() -> Self {
        Self {
            runtime_mountable: false,
        }
    }

    pub fn allows_runtime_mount(&self) -> bool {
        self.runtime_mountable
    }
}

impl Default for SubAgentMountPermissions {
    fn default() -> Self {
        Self::not_runtime_mountable()
    }
}

#[derive(Clone)]
pub struct SubAgentDefinition {
    id: String,
    capabilities: Vec<String>,
    mount_permissions: SubAgentMountPermissions,
    runner: Arc<dyn SubAgentRunner>,
}

impl SubAgentDefinition {
    pub fn new<P>(id: impl Into<String>, config: AgentConfig<P>) -> Self
    where
        P: ModelProvider + 'static,
    {
        Self {
            id: id.into(),
            capabilities: Vec::new(),
            mount_permissions: SubAgentMountPermissions::default(),
            runner: Arc::new(ConfiguredSubAgent {
                agent: Agent::new(config),
            }),
        }
    }

    pub fn with_capabilities<I, S>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.capabilities = capabilities.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_mount_permissions(mut self, permissions: SubAgentMountPermissions) -> Self {
        self.mount_permissions = permissions;
        self
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    pub fn mount_permissions(&self) -> &SubAgentMountPermissions {
        &self.mount_permissions
    }
}

#[derive(Clone, Default)]
pub struct SubAgentRegistry {
    agents: BTreeMap<String, SubAgentDefinition>,
}

impl SubAgentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, definition: SubAgentDefinition) {
        self.agents.insert(definition.id.clone(), definition);
    }

    pub fn get(&self, id: &str) -> Option<&SubAgentDefinition> {
        self.agents.get(id)
    }

    pub fn runtime_mountable(&self) -> impl Iterator<Item = &SubAgentDefinition> {
        self.agents
            .values()
            .filter(|definition| definition.mount_permissions.allows_runtime_mount())
    }
}

pub struct DefaultSubAgentTool {
    registry: SubAgentRegistry,
}

impl DefaultSubAgentTool {
    pub fn new(registry: SubAgentRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for DefaultSubAgentTool {
    fn definition(&self) -> ToolDefinition {
        let agents = self
            .registry
            .runtime_mountable()
            .map(|definition| definition.id().to_owned())
            .collect::<Vec<_>>();
        let description = self.tool_description();

        ToolDefinition::new(
            DEFAULT_SUB_AGENT_TOOL_NAME,
            description,
            json!({
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": agents
                    },
                    "input": {
                        "type": "string"
                    },
                    "state": {
                        "type": "object"
                    }
                },
                "required": ["agent", "input"]
            }),
        )
    }

    async fn execute(&self, arguments: Value) -> ToolResult<String> {
        let call: SubAgentCall = serde_json::from_value(arguments)
            .map_err(|error| ToolError::Execution(format!("invalid sub-agent call: {error}")))?;
        let definition = self
            .registry
            .get(&call.agent)
            .ok_or_else(|| ToolError::NotFound(call.agent.clone()))?;

        if !definition.mount_permissions().allows_runtime_mount() {
            return Err(ToolError::Execution(format!(
                "sub-agent `{}` is not runtime mountable",
                call.agent
            )));
        }

        let state = call.state.unwrap_or_else(|| json!({}));
        if !state.is_object() {
            return Err(ToolError::Execution(format!(
                "sub-agent `{}` state must be an object",
                call.agent
            )));
        }

        let output = definition
            .runner
            .run(call.input, state)
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;

        serde_json::to_string(&json!({
            "agent": definition.id(),
            "final_message": output.final_message,
            "tool_results": output.tool_results,
        }))
        .map_err(|error| ToolError::Execution(format!("serialize sub-agent output: {error}")))
    }
}

impl DefaultSubAgentTool {
    fn tool_description(&self) -> String {
        let agents = self
            .registry
            .runtime_mountable()
            .map(|definition| {
                if definition.capabilities().is_empty() {
                    definition.id().to_owned()
                } else {
                    format!(
                        "{} ({})",
                        definition.id(),
                        definition.capabilities().join(", ")
                    )
                }
            })
            .collect::<Vec<_>>();

        if agents.is_empty() {
            "Run one of the runtime-mountable sub-agents with a scoped input and optional object state.".to_owned()
        } else {
            format!(
                "Run one of the runtime-mountable sub-agents with a scoped input and optional object state. Available agents: {}.",
                agents.join("; ")
            )
        }
    }
}

#[derive(Deserialize)]
struct SubAgentCall {
    agent: String,
    input: String,
    state: Option<Value>,
}

#[async_trait]
trait SubAgentRunner: Send + Sync {
    async fn run(&self, input: String, state: Value) -> Result<LoopOutput, LoopError>;
}

struct ConfiguredSubAgent<P> {
    agent: Agent<P>,
}

#[async_trait]
impl<P> SubAgentRunner for ConfiguredSubAgent<P>
where
    P: ModelProvider + 'static,
{
    async fn run(&self, input: String, state: Value) -> Result<LoopOutput, LoopError> {
        self.agent
            .run_messages(vec![Message::user(render_child_input(&input, &state))])
            .await
    }
}

fn render_child_input(input: &str, state: &Value) -> String {
    let state = serde_json::to_string_pretty(state).unwrap_or_else(|_| state.to_string());
    format!("Input:\n{input}\n\nState:\n{state}")
}
