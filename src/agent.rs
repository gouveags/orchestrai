use crate::{
    loop_runner::{AgentLoop, AgentLoopConfig, LoopError, LoopEvent, LoopOutput},
    provider::ModelProvider,
    tool::{Tool, ToolRegistry},
    types::Message,
};

pub type AgentOutput = LoopOutput;

pub fn create_agent<P>(config: AgentConfig<P>) -> Agent<P>
where
    P: ModelProvider,
{
    Agent::new(config)
}

pub struct AgentConfig<P> {
    pub provider: P,
    pub model: String,
    pub instructions: Option<String>,
    pub tools: ToolRegistry,
    pub max_tool_rounds: usize,
    pub max_tokens: Option<u32>,
}

impl<P> AgentConfig<P> {
    pub fn new(provider: P, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            instructions: None,
            tools: ToolRegistry::new(),
            max_tool_rounds: 8,
            max_tokens: None,
        }
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.register(tool);
        self
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = max_tool_rounds;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

pub struct Agent<P> {
    loop_runner: AgentLoop<P>,
    instructions: Option<String>,
}

impl<P> Agent<P>
where
    P: ModelProvider,
{
    pub fn new(config: AgentConfig<P>) -> Self {
        let mut loop_config =
            AgentLoopConfig::new(config.model).with_max_tool_rounds(config.max_tool_rounds);
        if let Some(max_tokens) = config.max_tokens {
            loop_config = loop_config.with_max_tokens(max_tokens);
        }

        Self {
            loop_runner: AgentLoop::new(config.provider, config.tools, loop_config),
            instructions: config.instructions,
        }
    }

    pub async fn run(&self, input: impl Into<String>) -> Result<AgentOutput, LoopError> {
        self.run_messages(vec![Message::user(input.into())]).await
    }

    pub async fn run_messages(&self, messages: Vec<Message>) -> Result<AgentOutput, LoopError> {
        self.loop_runner.run(self.prepare_messages(messages)).await
    }

    pub async fn run_stream<F, Fut>(
        &self,
        input: impl Into<String>,
        on_event: F,
    ) -> Result<AgentOutput, LoopError>
    where
        F: FnMut(LoopEvent) -> Fut + Send,
        Fut: std::future::Future<Output = ()> + Send,
    {
        self.run_messages_stream(vec![Message::user(input.into())], on_event)
            .await
    }

    pub async fn run_messages_stream<F, Fut>(
        &self,
        messages: Vec<Message>,
        on_event: F,
    ) -> Result<AgentOutput, LoopError>
    where
        F: FnMut(LoopEvent) -> Fut + Send,
        Fut: std::future::Future<Output = ()> + Send,
    {
        self.loop_runner
            .run_stream(self.prepare_messages(messages), on_event)
            .await
    }

    fn prepare_messages(&self, messages: Vec<Message>) -> Vec<Message> {
        let Some(instructions) = &self.instructions else {
            return messages;
        };

        let mut prepared = Vec::with_capacity(messages.len() + 1);
        prepared.push(Message::system(instructions.clone()));
        prepared.extend(messages);
        prepared
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{
        provider::{ModelRequest, ModelResponse, ModelStream, ModelStreamEvent, ProviderResult},
        tool::FnTool,
        types::{Role, ToolCall, ToolDefinition},
    };

    #[tokio::test]
    async fn run_prepends_configured_instructions_to_simple_input() {
        let last_request = Arc::new(Mutex::new(None));
        let agent = create_agent(
            AgentConfig::new(
                FakeProvider::new(ModelResponse::text("final"), Arc::clone(&last_request)),
                "fake-model",
            )
            .with_instructions("You are terse."),
        );

        let output = agent.run("hello").await.unwrap();

        assert_eq!(output.final_message, "final");
        let request = last_request.lock().unwrap();
        let messages = &request.as_ref().unwrap().messages;
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are terse.");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "hello");
    }

    #[tokio::test]
    async fn run_keeps_tools_behind_the_clean_agent_api() {
        let agent = create_agent(
            AgentConfig::new(
                FakeProvider::sequence(
                    vec![
                        ModelResponse {
                            message: String::new(),
                            tool_calls: vec![ToolCall {
                                id: "call_1".to_owned(),
                                name: "double".to_owned(),
                                arguments: json!({"value": 4}),
                            }],
                            usage: None,
                        },
                        ModelResponse::text("8"),
                    ],
                    Arc::new(Mutex::new(None)),
                ),
                "fake-model",
            )
            .with_max_tool_rounds(1)
            .with_tool(FnTool::new(
                ToolDefinition::new(
                    "double",
                    "Double an integer.",
                    json!({
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"]
                    }),
                ),
                |arguments| {
                    Box::pin(
                        async move { Ok((arguments["value"].as_i64().unwrap() * 2).to_string()) },
                    )
                },
            )),
        );

        let output = agent.run("double 4").await.unwrap();

        assert_eq!(output.final_message, "8");
        assert_eq!(output.tool_results[0].content, "8");
    }

    struct FakeProvider {
        responses: Mutex<Vec<ModelResponse>>,
        last_request: Arc<Mutex<Option<ModelRequest>>>,
    }

    impl FakeProvider {
        fn new(response: ModelResponse, last_request: Arc<Mutex<Option<ModelRequest>>>) -> Self {
            Self::sequence(vec![response], last_request)
        }

        fn sequence(
            mut responses: Vec<ModelResponse>,
            last_request: Arc<Mutex<Option<ModelRequest>>>,
        ) -> Self {
            responses.reverse();
            Self {
                responses: Mutex::new(responses),
                last_request,
            }
        }
    }

    impl ModelResponse {
        fn text(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                tool_calls: Vec::new(),
                usage: None,
            }
        }
    }

    #[async_trait]
    impl ModelProvider for FakeProvider {
        async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
            *self.last_request.lock().unwrap() = Some(request);
            Ok(self.responses.lock().unwrap().pop().unwrap())
        }

        async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
            Ok(Box::pin(futures_util::stream::iter(vec![Ok(
                ModelStreamEvent::Done,
            )])))
        }
    }
}
