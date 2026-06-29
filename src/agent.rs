use crate::{
    capabilities::{CapabilityBundleSet, CapabilitySelection},
    loop_runner::{
        AgentLoop, AgentLoopConfig, BeforeModelCallHook, LoopError, LoopEvent, LoopOutput,
        RunStateCallOptions,
    },
    provider::{CachePolicy, ModelProvider},
    run_state::{BeforeModelCall, RunOptions, RunState, StateInstructionPolicy},
    tool::{Tool, ToolRegistry},
    types::Message,
};

use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};

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
    pub capability_bundles: CapabilityBundleSet,
    pub cache_policy: CachePolicy,
    pub(crate) run_state_instructions: Option<StateInstructionPolicy>,
    pub(crate) model_modes: BTreeMap<String, String>,
    pub(crate) before_model_call: Option<BeforeModelCallHook>,
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
            capability_bundles: CapabilityBundleSet::new(),
            cache_policy: CachePolicy::disabled(),
            run_state_instructions: None,
            model_modes: BTreeMap::new(),
            before_model_call: None,
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

    pub fn with_capability_bundles(mut self, capability_bundles: CapabilityBundleSet) -> Self {
        self.capability_bundles = capability_bundles;
        self
    }

    pub fn with_cache_policy(mut self, cache_policy: CachePolicy) -> Self {
        self.cache_policy = cache_policy;
        self
    }

    pub fn with_run_state_instructions(mut self, policy: StateInstructionPolicy) -> Self {
        self.run_state_instructions = Some(policy);
        self
    }

    pub fn with_model_modes<I, K, V>(mut self, modes: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.model_modes = modes
            .into_iter()
            .map(|(mode, model)| (mode.into(), model.into()))
            .collect();
        self
    }

    pub fn with_before_model_call<F, Fut>(mut self, hook: F) -> Self
    where
        F: Fn(BeforeModelCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<BeforeModelCall, LoopError>> + Send + 'static,
    {
        self.before_model_call = Some(Arc::new(move |call| {
            Box::pin(hook(call))
                as Pin<Box<dyn Future<Output = Result<BeforeModelCall, LoopError>> + Send>>
        }));
        self
    }
}

pub struct Agent<P> {
    loop_runner: AgentLoop<P>,
    instructions: Option<String>,
    run_state_options: RunStateCallOptions,
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
        loop_config = loop_config
            .with_capability_bundles(config.capability_bundles)
            .with_cache_policy(config.cache_policy);

        Self {
            loop_runner: AgentLoop::new(config.provider, config.tools, loop_config),
            instructions: config.instructions,
            run_state_options: RunStateCallOptions {
                instruction_policy: config.run_state_instructions,
                model_modes: config.model_modes,
                model_mode: None,
                before_model_call: config.before_model_call,
            },
        }
    }

    pub async fn run(&self, input: impl Into<String>) -> Result<AgentOutput, LoopError> {
        self.run_messages(vec![Message::user(input.into())]).await
    }

    pub async fn run_with_capabilities(
        &self,
        input: impl Into<String>,
        selection: CapabilitySelection,
    ) -> Result<AgentOutput, LoopError> {
        self.run_messages_with_capabilities(vec![Message::user(input.into())], selection)
            .await
    }

    pub async fn run_with_state(
        &self,
        input: impl Into<String>,
        state: RunState,
    ) -> Result<AgentOutput, LoopError> {
        self.run_messages_with_state_and_mode(vec![Message::user(input.into())], state, None)
            .await
    }

    pub async fn run_with_options(&self, options: RunOptions) -> Result<AgentOutput, LoopError> {
        self.run_messages_with_state_and_mode(
            vec![Message::user(options.input)],
            options.state,
            options.model_mode,
        )
        .await
    }

    pub async fn run_messages(&self, messages: Vec<Message>) -> Result<AgentOutput, LoopError> {
        self.loop_runner.run(self.prepare_messages(messages)).await
    }

    pub async fn run_messages_with_capabilities(
        &self,
        messages: Vec<Message>,
        selection: CapabilitySelection,
    ) -> Result<AgentOutput, LoopError> {
        self.loop_runner
            .run_with_capabilities(self.prepare_messages(messages), selection)
            .await
    }

    pub async fn run_messages_with_state(
        &self,
        messages: Vec<Message>,
        state: RunState,
    ) -> Result<AgentOutput, LoopError> {
        self.run_messages_with_state_and_mode(messages, state, None)
            .await
    }

    async fn run_messages_with_state_and_mode(
        &self,
        messages: Vec<Message>,
        state: RunState,
        model_mode: Option<String>,
    ) -> Result<AgentOutput, LoopError> {
        let mut options = self.run_state_options.clone();
        options.model_mode = model_mode;

        self.loop_runner
            .run_with_state(self.prepare_messages(messages), state, options)
            .await
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
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = create_agent(
            AgentConfig::new(
                FakeProvider::new(ModelResponse::text("final"), Arc::clone(&requests)),
                "fake-model",
            )
            .with_instructions("You are terse."),
        );

        let output = agent.run("hello").await.unwrap();

        assert_eq!(output.final_message, "final");
        let request = requests.lock().unwrap();
        let messages = &request[0].messages;
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "You are terse.");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "hello");
    }

    #[tokio::test]
    async fn run_keeps_tools_behind_the_clean_agent_api() {
        let requests = Arc::new(Mutex::new(Vec::new()));
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
                    Arc::clone(&requests),
                ),
                "fake-model",
            )
            .with_max_tool_rounds(1)
            .with_max_tokens(64)
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
        let requests = requests.lock().unwrap();
        assert_eq!(requests[0].model, "fake-model");
        assert_eq!(requests[0].max_tokens, Some(64));
        assert_eq!(requests[0].tools.len(), 1);
        assert_eq!(requests[0].tools[0].name, "double");
    }

    struct FakeProvider {
        responses: Mutex<Vec<ModelResponse>>,
        requests: Arc<Mutex<Vec<ModelRequest>>>,
    }

    impl FakeProvider {
        fn new(response: ModelResponse, requests: Arc<Mutex<Vec<ModelRequest>>>) -> Self {
            Self::sequence(vec![response], requests)
        }

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
            self.requests.lock().unwrap().push(request);
            Ok(self.responses.lock().unwrap().pop().unwrap())
        }

        async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
            Ok(Box::pin(futures_util::stream::iter(vec![Ok(
                ModelStreamEvent::Done,
            )])))
        }
    }
}
