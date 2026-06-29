use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};

use futures_util::StreamExt;

use crate::{
    capabilities::{CapabilityBundleSet, CapabilityError, CapabilitySelection},
    provider::{CacheHint, CacheHintScope, CachePolicy, ProviderFeature},
    provider::{ModelProvider, ModelRequest, ModelResponse, ModelStreamEvent, ProviderError},
    run_state::{BeforeModelCall, ModelCallConfig, RunState, StateInstructionPolicy},
    summarization::{ConversationSummary, PreparedMessages, SummaryPolicy},
    telemetry::{TelemetryConfig, TelemetryEvent},
    tool::{ToolError, ToolRegistry},
    types::{Message, Role, ToolCall, ToolResult},
    usage::{UsageLimitError, UsageLimitKind, UsageLimits, UsageMeter, UsageSnapshot},
};

pub(crate) type BeforeModelCallHook = Arc<
    dyn Fn(
            BeforeModelCall,
        ) -> Pin<Box<dyn Future<Output = Result<BeforeModelCall, LoopError>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone, Default)]
pub(crate) struct RunStateCallOptions {
    pub instruction_policy: Option<StateInstructionPolicy>,
    pub model_modes: BTreeMap<String, String>,
    pub model_mode: Option<String>,
    pub before_model_call: Option<BeforeModelCallHook>,
}

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: String,
    pub max_tool_rounds: usize,
    pub max_tokens: Option<u32>,
    pub summary_policy: Option<SummaryPolicy>,
    pub capability_bundles: CapabilityBundleSet,
    pub cache_policy: CachePolicy,
    pub telemetry: TelemetryConfig,
    pub usage_meter: UsageMeter,
    pub usage_limits: UsageLimits,
}

impl AgentLoopConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tool_rounds: 8,
            max_tokens: None,
            summary_policy: None,
            capability_bundles: CapabilityBundleSet::new(),
            cache_policy: CachePolicy::disabled(),
            telemetry: TelemetryConfig::default(),
            usage_meter: UsageMeter::default(),
            usage_limits: UsageLimits::default(),
        }
    }

    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = max_tool_rounds;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_summary_policy(mut self, summary_policy: SummaryPolicy) -> Self {
        self.summary_policy = Some(summary_policy);
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

    pub fn with_telemetry(mut self, telemetry: TelemetryConfig) -> Self {
        self.telemetry = telemetry;
        self
    }

    pub fn with_usage_meter(mut self, usage_meter: UsageMeter) -> Self {
        self.usage_meter = usage_meter;
        self
    }

    pub fn with_usage_limits(mut self, usage_limits: UsageLimits) -> Self {
        self.usage_limits = usage_limits;
        self
    }
}

pub struct AgentLoop<P> {
    provider: P,
    tools: ToolRegistry,
    config: AgentLoopConfig,
}

impl<P> AgentLoop<P>
where
    P: ModelProvider,
{
    pub fn new(provider: P, tools: ToolRegistry, config: AgentLoopConfig) -> Self {
        Self {
            provider,
            tools,
            config,
        }
    }

    pub async fn run(&self, messages: Vec<Message>) -> Result<LoopOutput, LoopError> {
        self.run_with_capabilities(messages, CapabilitySelection::default())
            .await
    }

    pub async fn run_with_capabilities(
        &self,
        messages: Vec<Message>,
        selection: CapabilitySelection,
    ) -> Result<LoopOutput, LoopError> {
        let mut usage_snapshot = self.start_run()?;
        let output = self
            .run_with_capabilities_inner(messages, selection, &mut usage_snapshot)
            .await;
        self.finish_run(output.is_ok());
        output
    }

    async fn run_with_capabilities_inner(
        &self,
        messages: Vec<Message>,
        selection: CapabilitySelection,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<LoopOutput, LoopError> {
        let context = self.run_context(&selection)?;
        let mut messages = apply_capability_prompts(messages, &context.prompts);
        let mut all_tool_results = Vec::new();

        for _round in 0..=self.config.max_tool_rounds {
            let model_call = self
                .call_model(messages.clone(), &context.tools, usage_snapshot)
                .await?;
            let injected_summary = model_call.injected_summary;
            let response = model_call.response;
            messages.push(Message::assistant_with_tool_calls(
                response.message.clone(),
                response.tool_calls.clone(),
            ));

            if response.tool_calls.is_empty() {
                return Ok(LoopOutput {
                    final_message: response.message,
                    messages,
                    tool_results: all_tool_results,
                    injected_summary,
                    usage_snapshot: *usage_snapshot,
                });
            }

            let tool_results = self
                .execute_tool_calls(&response.tool_calls, &context.tools, usage_snapshot)
                .await?;
            for result in &tool_results {
                messages.push(Message::tool(
                    result.tool_call_id.clone(),
                    result.content.clone(),
                ));
            }
            all_tool_results.extend(tool_results);
        }

        Err(LoopError::TooManyToolRounds(self.config.max_tool_rounds))
    }

    pub(crate) async fn run_with_state(
        &self,
        messages: Vec<Message>,
        state: RunState,
        options: RunStateCallOptions,
    ) -> Result<LoopOutput, LoopError> {
        let mut usage_snapshot = self.start_run()?;
        let output = self
            .run_with_state_inner(messages, state, options, &mut usage_snapshot)
            .await;
        self.finish_run(output.is_ok());
        output
    }

    async fn run_with_state_inner(
        &self,
        messages: Vec<Message>,
        state: RunState,
        options: RunStateCallOptions,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<LoopOutput, LoopError> {
        let context = self.run_context(&CapabilitySelection::default())?;
        let mut messages = apply_capability_prompts(messages, &context.prompts);
        let mut state = state;
        let mut all_tool_results = Vec::new();

        for _round in 0..=self.config.max_tool_rounds {
            let model_call = self
                .call_model_with_state(
                    messages.clone(),
                    &mut state,
                    &options,
                    &context.tools,
                    usage_snapshot,
                )
                .await?;
            let injected_summary = model_call.injected_summary;
            let response = model_call.response;
            messages.push(Message::assistant_with_tool_calls(
                response.message.clone(),
                response.tool_calls.clone(),
            ));

            if response.tool_calls.is_empty() {
                return Ok(LoopOutput {
                    final_message: response.message,
                    messages,
                    tool_results: all_tool_results,
                    injected_summary,
                    usage_snapshot: *usage_snapshot,
                });
            }

            let tool_results = self
                .execute_tool_calls(&response.tool_calls, &context.tools, usage_snapshot)
                .await?;
            for result in &tool_results {
                messages.push(Message::tool(
                    result.tool_call_id.clone(),
                    result.content.clone(),
                ));
            }
            all_tool_results.extend(tool_results);
        }

        Err(LoopError::TooManyToolRounds(self.config.max_tool_rounds))
    }

    pub async fn run_stream<F, Fut>(
        &self,
        messages: Vec<Message>,
        mut on_event: F,
    ) -> Result<LoopOutput, LoopError>
    where
        F: FnMut(LoopEvent) -> Fut + Send,
        Fut: std::future::Future<Output = ()> + Send,
    {
        let mut usage_snapshot = self.start_run()?;
        let output = self
            .run_stream_inner(messages, &mut on_event, &mut usage_snapshot)
            .await;
        self.finish_run(output.is_ok());
        output
    }

    async fn run_stream_inner<F, Fut>(
        &self,
        messages: Vec<Message>,
        on_event: &mut F,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<LoopOutput, LoopError>
    where
        F: FnMut(LoopEvent) -> Fut + Send,
        Fut: std::future::Future<Output = ()> + Send,
    {
        let context = self.run_context(&CapabilitySelection::default())?;
        let mut messages = apply_capability_prompts(messages, &context.prompts);
        let mut all_tool_results = Vec::new();

        for _round in 0..=self.config.max_tool_rounds {
            let prepared = self.request(messages.clone(), &context.tools)?;
            let injected_summary = prepared.injected_summary;
            let model = prepared.request.model.clone();
            self.ensure_model_call_allowed()?;
            self.record_telemetry(TelemetryEvent::ModelCallStarted {
                model: model.clone(),
            });
            let mut stream = self
                .provider
                .stream(prepared.request)
                .await
                .map_err(LoopError::Provider)?;
            let mut aggregation = StreamAggregation::default();

            while let Some(event) = stream.next().await {
                let event = event.map_err(LoopError::Provider)?;
                aggregation.apply(&event)?;

                match event {
                    ModelStreamEvent::MessageDelta(delta) => {
                        on_event(LoopEvent::MessageDelta(delta)).await;
                    }
                    ModelStreamEvent::ToolCallDelta {
                        index,
                        id,
                        name,
                        arguments_delta,
                    } => {
                        on_event(LoopEvent::ToolCallDelta {
                            index,
                            id,
                            name,
                            arguments_delta,
                        })
                        .await;
                    }
                    ModelStreamEvent::Usage(usage) => {
                        on_event(LoopEvent::Usage {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                        })
                        .await;
                    }
                    ModelStreamEvent::Done => {}
                }
            }

            let response = aggregation.finish()?;
            self.record_telemetry(TelemetryEvent::ModelCallFinished {
                model,
                usage: response.usage.clone(),
            });
            let delta = self.record_model_usage(&response);
            usage_snapshot.add_assign(delta);
            messages.push(Message::assistant_with_tool_calls(
                response.message.clone(),
                response.tool_calls.clone(),
            ));

            if response.tool_calls.is_empty() {
                on_event(LoopEvent::Done).await;
                return Ok(LoopOutput {
                    final_message: response.message,
                    messages,
                    tool_results: all_tool_results,
                    injected_summary,
                    usage_snapshot: *usage_snapshot,
                });
            }

            for call in &response.tool_calls {
                on_event(LoopEvent::ToolStarted {
                    tool_call_id: call.id.clone(),
                    name: call.name.clone(),
                })
                .await;

                let result = self
                    .execute_tool_call(call, &context.tools, usage_snapshot)
                    .await?;

                on_event(LoopEvent::ToolFinished {
                    tool_call_id: result.tool_call_id.clone(),
                    name: result.name.clone(),
                    content: result.content.clone(),
                })
                .await;
                messages.push(Message::tool(
                    result.tool_call_id.clone(),
                    result.content.clone(),
                ));
                all_tool_results.push(result);
            }
        }

        Err(LoopError::TooManyToolRounds(self.config.max_tool_rounds))
    }

    async fn call_model(
        &self,
        messages: Vec<Message>,
        tools: &ToolRegistry,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<ModelCall, LoopError> {
        let prepared = self.request(messages, tools)?;
        self.ensure_model_call_allowed()?;
        let model = prepared.request.model.clone();
        self.record_telemetry(TelemetryEvent::ModelCallStarted {
            model: model.clone(),
        });
        let response = self
            .provider
            .complete(prepared.request)
            .await
            .map_err(LoopError::Provider)?;
        self.record_telemetry(TelemetryEvent::ModelCallFinished {
            model,
            usage: response.usage.clone(),
        });
        let delta = self.record_model_usage(&response);
        usage_snapshot.add_assign(delta);

        Ok(ModelCall {
            response,
            injected_summary: prepared.injected_summary,
        })
    }

    async fn call_model_with_state(
        &self,
        messages: Vec<Message>,
        state: &mut RunState,
        options: &RunStateCallOptions,
        tools: &ToolRegistry,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<ModelCall, LoopError> {
        let prepared = self
            .request_with_state(messages, state, options, tools)
            .await?;
        self.ensure_model_call_allowed()?;
        let model = prepared.request.model.clone();
        self.record_telemetry(TelemetryEvent::ModelCallStarted {
            model: model.clone(),
        });
        let response = self
            .provider
            .complete(prepared.request)
            .await
            .map_err(LoopError::Provider)?;
        self.record_telemetry(TelemetryEvent::ModelCallFinished {
            model,
            usage: response.usage.clone(),
        });
        let delta = self.record_model_usage(&response);
        usage_snapshot.add_assign(delta);

        Ok(ModelCall {
            response,
            injected_summary: prepared.injected_summary,
        })
    }

    fn request(
        &self,
        messages: Vec<Message>,
        tools: &ToolRegistry,
    ) -> Result<PreparedRequest, LoopError> {
        let prepared_messages = self.prepare_messages(messages);
        let mut request = ModelRequest {
            model: self.config.model.clone(),
            messages: prepared_messages.messages,
            tools: tools.definitions(),
            max_tokens: self.config.max_tokens,
            cache_hints: Vec::new(),
        };
        self.apply_cache_hints(&mut request)?;

        Ok(PreparedRequest {
            request,
            injected_summary: prepared_messages.injected_summary,
        })
    }

    async fn request_with_state(
        &self,
        messages: Vec<Message>,
        state: &mut RunState,
        options: &RunStateCallOptions,
        tools: &ToolRegistry,
    ) -> Result<PreparedRequest, LoopError> {
        let mut call = BeforeModelCall {
            state: state.clone(),
            config: self.call_config_for_state(options)?,
        };

        if let Some(hook) = &options.before_model_call {
            call = hook(call).await?;
        }

        *state = call.state;
        let prepared_messages =
            self.prepare_messages_with_state(messages, state, options.instruction_policy.as_ref());
        let mut request = ModelRequest {
            model: call.config.model,
            messages: prepared_messages.messages,
            tools: tools.definitions(),
            max_tokens: call.config.max_tokens,
            cache_hints: Vec::new(),
        };
        self.apply_cache_hints(&mut request)?;

        Ok(PreparedRequest {
            request,
            injected_summary: prepared_messages.injected_summary,
        })
    }

    fn call_config_for_state(
        &self,
        options: &RunStateCallOptions,
    ) -> Result<ModelCallConfig, LoopError> {
        let mut model = self.config.model.clone();
        if let Some(mode) = &options.model_mode {
            let Some(mode_model) = options.model_modes.get(mode) else {
                return Err(LoopError::Provider(ProviderError::Config(format!(
                    "model mode `{mode}` is not configured"
                ))));
            };
            model = mode_model.clone();
        }

        Ok(ModelCallConfig {
            model,
            max_tokens: self.config.max_tokens,
        })
    }

    fn run_context(&self, selection: &CapabilitySelection) -> Result<RunContext, LoopError> {
        let resolved = self
            .config
            .capability_bundles
            .resolve(selection, &self.tools)
            .map_err(LoopError::Capability)?;

        Ok(RunContext {
            prompts: resolved.prompts,
            tools: resolved.tools,
        })
    }

    fn apply_cache_hints(&self, request: &mut ModelRequest) -> Result<(), LoopError> {
        if !self.config.cache_policy.uses_provider_prompt_cache() {
            return Ok(());
        }

        self.provider
            .ensure_supports(&request.model, ProviderFeature::PromptCache)
            .map_err(LoopError::Provider)?;

        if let Some(message_index) = request
            .messages
            .iter()
            .position(|message| message.role == Role::System)
        {
            request.cache_hints.push(
                CacheHint::new(CacheHintScope::SystemPrompt { message_index })
                    .for_feature(ProviderFeature::PromptCache),
            );
        }

        if !request.tools.is_empty() {
            request.cache_hints.push(
                CacheHint::new(CacheHintScope::ToolDefinitions)
                    .for_feature(ProviderFeature::PromptCache),
            );
        }

        Ok(())
    }

    fn prepare_messages(&self, messages: Vec<Message>) -> PreparedMessages {
        if let Some(policy) = &self.config.summary_policy {
            return policy.prepare(&messages);
        }

        PreparedMessages {
            messages,
            injected_summary: None,
        }
    }

    fn prepare_messages_with_state(
        &self,
        messages: Vec<Message>,
        state: &RunState,
        policy: Option<&StateInstructionPolicy>,
    ) -> PreparedMessages {
        let mut prepared = if let Some(policy) = &self.config.summary_policy {
            policy.prepare(&messages)
        } else {
            PreparedMessages {
                messages,
                injected_summary: None,
            }
        };

        let Some(state_policy) = policy else {
            return prepared;
        };
        let Some(state_instructions) = state.rendered_instructions(state_policy) else {
            return prepared;
        };

        match prepared.messages.first_mut() {
            Some(message) if message.role == crate::types::Role::System => {
                message.content = format!("{}\n\n{state_instructions}", message.content);
            }
            _ => prepared
                .messages
                .insert(0, Message::system(state_instructions)),
        }

        prepared
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        tools: &ToolRegistry,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<Vec<ToolResult>, LoopError> {
        let mut results = Vec::with_capacity(tool_calls.len());

        for call in tool_calls {
            results.push(self.execute_tool_call(call, tools, usage_snapshot).await?);
        }

        Ok(results)
    }

    async fn execute_tool_call(
        &self,
        call: &ToolCall,
        tools: &ToolRegistry,
        usage_snapshot: &mut UsageSnapshot,
    ) -> Result<ToolResult, LoopError> {
        self.ensure_tool_call_allowed()?;
        self.record_telemetry(TelemetryEvent::ToolCallStarted {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
        });

        match tools.execute(&call.name, call.arguments.clone()).await {
            Ok(content) => {
                let delta = self.config.usage_meter.record_tool_call();
                usage_snapshot.add_assign(delta);
                self.record_telemetry(TelemetryEvent::ToolCallFinished {
                    tool_call_id: call.id.clone(),
                    name: call.name.clone(),
                    is_error: false,
                    recoverable: false,
                });
                Ok(ToolResult::ok(call.id.clone(), call.name.clone(), content))
            }
            Err(ToolError::ResultContent(content)) => {
                let delta = self.config.usage_meter.record_tool_call();
                usage_snapshot.add_assign(delta);
                self.record_telemetry(TelemetryEvent::ToolCallFinished {
                    tool_call_id: call.id.clone(),
                    name: call.name.clone(),
                    is_error: true,
                    recoverable: true,
                });
                Ok(ToolResult::error(
                    call.id.clone(),
                    call.name.clone(),
                    content,
                ))
            }
            Err(error) => {
                let delta = self.config.usage_meter.record_tool_call();
                usage_snapshot.add_assign(delta);
                self.record_telemetry(TelemetryEvent::ToolCallFinished {
                    tool_call_id: call.id.clone(),
                    name: call.name.clone(),
                    is_error: true,
                    recoverable: false,
                });
                Err(LoopError::Tool(error))
            }
        }
    }

    fn start_run(&self) -> Result<UsageSnapshot, LoopError> {
        let delta = self
            .config
            .usage_meter
            .reserve_run(&self.config.usage_limits)?;
        self.record_telemetry(TelemetryEvent::RunStarted);
        Ok(delta)
    }

    fn finish_run(&self, success: bool) {
        self.record_telemetry(TelemetryEvent::RunFinished { success });
    }

    fn ensure_model_call_allowed(&self) -> Result<(), LoopError> {
        self.config
            .usage_meter
            .ensure_model_call_allowed(&self.config.usage_limits)
            .map_err(LoopError::from)
    }

    fn ensure_tool_call_allowed(&self) -> Result<(), LoopError> {
        self.config
            .usage_meter
            .ensure_tool_call_allowed(&self.config.usage_limits)
            .map_err(LoopError::from)
    }

    fn record_model_usage(&self, response: &ModelResponse) -> UsageSnapshot {
        self.config
            .usage_meter
            .record_model_call(response.usage.as_ref())
    }

    fn record_telemetry(&self, event: TelemetryEvent) {
        self.config.telemetry.record(event);
    }
}

struct RunContext {
    prompts: Vec<String>,
    tools: ToolRegistry,
}

struct PreparedRequest {
    request: ModelRequest,
    injected_summary: Option<ConversationSummary>,
}

struct ModelCall {
    response: ModelResponse,
    injected_summary: Option<ConversationSummary>,
}

#[derive(Debug, Clone)]
pub struct LoopOutput {
    pub final_message: String,
    pub messages: Vec<Message>,
    pub tool_results: Vec<ToolResult>,
    pub injected_summary: Option<ConversationSummary>,
    usage_snapshot: UsageSnapshot,
}

impl LoopOutput {
    pub fn usage_snapshot(&self) -> UsageSnapshot {
        self.usage_snapshot
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopEvent {
    MessageDelta(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    ToolStarted {
        tool_call_id: String,
        name: String,
    },
    ToolFinished {
        tool_call_id: String,
        name: String,
        content: String,
    },
    Usage {
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    Done,
}

#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Capability(#[from] CapabilityError),
    #[error("model requested more than {0} tool rounds")]
    TooManyToolRounds(usize),
    #[error("streamed tool call `{0}` did not include valid JSON arguments: {1}")]
    InvalidToolArguments(String, serde_json::Error),
    #[error("usage limit exceeded for {kind:?}: current {current}, limit {limit}")]
    UsageLimitExceeded {
        kind: UsageLimitKind,
        limit: u64,
        current: u64,
    },
}

impl From<UsageLimitError> for LoopError {
    fn from(error: UsageLimitError) -> Self {
        Self::UsageLimitExceeded {
            kind: error.kind,
            limit: error.limit,
            current: error.current,
        }
    }
}

fn apply_capability_prompts(mut messages: Vec<Message>, prompts: &[String]) -> Vec<Message> {
    if prompts.is_empty() {
        return messages;
    }

    let prompt = prompts.join("\n\n");
    match messages.first_mut() {
        Some(message) if message.role == Role::System => {
            message.content = format!("{}\n\n{prompt}", message.content);
        }
        _ => messages.insert(0, Message::system(prompt)),
    }

    messages
}

#[derive(Default)]
struct StreamAggregation {
    message: String,
    tool_calls: Vec<PartialToolCall>,
    usage: Option<crate::types::Usage>,
}

impl StreamAggregation {
    fn apply(&mut self, event: &ModelStreamEvent) -> Result<(), LoopError> {
        match event {
            ModelStreamEvent::MessageDelta(delta) => self.message.push_str(delta),
            ModelStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                while self.tool_calls.len() <= *index {
                    self.tool_calls.push(PartialToolCall::default());
                }
                let call = &mut self.tool_calls[*index];
                if let Some(id) = id {
                    call.id = Some(id.clone());
                }
                if let Some(name) = name {
                    call.name = Some(name.clone());
                }
                call.arguments.push_str(arguments_delta);
            }
            ModelStreamEvent::Usage(usage) => {
                self.usage = Some(usage.clone());
            }
            ModelStreamEvent::Done => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<ModelResponse, LoopError> {
        let mut tool_calls = Vec::new();
        for (index, call) in self.tool_calls.into_iter().enumerate() {
            let id = call.id.unwrap_or_else(|| format!("tool_call_{index}"));
            let name = call.name.unwrap_or_default();
            let arguments = if call.arguments.trim().is_empty() {
                serde_json::Value::Object(Default::default())
            } else {
                serde_json::from_str(&call.arguments)
                    .map_err(|error| LoopError::InvalidToolArguments(id.clone(), error))?
            };
            tool_calls.push(ToolCall {
                id,
                name,
                arguments,
            });
        }

        Ok(ModelResponse {
            message: self.message,
            tool_calls,
            usage: self.usage,
        })
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use futures_util::stream;
    use serde_json::json;

    use super::*;
    use crate::{
        provider::{ModelStream, ProviderResult},
        summarization::{ConversationSummary, SummaryPolicy},
        tool::{FnTool, ToolError},
        types::{Role, ToolDefinition},
    };

    #[tokio::test]
    async fn run_executes_tool_calls_until_the_model_returns_a_final_message() {
        let provider = FakeProvider::with_responses(vec![
            ModelResponse {
                message: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".to_owned(),
                    name: "add".to_owned(),
                    arguments: json!({"a": 2, "b": 3}),
                }],
                usage: None,
            },
            ModelResponse {
                message: "The answer is 5.".to_owned(),
                tool_calls: Vec::new(),
                usage: None,
            },
        ]);
        let loop_runner = AgentLoop::new(provider, math_tools(), AgentLoopConfig::new("fake"));

        let output = loop_runner
            .run(vec![Message::user("what is 2 + 3?")])
            .await
            .unwrap();

        assert_eq!(output.final_message, "The answer is 5.");
        assert_eq!(output.tool_results[0].content, "5");
        assert_eq!(output.messages.len(), 4);
        assert_eq!(output.messages[2].role, Role::Tool);
        assert_eq!(output.messages[2].content, "5");
    }

    #[tokio::test]
    async fn run_stream_aggregates_streamed_tool_calls_and_continues_the_loop() {
        let provider = FakeProvider::with_streams(vec![
            vec![
                ModelStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call_1".to_owned()),
                    name: Some("add".to_owned()),
                    arguments_delta: "{\"a\":2".to_owned(),
                },
                ModelStreamEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: ",\"b\":3}".to_owned(),
                },
                ModelStreamEvent::Done,
            ],
            vec![
                ModelStreamEvent::MessageDelta("The answer ".to_owned()),
                ModelStreamEvent::MessageDelta("is 5.".to_owned()),
                ModelStreamEvent::Done,
            ],
        ]);
        let loop_runner = AgentLoop::new(provider, math_tools(), AgentLoopConfig::new("fake"));
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_sink = Arc::clone(&events);

        let output = loop_runner
            .run_stream(vec![Message::user("what is 2 + 3?")], move |event| {
                let event_sink = Arc::clone(&event_sink);
                async move {
                    event_sink.lock().unwrap().push(event);
                }
            })
            .await
            .unwrap();

        assert_eq!(output.final_message, "The answer is 5.");
        assert_eq!(output.tool_results[0].content, "5");
        assert!(
            events
                .lock()
                .unwrap()
                .contains(&LoopEvent::MessageDelta("The answer ".to_owned()))
        );
        assert!(events.lock().unwrap().contains(&LoopEvent::Done));
    }

    #[tokio::test]
    async fn run_stream_emits_tool_started_before_executing_the_tool() {
        let provider = FakeProvider::with_streams(vec![
            vec![
                ModelStreamEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call_1".to_owned()),
                    name: Some("record".to_owned()),
                    arguments_delta: "{}".to_owned(),
                },
                ModelStreamEvent::Done,
            ],
            vec![
                ModelStreamEvent::MessageDelta("done".to_owned()),
                ModelStreamEvent::Done,
            ],
        ]);
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut tools = ToolRegistry::new();
        let tool_order = Arc::clone(&order);
        tools.register(FnTool::new(
            ToolDefinition::new("record", "Record execution order.", json!({})),
            move |_arguments| {
                let tool_order = Arc::clone(&tool_order);
                Box::pin(async move {
                    tool_order.lock().unwrap().push("tool-executed");
                    Ok("recorded".to_owned())
                })
            },
        ));
        let loop_runner = AgentLoop::new(provider, tools, AgentLoopConfig::new("fake"));
        let event_order = Arc::clone(&order);

        let output = loop_runner
            .run_stream(vec![Message::user("record")], move |event| {
                let event_order = Arc::clone(&event_order);
                async move {
                    match event {
                        LoopEvent::ToolStarted { .. } => {
                            event_order.lock().unwrap().push("tool-started");
                        }
                        LoopEvent::ToolFinished { .. } => {
                            event_order.lock().unwrap().push("tool-finished");
                        }
                        _ => {}
                    }
                }
            })
            .await
            .unwrap();

        assert_eq!(output.final_message, "done");
        assert_eq!(
            &*order.lock().unwrap(),
            &["tool-started", "tool-executed", "tool-finished"]
        );
    }

    #[tokio::test]
    async fn run_injects_summary_context_without_rewriting_the_transcript() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = FakeProvider::with_responses_and_recorder(
            vec![ModelResponse {
                message: "Still on Rust.".to_owned(),
                tool_calls: Vec::new(),
                usage: None,
            }],
            Arc::clone(&requests),
        );
        let config = AgentLoopConfig::new("fake").with_summary_policy(SummaryPolicy::always(
            ConversationSummary::new("The user chose Rust and dislikes hidden state."),
        ));
        let loop_runner = AgentLoop::new(provider, ToolRegistry::new(), config);
        let original_messages = vec![Message::user("What did we decide?")];

        let output = loop_runner.run(original_messages.clone()).await.unwrap();

        let requests = requests.lock().unwrap();
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[0].messages[0].role, Role::System);
        assert!(
            requests[0].messages[0]
                .content
                .contains("The user chose Rust")
        );
        assert_eq!(requests[0].messages[1..], original_messages);
        assert_eq!(output.messages.len(), 2);
        assert_eq!(output.messages[0], Message::user("What did we decide?"));
        assert_eq!(output.messages[1], Message::assistant("Still on Rust."));
        assert_eq!(
            output.injected_summary,
            Some(ConversationSummary::new(
                "The user chose Rust and dislikes hidden state."
            ))
        );
    }

    #[tokio::test]
    async fn run_returns_recoverable_tool_errors_as_agent_visible_tool_results() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = FakeProvider::with_responses_and_recorder(
            vec![
                ModelResponse {
                    message: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_owned(),
                        name: "read_cache".to_owned(),
                        arguments: json!({}),
                    }],
                    usage: None,
                },
                ModelResponse {
                    message: "I can recover from that tool error.".to_owned(),
                    tool_calls: Vec::new(),
                    usage: None,
                },
            ],
            Arc::clone(&requests),
        );
        let mut tools = ToolRegistry::new();
        tools.register(FnTool::new(
            ToolDefinition::new("read_cache", "Read cached content.", json!({})),
            |_arguments| {
                Box::pin(async {
                    Err(ToolError::result_content(
                        r#"{"error":"cache entry was not found"}"#,
                    ))
                })
            },
        ));
        let loop_runner = AgentLoop::new(provider, tools, AgentLoopConfig::new("fake"));

        let output = loop_runner
            .run(vec![Message::user("try it")])
            .await
            .unwrap();

        assert_eq!(output.final_message, "I can recover from that tool error.");
        assert!(output.tool_results[0].is_error);
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].messages[2].role, Role::Tool);
        assert_eq!(
            requests[1].messages[2].tool_call_id,
            Some("call_1".to_owned())
        );
        assert_eq!(
            requests[1].messages[2].content,
            r#"{"error":"cache entry was not found"}"#
        );
    }

    #[tokio::test]
    async fn run_fails_hard_when_the_model_requests_an_unknown_tool() {
        let provider = FakeProvider::with_responses(vec![ModelResponse {
            message: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".to_owned(),
                name: "missing_tool".to_owned(),
                arguments: json!({}),
            }],
            usage: None,
        }]);
        let loop_runner =
            AgentLoop::new(provider, ToolRegistry::new(), AgentLoopConfig::new("fake"));

        let error = loop_runner
            .run(vec![Message::user("try it")])
            .await
            .unwrap_err();

        assert!(
            matches!(error, LoopError::Tool(ToolError::NotFound(name)) if name == "missing_tool")
        );
    }

    fn math_tools() -> ToolRegistry {
        let mut tools = ToolRegistry::new();
        tools.register(FnTool::new(
            ToolDefinition::new(
                "add",
                "Add two integers.",
                json!({
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }),
            ),
            |arguments| {
                Box::pin(async move {
                    let a = arguments["a"].as_i64().unwrap();
                    let b = arguments["b"].as_i64().unwrap();
                    Ok((a + b).to_string())
                })
            },
        ));
        tools
    }

    #[tokio::test]
    async fn run_returns_recoverable_tool_errors_as_tool_result_messages() {
        let provider = FakeProvider::with_responses(vec![
            ModelResponse {
                message: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".to_owned(),
                    name: "missing_file".to_owned(),
                    arguments: json!({}),
                }],
                usage: None,
            },
            ModelResponse {
                message: "I could not read it.".to_owned(),
                tool_calls: Vec::new(),
                usage: None,
            },
        ]);
        let mut tools = ToolRegistry::new();
        tools.register(FnTool::new(
            ToolDefinition::new("missing_file", "Fails recoverably.", json!({})),
            |_arguments| {
                Box::pin(async {
                    Err(ToolError::result_content(
                        r#"{"error":"file was not found"}"#,
                    ))
                })
            },
        ));
        let loop_runner = AgentLoop::new(provider, tools, AgentLoopConfig::new("fake"));

        let output = loop_runner
            .run(vec![Message::user("read it")])
            .await
            .unwrap();

        assert_eq!(output.final_message, "I could not read it.");
        assert_eq!(
            output.tool_results[0].content,
            r#"{"error":"file was not found"}"#
        );
        assert_eq!(
            output.messages[2].content,
            r#"{"error":"file was not found"}"#
        );
    }

    struct FakeProvider {
        responses: Mutex<VecDeque<ModelResponse>>,
        streams: Mutex<VecDeque<Vec<ModelStreamEvent>>>,
        requests: Arc<Mutex<Vec<ModelRequest>>>,
    }

    impl FakeProvider {
        fn with_responses(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                streams: Mutex::new(VecDeque::new()),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_streams(streams: Vec<Vec<ModelStreamEvent>>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::new()),
                streams: Mutex::new(streams.into()),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_responses_and_recorder(
            responses: Vec<ModelResponse>,
            requests: Arc<Mutex<Vec<ModelRequest>>>,
        ) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                streams: Mutex::new(VecDeque::new()),
                requests,
            }
        }
    }

    #[async_trait]
    impl ModelProvider for FakeProvider {
        async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
            self.requests.lock().unwrap().push(request);
            Ok(self.responses.lock().unwrap().pop_front().unwrap())
        }

        async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
            self.requests.lock().unwrap().push(request);
            let events = self.streams.lock().unwrap().pop_front().unwrap();
            Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
        }
    }
}
