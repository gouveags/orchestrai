use futures_util::StreamExt;

use crate::{
    provider::{ModelProvider, ModelRequest, ModelResponse, ModelStreamEvent, ProviderError},
    tool::{ToolError, ToolRegistry},
    types::{Message, ToolCall, ToolResult},
};

#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    pub model: String,
    pub max_tool_rounds: usize,
    pub max_tokens: Option<u32>,
}

impl AgentLoopConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_tool_rounds: 8,
            max_tokens: None,
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
        let mut messages = messages;
        let mut all_tool_results = Vec::new();

        for _round in 0..=self.config.max_tool_rounds {
            let response = self.call_model(messages.clone()).await?;
            messages.push(Message::assistant_with_tool_calls(
                response.message.clone(),
                response.tool_calls.clone(),
            ));

            if response.tool_calls.is_empty() {
                return Ok(LoopOutput {
                    final_message: response.message,
                    messages,
                    tool_results: all_tool_results,
                });
            }

            let tool_results = self.execute_tool_calls(&response.tool_calls).await?;
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
        let mut messages = messages;
        let mut all_tool_results = Vec::new();

        for _round in 0..=self.config.max_tool_rounds {
            let mut stream = self
                .provider
                .stream(self.request(messages.clone()))
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
                });
            }

            let tool_results = self.execute_tool_calls(&response.tool_calls).await?;
            for result in &tool_results {
                on_event(LoopEvent::ToolStarted {
                    tool_call_id: result.tool_call_id.clone(),
                    name: result.name.clone(),
                })
                .await;
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
            }
            all_tool_results.extend(tool_results);
        }

        Err(LoopError::TooManyToolRounds(self.config.max_tool_rounds))
    }

    async fn call_model(&self, messages: Vec<Message>) -> Result<ModelResponse, LoopError> {
        self.provider
            .complete(self.request(messages))
            .await
            .map_err(LoopError::Provider)
    }

    fn request(&self, messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: self.config.model.clone(),
            messages,
            tools: self.tools.definitions(),
            max_tokens: self.config.max_tokens,
        }
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: &[ToolCall],
    ) -> Result<Vec<ToolResult>, LoopError> {
        let mut results = Vec::with_capacity(tool_calls.len());

        for call in tool_calls {
            let content = self
                .tools
                .execute(&call.name, call.arguments.clone())
                .await
                .map_err(LoopError::Tool)?;
            results.push(ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                content,
            });
        }

        Ok(results)
    }
}

#[derive(Debug, Clone)]
pub struct LoopOutput {
    pub final_message: String,
    pub messages: Vec<Message>,
    pub tool_results: Vec<ToolResult>,
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
    #[error("model requested more than {0} tool rounds")]
    TooManyToolRounds(usize),
    #[error("streamed tool call `{0}` did not include valid JSON arguments: {1}")]
    InvalidToolArguments(String, serde_json::Error),
}

#[derive(Default)]
struct StreamAggregation {
    message: String,
    tool_calls: Vec<PartialToolCall>,
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
            ModelStreamEvent::Usage(_) | ModelStreamEvent::Done => {}
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
            usage: None,
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
        tool::FnTool,
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

    struct FakeProvider {
        responses: Mutex<VecDeque<ModelResponse>>,
        streams: Mutex<VecDeque<Vec<ModelStreamEvent>>>,
    }

    impl FakeProvider {
        fn with_responses(responses: Vec<ModelResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                streams: Mutex::new(VecDeque::new()),
            }
        }

        fn with_streams(streams: Vec<Vec<ModelStreamEvent>>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::new()),
                streams: Mutex::new(streams.into()),
            }
        }
    }

    #[async_trait]
    impl ModelProvider for FakeProvider {
        async fn complete(&self, _request: ModelRequest) -> ProviderResult<ModelResponse> {
            Ok(self.responses.lock().unwrap().pop_front().unwrap())
        }

        async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
            let events = self.streams.lock().unwrap().pop_front().unwrap();
            Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
        }
    }
}
