use std::{collections::VecDeque, sync::Mutex};

use async_trait::async_trait;
use futures_util::stream;
use orchestrai::{
    AgentConfig, FnTool, InMemoryRunStore, LoopError, ModelProvider, ModelRequest, ModelResponse,
    ModelStream, ModelStreamEvent, ProviderResult, RunEvent, RunStatus, ToolCall, ToolDefinition,
    ToolError, Usage, create_agent,
};
use serde_json::json;

#[tokio::test]
async fn successful_run_records_sanitized_lifecycle_model_and_tool_events() {
    let run_store = InMemoryRunStore::new();
    let provider = RecordingProvider::new(vec![
        ModelResponse {
            message: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".to_owned(),
                name: "lookup".to_owned(),
                arguments: json!({"secret": "must-not-be-stored"}),
            }],
            usage: Some(usage(7, 3)),
        },
        ModelResponse {
            message: "done".to_owned(),
            tool_calls: Vec::new(),
            usage: Some(usage(5, 2)),
        },
    ]);
    let agent = create_agent(
        AgentConfig::new(provider, "fake-model")
            .with_run_store(run_store.clone())
            .with_max_tool_rounds(1)
            .with_tool(FnTool::new(
                ToolDefinition::new("lookup", "Lookup data.", json!({"type": "object"})),
                |_arguments| Box::pin(async { Ok("secret tool result".to_owned()) }),
            )),
    );

    let output = agent.run("hello secret prompt").await.unwrap();

    assert_eq!(output.final_message, "done");
    assert!(!output.run_id.as_str().is_empty());
    let events = run_store.events();
    assert_eq!(
        events,
        vec![
            RunEvent::RunStarted {
                run_id: output.run_id.clone()
            },
            RunEvent::ModelCallStarted {
                run_id: output.run_id.clone(),
                model: "fake-model".to_owned()
            },
            RunEvent::ModelCallFinished {
                run_id: output.run_id.clone(),
                model: "fake-model".to_owned(),
                input_tokens: Some(7),
                output_tokens: Some(3)
            },
            RunEvent::ToolCallStarted {
                run_id: output.run_id.clone(),
                tool_call_id: "call_1".to_owned(),
                name: "lookup".to_owned()
            },
            RunEvent::ToolCallFinished {
                run_id: output.run_id.clone(),
                tool_call_id: "call_1".to_owned(),
                name: "lookup".to_owned(),
                is_error: false,
                recoverable: false
            },
            RunEvent::ModelCallStarted {
                run_id: output.run_id.clone(),
                model: "fake-model".to_owned()
            },
            RunEvent::ModelCallFinished {
                run_id: output.run_id.clone(),
                model: "fake-model".to_owned(),
                input_tokens: Some(5),
                output_tokens: Some(2)
            },
            RunEvent::RunFinished {
                run_id: output.run_id.clone(),
                status: RunStatus::Succeeded,
                usage: output.usage_snapshot()
            },
        ]
    );

    let serialized = serde_json::to_string(&events).unwrap();
    assert!(!serialized.contains("hello secret prompt"));
    assert!(!serialized.contains("must-not-be-stored"));
    assert!(!serialized.contains("secret tool result"));
}

#[tokio::test]
async fn failed_tool_run_records_failed_status_without_tool_content() {
    let run_store = InMemoryRunStore::new();
    let provider = RecordingProvider::new(vec![ModelResponse {
        message: String::new(),
        tool_calls: vec![ToolCall {
            id: "call_1".to_owned(),
            name: "explode".to_owned(),
            arguments: json!({}),
        }],
        usage: Some(usage(4, 1)),
    }]);
    let agent = create_agent(
        AgentConfig::new(provider, "fake-model")
            .with_run_store(run_store.clone())
            .with_tool(FnTool::new(
                ToolDefinition::new("explode", "Fail.", json!({"type": "object"})),
                |_arguments| {
                    Box::pin(async {
                        Err(ToolError::Execution(
                            "private failure details should not be in store".to_owned(),
                        ))
                    })
                },
            )),
    );

    let error = agent.run("trigger failure").await.unwrap_err();
    assert!(matches!(error, LoopError::Tool(_)));

    let events = run_store.events();
    assert!(
        matches!(
            events.last(),
            Some(RunEvent::RunFinished {
                status: RunStatus::Failed,
                ..
            })
        ),
        "expected failed run finish event, got {events:?}"
    );
    let serialized = serde_json::to_string(&events).unwrap();
    assert!(!serialized.contains("private failure details"));
}

struct RecordingProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
}

impl RecordingProvider {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }
}

#[async_trait]
impl ModelProvider for RecordingProvider {
    async fn complete(&self, _request: ModelRequest) -> ProviderResult<ModelResponse> {
        Ok(self.responses.lock().unwrap().pop_front().unwrap())
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        Ok(Box::pin(stream::iter(vec![Ok(ModelStreamEvent::Done)])))
    }
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
    }
}
