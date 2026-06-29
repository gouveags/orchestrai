use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures_util::stream;
use orchestrai::provider::{ModelStreamEvent, ProviderResult};
use orchestrai::{
    AgentConfig, AgentLoop, AgentLoopConfig, Message, ModelProvider, ModelRequest, ModelResponse,
    ModelStream, TelemetryConfig, TelemetryEvent, TelemetrySink, Tool, ToolCall, ToolDefinition,
    ToolError, ToolRegistry, Usage, create_agent,
};
use serde_json::json;

#[tokio::test]
async fn agent_config_telemetry_captures_lifecycle_model_tool_and_usage_events() {
    let telemetry = RecordingTelemetrySink::default();
    let provider = RecordingProvider::new([
        ModelResponse {
            message: String::new(),
            tool_calls: vec![ToolCall {
                id: "tool_1".to_owned(),
                name: "lookup_account".to_owned(),
                arguments: json!({"account": "acme", "password": "secret-tool-argument"}),
            }],
            usage: Some(Usage {
                input_tokens: Some(11),
                output_tokens: Some(7),
            }),
        },
        ModelResponse {
            message: "recovered".to_owned(),
            tool_calls: Vec::new(),
            usage: Some(Usage {
                input_tokens: Some(13),
                output_tokens: Some(5),
            }),
        },
    ]);
    let agent = create_agent(
        AgentConfig::new(provider, "fake-observed-model")
            .with_max_tool_rounds(1)
            .with_telemetry(TelemetryConfig::new().with_sink(telemetry.clone()))
            .with_tool(RecoverableErrorTool),
    );

    let output = agent
        .run("user message with secret-message-body")
        .await
        .unwrap();

    assert_eq!(output.final_message, "recovered");
    let events = telemetry.events();
    assert_contains_event(&events, |event| matches!(event, TelemetryEvent::RunStarted));
    assert_contains_event(&events, |event| {
        matches!(event, TelemetryEvent::RunFinished { success: true })
    });
    assert_contains_event(&events, |event| {
        matches!(
            event,
            TelemetryEvent::ModelCallStarted { model } if model == "fake-observed-model"
        )
    });
    assert_contains_event(&events, |event| {
        matches!(
            event,
            TelemetryEvent::ModelCallFinished {
                model,
                usage: Some(Usage {
                    input_tokens: Some(11),
                    output_tokens: Some(7),
                }),
            } if model == "fake-observed-model"
        )
    });
    assert_contains_event(&events, |event| {
        matches!(
            event,
            TelemetryEvent::ModelCallFinished {
                model,
                usage: Some(Usage {
                    input_tokens: Some(13),
                    output_tokens: Some(5),
                }),
            } if model == "fake-observed-model"
        )
    });
    assert_contains_event(&events, |event| {
        matches!(
            event,
            TelemetryEvent::ToolCallStarted { tool_call_id, name }
                if tool_call_id == "tool_1" && name == "lookup_account"
        )
    });
    assert_contains_event(&events, |event| {
        matches!(
            event,
            TelemetryEvent::ToolCallFinished {
                tool_call_id,
                name,
                is_error: true,
                recoverable: true,
            } if tool_call_id == "tool_1" && name == "lookup_account"
        )
    });

    let debug_events = format!("{events:?}");
    assert!(!debug_events.contains("secret-message-body"));
    assert!(!debug_events.contains("secret-tool-argument"));
    assert!(!debug_events.contains("recoverable-result-content-secret"));
}

#[tokio::test]
async fn agent_loop_config_telemetry_does_not_require_per_agent_plumbing() {
    let telemetry = RecordingTelemetrySink::default();
    let provider = RecordingProvider::new([text_response("loop done")]);
    let loop_runner = AgentLoop::new(
        provider,
        ToolRegistry::new(),
        AgentLoopConfig::new("fake-loop-model")
            .with_telemetry(TelemetryConfig::new().with_sink(telemetry.clone())),
    );

    let output = loop_runner
        .run(vec![Message::user("base loop lifecycle")])
        .await
        .unwrap();

    assert_eq!(output.final_message, "loop done");
    let events = telemetry.events();
    assert_contains_event(&events, |event| matches!(event, TelemetryEvent::RunStarted));
    assert_contains_event(&events, |event| {
        matches!(
            event,
            TelemetryEvent::ModelCallStarted { model } if model == "fake-loop-model"
        )
    });
    assert_contains_event(&events, |event| {
        matches!(event, TelemetryEvent::RunFinished { success: true })
    });
}

struct RecoverableErrorTool;

#[async_trait]
impl Tool for RecoverableErrorTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "lookup_account",
            "Lookup account details.",
            json!({
                "type": "object",
                "properties": {"account": {"type": "string"}},
                "required": ["account"]
            }),
        )
    }

    async fn execute(&self, _arguments: serde_json::Value) -> Result<String, ToolError> {
        Err(ToolError::result_content(
            "recoverable-result-content-secret",
        ))
    }
}

fn assert_contains_event(events: &[TelemetryEvent], predicate: impl Fn(&TelemetryEvent) -> bool) {
    assert!(
        events.iter().any(predicate),
        "missing expected telemetry event in {events:#?}"
    );
}

fn text_response(message: impl Into<String>) -> ModelResponse {
    ModelResponse {
        message: message.into(),
        tool_calls: Vec::new(),
        usage: None,
    }
}

#[derive(Clone, Default)]
struct RecordingTelemetrySink {
    events: Arc<Mutex<Vec<TelemetryEvent>>>,
}

impl RecordingTelemetrySink {
    fn events(&self) -> Vec<TelemetryEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl TelemetrySink for RecordingTelemetrySink {
    fn record(&self, event: TelemetryEvent) {
        self.events.lock().unwrap().push(event);
    }
}

struct RecordingProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
}

impl RecordingProvider {
    fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
        }
    }
}

#[async_trait]
impl ModelProvider for RecordingProvider {
    async fn complete(&self, _request: ModelRequest) -> ProviderResult<ModelResponse> {
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("fake provider received unexpected request"))
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        Ok(Box::pin(stream::iter(vec![Ok(ModelStreamEvent::Done)])))
    }
}
