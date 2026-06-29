use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use futures_util::stream;
use orchestrai::{
    AgentConfig, FnTool, LoopError, Message, ModelProvider, ModelRequest, ModelResponse,
    ModelStream, ModelStreamEvent, ProviderError, ProviderResult, ToolCall, ToolDefinition,
    ToolError, Usage, UsageLimitKind, UsageLimits, UsageMeter, UsageSnapshot, create_agent,
};
use serde_json::{Value, json};

#[tokio::test]
async fn run_exposes_usage_snapshot_and_updates_shared_meter_from_full_responses() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let meter = UsageMeter::default();
    let agent = create_agent(
        AgentConfig::new(
            FakeProvider::responses(
                vec![
                    tool_response("call_1", "lookup", json!({"id": "acct_1"})).with_usage(11, 3),
                    text_response("ready").with_usage(7, 5),
                ],
                Arc::clone(&requests),
            ),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_tool(lookup_tool()),
    );

    let output = agent.run("lookup acct_1").await.unwrap();

    let expected = UsageSnapshot {
        runs: 1,
        model_calls: 2,
        tool_calls: 1,
        input_tokens: 18,
        output_tokens: 8,
    };
    assert_eq!(output.usage_snapshot(), expected);
    assert_eq!(meter.snapshot(), expected);
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn run_stream_uses_the_last_cumulative_usage_snapshot_for_each_model_call() {
    let meter = UsageMeter::default();
    let agent = create_agent(
        AgentConfig::new(
            FakeProvider::streams(vec![
                vec![
                    ModelStreamEvent::ToolCallDelta {
                        index: 0,
                        id: Some("call_1".to_owned()),
                        name: Some("lookup".to_owned()),
                        arguments_delta: r#"{"id":"acct_1"}"#.to_owned(),
                    },
                    ModelStreamEvent::Usage(usage(2, 1)),
                    ModelStreamEvent::Usage(usage(4, 2)),
                    ModelStreamEvent::Done,
                ],
                vec![
                    ModelStreamEvent::MessageDelta("ready".to_owned()),
                    ModelStreamEvent::Usage(usage(5, 4)),
                    ModelStreamEvent::Usage(usage(9, 6)),
                    ModelStreamEvent::Done,
                ],
            ]),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_tool(lookup_tool()),
    );

    let output = agent
        .run_stream("lookup acct_1", |_| async {})
        .await
        .unwrap();

    let expected = UsageSnapshot {
        runs: 1,
        model_calls: 2,
        tool_calls: 1,
        input_tokens: 13,
        output_tokens: 8,
    };
    assert_eq!(output.usage_snapshot(), expected);
    assert_eq!(meter.snapshot(), expected);
}

#[tokio::test]
async fn run_limit_fails_closed_before_the_provider_is_called() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let meter = UsageMeter::default();
    let agent = create_agent(
        AgentConfig::new(
            FakeProvider::responses(vec![text_response("should not run")], Arc::clone(&requests)),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_usage_limits(UsageLimits::default().with_max_runs(0)),
    );

    let error = agent.run("hello").await.unwrap_err();

    assert!(matches!(
        error,
        LoopError::UsageLimitExceeded {
            kind: UsageLimitKind::Runs,
            ..
        }
    ));
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(meter.snapshot(), UsageSnapshot::default());
}

#[tokio::test]
async fn exhausted_token_budget_fails_before_another_expensive_provider_call() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let meter = UsageMeter::from_snapshot(UsageSnapshot {
        runs: 1,
        model_calls: 1,
        tool_calls: 0,
        input_tokens: 8,
        output_tokens: 2,
    });
    let agent = create_agent(
        AgentConfig::new(
            FakeProvider::responses(vec![text_response("should not run")], Arc::clone(&requests)),
            "fake-model",
        )
        .with_usage_meter(meter)
        .with_usage_limits(UsageLimits::default().with_max_total_tokens(10)),
    );

    let error = agent.run("hello again").await.unwrap_err();

    assert!(matches!(
        error,
        LoopError::UsageLimitExceeded {
            kind: UsageLimitKind::TotalTokens,
            ..
        }
    ));
    assert!(requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn token_limit_reached_after_provider_response_stops_tools_and_next_provider_call() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let meter = UsageMeter::default();
    let tool_counter = Arc::clone(&tool_calls);
    let agent = create_agent(
        AgentConfig::new(
            FakeProvider::responses(
                vec![
                    tool_response("call_1", "expensive_lookup", json!({})).with_usage(8, 2),
                    text_response("should not run").with_usage(1, 1),
                ],
                Arc::clone(&requests),
            ),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_usage_limits(UsageLimits::default().with_max_total_tokens(10))
        .with_tool(FnTool::new(
            ToolDefinition::new("expensive_lookup", "Should not execute.", json!({})),
            move |_arguments| {
                let tool_counter = Arc::clone(&tool_counter);
                Box::pin(async move {
                    tool_counter.fetch_add(1, Ordering::SeqCst);
                    Ok("tool result should not be visible".to_owned())
                })
            },
        )),
    );

    let error = agent.run("use the expensive lookup").await.unwrap_err();

    assert!(matches!(
        error,
        LoopError::UsageLimitExceeded {
            kind: UsageLimitKind::TotalTokens,
            ..
        }
    ));
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        meter.snapshot(),
        UsageSnapshot {
            runs: 1,
            model_calls: 1,
            tool_calls: 0,
            input_tokens: 8,
            output_tokens: 2,
        }
    );
}

#[tokio::test]
async fn recoverable_tool_errors_are_counted_and_still_agent_visible() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let meter = UsageMeter::default();
    let agent = create_agent(
        AgentConfig::new(
            FakeProvider::responses(
                vec![
                    tool_response("call_1", "recoverable_lookup", json!({})).with_usage(3, 1),
                    text_response("I can recover from that tool error.").with_usage(5, 2),
                ],
                Arc::clone(&requests),
            ),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_tool(FnTool::new(
            ToolDefinition::new("recoverable_lookup", "Fails recoverably.", json!({})),
            |_arguments| {
                Box::pin(async {
                    Err(ToolError::result_content(
                        r#"{"error":"cache entry was not found"}"#,
                    ))
                })
            },
        )),
    );

    let output = agent.run("try lookup").await.unwrap();

    assert_eq!(output.final_message, "I can recover from that tool error.");
    assert!(output.tool_results[0].is_error);
    assert_eq!(
        output.tool_results[0].content,
        r#"{"error":"cache entry was not found"}"#
    );
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].messages.contains(&Message::tool(
        "call_1",
        r#"{"error":"cache entry was not found"}"#
    )));
    assert_eq!(
        output.usage_snapshot(),
        UsageSnapshot {
            runs: 1,
            model_calls: 2,
            tool_calls: 1,
            input_tokens: 8,
            output_tokens: 3,
        }
    );
    assert_eq!(meter.snapshot(), output.usage_snapshot());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_model_call_limit_is_reserved_before_provider_await() {
    let meter = UsageMeter::default();
    let limits = UsageLimits::default().with_max_model_calls(1);
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let agent_a = create_agent(
        AgentConfig::new(
            SleepingProvider::new("first", Arc::clone(&provider_calls)),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_usage_limits(limits),
    );
    let agent_b = create_agent(
        AgentConfig::new(
            SleepingProvider::new("second", Arc::clone(&provider_calls)),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_usage_limits(limits),
    );

    let first = tokio::spawn(async move { agent_a.run("hello").await });
    let second = tokio::spawn(async move { agent_b.run("hello").await });
    let (first, second) = tokio::join!(first, second);
    let first = first.unwrap();
    let second = second.unwrap();

    assert_eq!(success_count([first.as_ref(), second.as_ref()]), 1);
    assert_eq!(
        usage_limit_error_count(
            [first.as_ref(), second.as_ref()],
            UsageLimitKind::ModelCalls
        ),
        1
    );
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        meter.snapshot(),
        UsageSnapshot {
            runs: 2,
            model_calls: 1,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
        }
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shared_tool_call_limit_is_reserved_before_tool_await() {
    let meter = UsageMeter::default();
    let limits = UsageLimits::default().with_max_tool_calls(1);
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let agent_a = create_agent(
        AgentConfig::new(
            FakeProvider::responses(
                vec![
                    tool_response("call_1", "slow_tool", json!({})),
                    text_response("done"),
                ],
                Arc::new(Mutex::new(Vec::new())),
            ),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_usage_limits(limits)
        .with_tool(slow_tool(Arc::clone(&tool_calls))),
    );
    let agent_b = create_agent(
        AgentConfig::new(
            FakeProvider::responses(
                vec![
                    tool_response("call_2", "slow_tool", json!({})),
                    text_response("done"),
                ],
                Arc::new(Mutex::new(Vec::new())),
            ),
            "fake-model",
        )
        .with_usage_meter(meter.clone())
        .with_usage_limits(limits)
        .with_tool(slow_tool(Arc::clone(&tool_calls))),
    );

    let first = tokio::spawn(async move { agent_a.run("use tool").await });
    let second = tokio::spawn(async move { agent_b.run("use tool").await });
    let (first, second) = tokio::join!(first, second);
    let first = first.unwrap();
    let second = second.unwrap();

    assert_eq!(success_count([first.as_ref(), second.as_ref()]), 1);
    assert_eq!(
        usage_limit_error_count([first.as_ref(), second.as_ref()], UsageLimitKind::ToolCalls),
        1
    );
    assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
    assert_eq!(meter.snapshot().tool_calls, 1);
}

#[tokio::test]
async fn failed_provider_calls_still_count_as_logical_model_call_attempts() {
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let meter = UsageMeter::default();
    let agent = create_agent(
        AgentConfig::new(
            FailingProvider {
                calls: Arc::clone(&provider_calls),
            },
            "fake-model",
        )
        .with_usage_meter(meter.clone()),
    );

    let error = agent.run("hello").await.unwrap_err();

    assert!(matches!(
        error,
        LoopError::Provider(ProviderError::Config(message)) if message == "provider failed"
    ));
    assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        meter.snapshot(),
        UsageSnapshot {
            runs: 1,
            model_calls: 1,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
        }
    );
}

fn lookup_tool() -> impl orchestrai::Tool {
    FnTool::new(
        ToolDefinition::new(
            "lookup",
            "Lookup an account.",
            json!({
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"]
            }),
        ),
        |_arguments| Box::pin(async { Ok(r#"{"tier":"pro"}"#.to_owned()) }),
    )
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

trait WithUsage {
    fn with_usage(self, input_tokens: u64, output_tokens: u64) -> Self;
}

impl WithUsage for ModelResponse {
    fn with_usage(mut self, input_tokens: u64, output_tokens: u64) -> Self {
        self.usage = Some(usage(input_tokens, output_tokens));
        self
    }
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
    }
}

fn slow_tool(tool_calls: Arc<AtomicUsize>) -> impl orchestrai::Tool {
    FnTool::new(
        ToolDefinition::new("slow_tool", "Slow tool.", json!({})),
        move |_arguments| {
            let tool_calls = Arc::clone(&tool_calls);
            Box::pin(async move {
                std::thread::sleep(Duration::from_millis(100));
                tool_calls.fetch_add(1, Ordering::SeqCst);
                Ok("ok".to_owned())
            })
        },
    )
}

fn success_count<'a>(
    results: impl IntoIterator<Item = Result<&'a orchestrai::AgentOutput, &'a LoopError>>,
) -> usize {
    results.into_iter().filter(Result::is_ok).count()
}

fn usage_limit_error_count<'a>(
    results: impl IntoIterator<Item = Result<&'a orchestrai::AgentOutput, &'a LoopError>>,
    kind: UsageLimitKind,
) -> usize {
    results
        .into_iter()
        .filter(|result| {
            matches!(
                result,
                Err(LoopError::UsageLimitExceeded {
                    kind: error_kind,
                    ..
                }) if *error_kind == kind
            )
        })
        .count()
}

struct FakeProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
    streams: Mutex<VecDeque<Vec<ModelStreamEvent>>>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
    complete_calls: AtomicUsize,
    stream_calls: AtomicUsize,
}

struct SleepingProvider {
    message: String,
    calls: Arc<AtomicUsize>,
}

impl SleepingProvider {
    fn new(message: impl Into<String>, calls: Arc<AtomicUsize>) -> Self {
        Self {
            message: message.into(),
            calls,
        }
    }
}

#[async_trait]
impl ModelProvider for SleepingProvider {
    async fn complete(&self, _request: ModelRequest) -> ProviderResult<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(100));
        Ok(text_response(&self.message))
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(100));
        Ok(Box::pin(stream::iter(vec![Ok(ModelStreamEvent::Done)])))
    }
}

struct FailingProvider {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ModelProvider for FailingProvider {
    async fn complete(&self, _request: ModelRequest) -> ProviderResult<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ProviderError::Config("provider failed".to_owned()))
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ProviderError::Config("provider failed".to_owned()))
    }
}

impl FakeProvider {
    fn responses(responses: Vec<ModelResponse>, requests: Arc<Mutex<Vec<ModelRequest>>>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            streams: Mutex::new(VecDeque::new()),
            requests,
            complete_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
        }
    }

    fn streams(streams: Vec<Vec<ModelStreamEvent>>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            streams: Mutex::new(streams.into()),
            requests: Arc::new(Mutex::new(Vec::new())),
            complete_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ModelProvider for FakeProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        self.complete_calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(request);
        Ok(self.responses.lock().unwrap().pop_front().unwrap())
    }

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push(request);
        let events = self.streams.lock().unwrap().pop_front().unwrap();
        Ok(Box::pin(stream::iter(events.into_iter().map(Ok))))
    }
}
