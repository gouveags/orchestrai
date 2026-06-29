use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use futures_util::stream;
use orchestrai::provider::{ModelStreamEvent, ProviderResult};
use orchestrai::{
    AgentConfig, FnTool, LoopError, Message, ModelProvider, ModelRequest, ModelResponse,
    ModelStream, RunOptions, RunState, StateInstructionPolicy, ToolCall, ToolDefinition,
    create_agent,
};
use serde_json::json;

#[tokio::test]
async fn run_state_renders_selected_entries_and_applies_builtin_model_mode() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = create_agent(
        AgentConfig::new(
            RecordingProvider::new(vec![text_response("done")], Arc::clone(&requests)),
            "fake-balanced",
        )
        .with_instructions("You are a support copilot.")
        .with_run_state_instructions(StateInstructionPolicy::selected([
            "tenant_name",
            "plan_tier",
        ]))
        .with_model_modes([("fast", "fake-fast"), ("accurate", "fake-accurate")]),
    );

    let output = agent
        .run_with_options(
            RunOptions::new("Draft the customer reply.")
                .with_state(
                    RunState::from_json(json!({
                        "tenant_name": "Acme Racing",
                        "plan_tier": "pro",
                        "private_note": "must never be shown to the model"
                    }))
                    .unwrap(),
                )
                .with_model_mode("fast"),
        )
        .await
        .unwrap();

    assert_eq!(output.final_message, "done");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].model, "fake-fast");
    assert_eq!(requests[0].max_tokens, None);
    assert!(
        requests[0]
            .messages
            .iter()
            .any(|message| message == &Message::user("Draft the customer reply."))
    );
    assert_system_contains(&requests[0], "You are a support copilot.");
    assert_system_contains(&requests[0], "tenant_name: Acme Racing");
    assert_system_contains(&requests[0], "plan_tier: pro");
    assert_system_excludes(&requests[0], "private_note");
}

#[tokio::test]
async fn before_model_call_hook_resolves_state_and_config_before_every_llm_call() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let hook_calls = Arc::new(AtomicUsize::new(0));
    let tool_call = ToolCall {
        id: "call_1".to_owned(),
        name: "lookup_customer".to_owned(),
        arguments: json!({"slug": "acme"}),
    };
    let provider = RecordingProvider::new(
        vec![
            ModelResponse {
                message: String::new(),
                tool_calls: vec![tool_call.clone()],
                usage: None,
            },
            text_response("ready"),
        ],
        Arc::clone(&requests),
    );
    let hook_counter = Arc::clone(&hook_calls);
    let agent = create_agent(
        AgentConfig::new(provider, "fake-accurate")
            .with_instructions("You are an account assistant.")
            .with_max_tool_rounds(1)
            .with_run_state_instructions(StateInstructionPolicy::selected([
                "account_slug",
                "resolved_profile",
                "hook_call_number",
            ]))
            .with_before_model_call(move |mut call| {
                let hook_counter = Arc::clone(&hook_counter);
                async move {
                    let call_number = hook_counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let account_slug = call.state.get_string("account_slug").unwrap();
                    call.state.insert(
                        "resolved_profile",
                        json!(format!("{account_slug}:profile:{call_number}")),
                    );
                    call.state.insert("hook_call_number", json!(call_number));
                    call.config.max_tokens = Some(if call_number == 1 { 32 } else { 64 });
                    Ok(call)
                }
            })
            .with_tool(FnTool::new(
                ToolDefinition::new(
                    "lookup_customer",
                    "Lookup a customer profile.",
                    json!({
                        "type": "object",
                        "properties": {"slug": {"type": "string"}},
                        "required": ["slug"]
                    }),
                ),
                |_arguments| Box::pin(async { Ok(r#"{"tier":"enterprise"}"#.to_owned()) }),
            )),
    );

    let output = agent
        .run_with_state(
            "Prepare the next action.",
            RunState::from_json(json!({"account_slug": "acme"})).unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.final_message, "ready");
    assert_eq!(hook_calls.load(Ordering::SeqCst), 2);

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].model, "fake-accurate");
    assert_eq!(requests[0].max_tokens, Some(32));
    assert_system_contains(&requests[0], "You are an account assistant.");
    assert_system_contains(&requests[0], "account_slug: acme");
    assert_system_contains(&requests[0], "resolved_profile: acme:profile:1");
    assert_system_contains(&requests[0], "hook_call_number: 1");

    assert_eq!(requests[1].model, "fake-accurate");
    assert_eq!(requests[1].max_tokens, Some(64));
    assert_system_contains(&requests[1], "resolved_profile: acme:profile:2");
    assert_system_contains(&requests[1], "hook_call_number: 2");
    assert!(
        requests[1]
            .messages
            .contains(&Message::assistant_with_tool_calls("", vec![tool_call]))
    );
    assert!(
        requests[1]
            .messages
            .contains(&Message::tool("call_1", r#"{"tier":"enterprise"}"#))
    );
}

#[tokio::test]
async fn unknown_explicit_model_mode_fails_before_provider_call() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = create_agent(
        AgentConfig::new(
            RecordingProvider::new(vec![text_response("should not run")], Arc::clone(&requests)),
            "fake-balanced",
        )
        .with_model_modes([("fast", "fake-fast")]),
    );

    let error = agent
        .run_with_options(RunOptions::new("hello").with_model_mode("missing"))
        .await
        .unwrap_err();

    match error {
        LoopError::Provider(orchestrai::provider::ProviderError::Config(message)) => {
            assert_eq!(message, "model mode `missing` is not configured");
        }
        other => panic!("expected model-mode configuration failure, got {other:?}"),
    }
    assert!(requests.lock().unwrap().is_empty());
}

fn assert_system_contains(request: &ModelRequest, expected: &str) {
    assert!(
        request
            .messages
            .iter()
            .any(|message| message.role == orchestrai::Role::System
                && message.content.contains(expected)),
        "expected system message to contain {expected:?}: {:?}",
        request.messages
    );
}

fn assert_system_excludes(request: &ModelRequest, forbidden: &str) {
    assert!(
        request
            .messages
            .iter()
            .filter(|message| message.role == orchestrai::Role::System)
            .all(|message| !message.content.contains(forbidden)),
        "system message should not contain {forbidden:?}: {:?}",
        request.messages
    );
}

fn text_response(message: impl Into<String>) -> ModelResponse {
    ModelResponse {
        message: message.into(),
        tool_calls: Vec::new(),
        usage: None,
    }
}

struct RecordingProvider {
    responses: Mutex<VecDeque<ModelResponse>>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
}

impl RecordingProvider {
    fn new(responses: Vec<ModelResponse>, requests: Arc<Mutex<Vec<ModelRequest>>>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            requests,
        }
    }
}

#[async_trait]
impl ModelProvider for RecordingProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        Ok(self.responses.lock().unwrap().pop_front().unwrap())
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        Ok(Box::pin(stream::iter(vec![Ok(ModelStreamEvent::Done)])))
    }
}
