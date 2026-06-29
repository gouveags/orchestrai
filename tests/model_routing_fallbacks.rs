use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures_util::stream;
use orchestrai::{
    AgentConfig, FallbackPolicy, ModelCatalog, ModelProvider, ModelRequest, ModelResponse,
    ModelStream, ProviderModel, ProviderRegistry, RoutedModelProvider, create_agent,
    loop_runner::LoopError,
    provider::{ModelStreamEvent, ProviderError, ProviderResult},
};

#[tokio::test]
async fn model_catalog_resolves_logical_alias_to_provider_model_pair() {
    let openai = Arc::new(FakeProvider::new([Outcome::success("regular answer")]));
    let anthropic = Arc::new(FakeProvider::new([Outcome::success("max answer")]));
    let provider = RoutedModelProvider::new(
        ProviderRegistry::new()
            .with_provider("openai", Arc::clone(&openai))
            .with_provider("anthropic", Arc::clone(&anthropic)),
        ModelCatalog::new()
            .with_alias(
                "movedot-regular",
                [ProviderModel::new("openai", "gpt-4.1-mini")],
            )
            .with_alias(
                "movedot-max",
                [ProviderModel::new("anthropic", "claude-opus-4-20250514")],
            ),
    )
    .with_fallback_policy(FallbackPolicy::disabled());
    let agent = create_agent(AgentConfig::new(provider, "movedot-regular"));

    let output = agent.run("hello").await.unwrap();

    assert_eq!(output.final_message, "regular answer");
    assert_eq!(openai.requested_models(), vec!["gpt-4.1-mini"]);
    assert!(anthropic.requested_models().is_empty());
}

#[tokio::test]
async fn transient_provider_errors_fall_through_to_next_catalog_entry() {
    let primary = Arc::new(FakeProvider::new([Outcome::transient_failure()]));
    let fallback = Arc::new(FakeProvider::new([Outcome::success("fallback answer")]));
    let provider = RoutedModelProvider::new(
        ProviderRegistry::new()
            .with_provider("anthropic", Arc::clone(&primary))
            .with_provider("openai", Arc::clone(&fallback)),
        ModelCatalog::new().with_alias(
            "movedot-max",
            [
                ProviderModel::new("anthropic", "claude-opus-4-20250514"),
                ProviderModel::new("openai", "gpt-4.1"),
            ],
        ),
    )
    .with_fallback_policy(FallbackPolicy::transient_provider_errors());
    let agent = create_agent(AgentConfig::new(provider, "movedot-max"));

    let output = agent.run("hello").await.unwrap();

    assert_eq!(output.final_message, "fallback answer");
    assert_eq!(primary.requested_models(), vec!["claude-opus-4-20250514"]);
    assert_eq!(fallback.requested_models(), vec!["gpt-4.1"]);
}

#[tokio::test]
async fn hard_provider_errors_surface_without_trying_fallback_models() {
    let primary = Arc::new(FakeProvider::new([Outcome::hard_failure()]));
    let fallback = Arc::new(FakeProvider::new([Outcome::success("should not run")]));
    let provider = RoutedModelProvider::new(
        ProviderRegistry::new()
            .with_provider("anthropic", Arc::clone(&primary))
            .with_provider("openai", Arc::clone(&fallback)),
        ModelCatalog::new().with_alias(
            "movedot-max",
            [
                ProviderModel::new("anthropic", "claude-opus-4-20250514"),
                ProviderModel::new("openai", "gpt-4.1"),
            ],
        ),
    )
    .with_fallback_policy(FallbackPolicy::transient_provider_errors());
    let agent = create_agent(AgentConfig::new(provider, "movedot-max"));

    let error = agent.run("hello").await.unwrap_err();

    match error {
        LoopError::Provider(ProviderError::Status { status, body }) => {
            assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
            assert_eq!(body, "bad api key");
        }
        other => panic!("expected hard provider status error, got {other:?}"),
    }
    assert_eq!(primary.requested_models(), vec!["claude-opus-4-20250514"]);
    assert!(fallback.requested_models().is_empty());
}

struct FakeProvider {
    outcomes: Mutex<VecDeque<Outcome>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl FakeProvider {
    fn new(outcomes: impl IntoIterator<Item = Outcome>) -> Self {
        Self {
            outcomes: Mutex::new(outcomes.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requested_models(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.model.clone())
            .collect()
    }
}

struct Outcome {
    result: Result<ModelResponse, FakeFailure>,
}

impl Outcome {
    fn success(message: impl Into<String>) -> Self {
        Self {
            result: Ok(ModelResponse {
                message: message.into(),
                tool_calls: Vec::new(),
                usage: None,
            }),
        }
    }

    fn transient_failure() -> Self {
        Self {
            result: Err(FakeFailure::Transient),
        }
    }

    fn hard_failure() -> Self {
        Self {
            result: Err(FakeFailure::Hard),
        }
    }
}

enum FakeFailure {
    Transient,
    Hard,
}

impl FakeFailure {
    fn into_provider_error(self) -> ProviderError {
        match self {
            Self::Transient => ProviderError::Status {
                status: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                body: "temporary outage".to_owned(),
            },
            Self::Hard => ProviderError::Status {
                status: reqwest::StatusCode::UNAUTHORIZED,
                body: "bad api key".to_owned(),
            },
        }
    }
}

#[async_trait]
impl ModelProvider for FakeProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        self.outcomes
            .lock()
            .unwrap()
            .pop_front()
            .expect("fake provider received unexpected request")
            .result
            .map_err(FakeFailure::into_provider_error)
    }

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
        let response = self.complete(request).await?;
        Ok(Box::pin(stream::iter([
            Ok(ModelStreamEvent::MessageDelta(response.message)),
            Ok(ModelStreamEvent::Done),
        ])))
    }
}
