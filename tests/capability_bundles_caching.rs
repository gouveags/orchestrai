use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::stream;
use serde_json::json;

use orchestrai::{
    AgentConfig, CacheHint, CacheHintScope, CachePolicy, CapabilityBundle, CapabilityBundleSet,
    CapabilityError, CapabilitySelection, FnTool, LoopError, ModelCatalog, ModelProvider,
    ModelRequest, ModelResponse, ModelStream, ModelStreamEvent, ProviderError, ProviderFeature,
    ProviderModel, ProviderRegistry, ProviderResult, RoutedModelProvider, Tool, ToolDefinition,
    create_agent,
};

#[tokio::test]
async fn run_loads_default_and_selected_bundle_prompts_and_tools() {
    let requests = RecordedRequests::default();
    let agent = create_agent(
        AgentConfig::new(
            InspectingProvider::supported(requests.clone()),
            "fake-model",
        )
        .with_capability_bundles(
            CapabilityBundleSet::new()
                .with_default(
                    CapabilityBundle::new("default")
                        .with_prompt("Base prompt: answer with repository context.")
                        .with_tool(fake_tool("repo.search")),
                )
                .with_bundle(
                    "maintainer",
                    CapabilityBundle::new("maintainer")
                        .with_prompt("Role prompt: prefer small reviewable changes.")
                        .with_tool(fake_tool("github.review_threads")),
                )
                .with_bundle(
                    "tests_only",
                    CapabilityBundle::new("tests_only")
                        .with_prompt("Policy prompt: do not edit src files.")
                        .with_tool(fake_tool("cargo.focused_test")),
                ),
        ),
    );

    let output = agent
        .run_with_capabilities(
            "add the missing coverage",
            CapabilitySelection::new(["maintainer", "tests_only"]),
        )
        .await
        .unwrap();

    assert_eq!(output.final_message, "done");
    let requests = requests.take();
    assert_eq!(requests.len(), 1);

    let request = &requests[0];
    assert_system_contains(request, "Base prompt: answer with repository context.");
    assert_system_contains(request, "Role prompt: prefer small reviewable changes.");
    assert_system_contains(request, "Policy prompt: do not edit src files.");

    let mut tool_names = request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    tool_names.sort_unstable();
    assert_eq!(
        tool_names,
        vec!["cargo.focused_test", "github.review_threads", "repo.search"]
    );
}

#[tokio::test]
async fn duplicate_tools_across_active_capability_bundles_fail_before_provider_call() {
    let requests = RecordedRequests::default();
    let agent = create_agent(
        AgentConfig::new(
            InspectingProvider::supported(requests.clone()),
            "fake-model",
        )
        .with_capability_bundles(
            CapabilityBundleSet::new()
                .with_default(CapabilityBundle::new("default").with_tool(fake_tool("repo.search")))
                .with_bundle(
                    "readonly",
                    CapabilityBundle::new("readonly").with_tool(fake_tool("repo.search")),
                ),
        ),
    );

    let error = agent
        .run_with_capabilities(
            "search for the regression",
            CapabilitySelection::new(["readonly"]),
        )
        .await
        .unwrap_err();

    assert!(
        matches!(
            error,
            LoopError::Capability(CapabilityError::DuplicateTool { ref name })
                if name == "repo.search"
        ),
        "expected duplicate tool capability error, got {error:?}"
    );
    assert!(requests.take().is_empty(), "provider must not be called");
}

#[tokio::test]
async fn prompt_cache_hints_are_sent_to_supported_providers_and_unsupported_errors_propagate() {
    let supported_requests = RecordedRequests::default();
    let supported_agent = create_agent(
        AgentConfig::new(
            InspectingProvider::supported(supported_requests.clone()),
            "fake-model",
        )
        .with_capability_bundles(
            CapabilityBundleSet::new().with_default(
                CapabilityBundle::new("default")
                    .with_prompt("Base prompt: stable reusable instructions.")
                    .with_tool(fake_tool("repo.search")),
            ),
        )
        .with_cache_policy(CachePolicy::provider_prompt_cache()),
    );

    supported_agent.run("use the cached context").await.unwrap();

    let requests = supported_requests.take();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].cache_hints,
        vec![
            CacheHint::new(CacheHintScope::SystemPrompt { message_index: 0 })
                .for_feature(ProviderFeature::PromptCache),
            CacheHint::new(CacheHintScope::ToolDefinitions)
                .for_feature(ProviderFeature::PromptCache),
        ]
    );

    let unsupported_requests = RecordedRequests::default();
    let unsupported_agent = create_agent(
        AgentConfig::new(
            InspectingProvider::unsupported(unsupported_requests.clone()),
            "fake-model",
        )
        .with_capability_bundles(CapabilityBundleSet::new().with_default(
            CapabilityBundle::new("default").with_prompt("Base prompt: stable instructions."),
        ))
        .with_cache_policy(CachePolicy::provider_prompt_cache()),
    );

    let error = unsupported_agent
        .run("this should surface the provider failure")
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            LoopError::Provider(ProviderError::UnsupportedFeature(
                ProviderFeature::PromptCache
            ))
        ),
        "unsupported prompt-cache providers should fail explicitly, got {error:?}"
    );
    assert_eq!(
        unsupported_requests.take().len(),
        0,
        "unsupported cache hints should fail before the provider call"
    );
}

#[tokio::test]
async fn prompt_cache_support_is_checked_after_model_alias_routing() {
    let routed_requests = RecordedRequests::default();
    let provider = RoutedModelProvider::new(
        ProviderRegistry::new().with_provider(
            "anthropic",
            Arc::new(InspectingProvider::supported(routed_requests.clone())),
        ),
        ModelCatalog::new().with_alias(
            "movedot-regular",
            [ProviderModel::new("anthropic", "claude-sonnet-4-6")],
        ),
    );
    let agent = create_agent(
        AgentConfig::new(provider, "movedot-regular")
            .with_capability_bundles(CapabilityBundleSet::new().with_default(
                CapabilityBundle::new("default").with_prompt("Cached routed prompt."),
            ))
            .with_cache_policy(CachePolicy::provider_prompt_cache()),
    );

    agent.run("use routed prompt cache").await.unwrap();

    let requests = routed_requests.take();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].model, "claude-sonnet-4-6");
    assert_eq!(
        requests[0].cache_hints,
        vec![
            CacheHint::new(CacheHintScope::SystemPrompt { message_index: 0 })
                .for_feature(ProviderFeature::PromptCache),
        ]
    );
}

fn assert_system_contains(request: &ModelRequest, expected: &str) {
    assert!(
        request
            .messages
            .iter()
            .any(|message| message.role == orchestrai::Role::System
                && message.content.contains(expected)),
        "expected system prompt to contain {expected:?}: {:?}",
        request.messages
    );
}

fn fake_tool(name: &'static str) -> impl Tool {
    FnTool::new(
        ToolDefinition::new(
            name,
            "fake test tool",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false,
            }),
        ),
        |_| Box::pin(async { Ok("ok".to_owned()) }),
    )
}

#[derive(Clone, Default)]
struct RecordedRequests {
    inner: Arc<Mutex<Vec<ModelRequest>>>,
}

impl RecordedRequests {
    fn push(&self, request: ModelRequest) {
        self.inner.lock().unwrap().push(request);
    }

    fn take(&self) -> Vec<ModelRequest> {
        std::mem::take(&mut self.inner.lock().unwrap())
    }
}

struct InspectingProvider {
    requests: RecordedRequests,
    supports_prompt_cache: bool,
}

impl InspectingProvider {
    fn supported(requests: RecordedRequests) -> Self {
        Self {
            requests,
            supports_prompt_cache: true,
        }
    }

    fn unsupported(requests: RecordedRequests) -> Self {
        Self {
            requests,
            supports_prompt_cache: false,
        }
    }
}

#[async_trait]
impl ModelProvider for InspectingProvider {
    fn supports(&self, feature: ProviderFeature) -> bool {
        match feature {
            ProviderFeature::PromptCache => self.supports_prompt_cache,
        }
    }

    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        let has_prompt_cache_hint = request
            .cache_hints
            .iter()
            .any(|hint| hint.feature == ProviderFeature::PromptCache);
        self.requests.push(request);

        if has_prompt_cache_hint && !self.supports_prompt_cache {
            return Err(ProviderError::UnsupportedFeature(
                ProviderFeature::PromptCache,
            ));
        }

        Ok(ModelResponse {
            message: "done".to_owned(),
            tool_calls: Vec::new(),
            usage: None,
        })
    }

    async fn stream(&self, _request: ModelRequest) -> ProviderResult<ModelStream> {
        Ok(Box::pin(stream::iter(vec![Ok(ModelStreamEvent::Done)])))
    }
}
