use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;

use crate::types::{Message, ToolCall, ToolDefinition, Usage};

pub type ProviderResult<T> = Result<T, ProviderError>;
pub type ModelStream = Pin<Box<dyn Stream<Item = ProviderResult<ModelStreamEvent>> + Send>>;

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
    pub cache_hints: Vec<CacheHint>,
}

impl ModelRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            max_tokens: None,
            cache_hints: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderFeature {
    PromptCache,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheHint {
    pub scope: CacheHintScope,
    pub feature: ProviderFeature,
}

impl CacheHint {
    pub fn new(scope: CacheHintScope) -> Self {
        Self {
            scope,
            feature: ProviderFeature::PromptCache,
        }
    }

    pub fn for_feature(mut self, feature: ProviderFeature) -> Self {
        self.feature = feature;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheHintScope {
    SystemPrompt { message_index: usize },
    ToolDefinitions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CachePolicy {
    provider_prompt_cache: bool,
}

impl CachePolicy {
    pub fn disabled() -> Self {
        Self {
            provider_prompt_cache: false,
        }
    }

    pub fn provider_prompt_cache() -> Self {
        Self {
            provider_prompt_cache: true,
        }
    }

    pub(crate) fn uses_provider_prompt_cache(self) -> bool {
        self.provider_prompt_cache
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelResponse {
    pub message: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStreamEvent {
    MessageDelta(String),
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    Usage(Usage),
    Done,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn supports(&self, _feature: ProviderFeature) -> bool {
        false
    }

    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse>;

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream>;
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("provider request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("provider returned status {status}: {body}")]
    Status {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("provider response was missing field `{0}`")]
    MissingField(&'static str),
    #[error("provider response could not be parsed: {0}")]
    Parse(String),
    #[error("provider configuration is invalid: {0}")]
    Config(String),
    #[error("provider does not support feature {0:?}")]
    UnsupportedFeature(ProviderFeature),
}

pub(crate) async fn error_for_status(
    response: reqwest::Response,
) -> ProviderResult<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_else(|_| String::new());
    Err(ProviderError::Status { status, body })
}
