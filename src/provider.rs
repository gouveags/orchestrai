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
}

impl ModelRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            max_tokens: None,
        }
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
