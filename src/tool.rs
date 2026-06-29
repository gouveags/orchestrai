use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

use async_trait::async_trait;
use serde_json::Value;

use crate::types::ToolDefinition;

pub type ToolResult<T> = Result<T, ToolError>;
type BoxToolFuture = Pin<Box<dyn Future<Output = ToolResult<String>> + Send>>;

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn execute(&self, arguments: Value) -> ToolResult<String>;
}

pub struct FnTool<F>
where
    F: Fn(Value) -> BoxToolFuture + Send + Sync,
{
    definition: ToolDefinition,
    handler: F,
}

impl<F> FnTool<F>
where
    F: Fn(Value) -> BoxToolFuture + Send + Sync,
{
    pub fn new(definition: ToolDefinition, handler: F) -> Self {
        Self {
            definition,
            handler,
        }
    }
}

#[async_trait]
impl<F> Tool for FnTool<F>
where
    F: Fn(Value) -> BoxToolFuture + Send + Sync,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(&self, arguments: Value) -> ToolResult<String> {
        (self.handler)(arguments).await
    }
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.insert(tool.definition().name, Arc::new(tool));
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|tool| tool.definition()).collect()
    }

    pub async fn execute(&self, name: &str, arguments: Value) -> ToolResult<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_owned()))?;

        tool.execute(arguments).await
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool `{0}` was not registered")]
    NotFound(String),
    #[error("tool returned error content: {0}")]
    ResultContent(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
}

impl ToolError {
    pub fn result_content(content: impl Into<String>) -> Self {
        Self::ResultContent(content.into())
    }
}
