use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    environment::{EnvironmentError, FileEnvironment},
    tool::{Tool, ToolError, ToolRegistry, ToolResult},
    types::ToolDefinition,
};

pub const READ_FILE_TOOL: &str = "fs_read";
pub const WRITE_FILE_TOOL: &str = "fs_write";
pub const LIST_FILES_TOOL: &str = "fs_list";
pub const SEARCH_FILES_TOOL: &str = "fs_search";

pub fn register_filesystem_tools(
    registry: &mut ToolRegistry,
    environment: Arc<dyn FileEnvironment>,
) {
    registry.register(FilesystemTool::new(
        FilesystemOperation::Read,
        Arc::clone(&environment),
    ));
    registry.register(FilesystemTool::new(
        FilesystemOperation::Write,
        Arc::clone(&environment),
    ));
    registry.register(FilesystemTool::new(
        FilesystemOperation::List,
        Arc::clone(&environment),
    ));
    registry.register(FilesystemTool::new(
        FilesystemOperation::Search,
        environment,
    ));
}

struct FilesystemTool {
    operation: FilesystemOperation,
    environment: Arc<dyn FileEnvironment>,
}

impl FilesystemTool {
    fn new(operation: FilesystemOperation, environment: Arc<dyn FileEnvironment>) -> Self {
        Self {
            operation,
            environment,
        }
    }
}

#[async_trait]
impl Tool for FilesystemTool {
    fn definition(&self) -> ToolDefinition {
        self.operation.definition()
    }

    async fn execute(&self, arguments: Value) -> ToolResult<String> {
        match self.operation {
            FilesystemOperation::Read => {
                let input: ReadFileInput = parse_arguments(arguments)?;
                let content = self
                    .environment
                    .read_text(&input.path)
                    .map_err(tool_error_from_environment)?;
                Ok(json!({
                    "path": input.path,
                    "content": content,
                })
                .to_string())
            }
            FilesystemOperation::Write => {
                let input: WriteFileInput = parse_arguments(arguments)?;
                let output = self
                    .environment
                    .write_text(&input.path, &input.content)
                    .map_err(tool_error_from_environment)?;
                serde_json::to_string(&output)
                    .map_err(|error| ToolError::Execution(error.to_string()))
            }
            FilesystemOperation::List => {
                let input: ListFilesInput = parse_arguments(arguments)?;
                let entries = self
                    .environment
                    .list(input.path.as_deref().unwrap_or(""))
                    .map_err(tool_error_from_environment)?;
                Ok(json!({ "entries": entries }).to_string())
            }
            FilesystemOperation::Search => {
                let input: SearchFilesInput = parse_arguments(arguments)?;
                let output = self
                    .environment
                    .search_text(
                        &input.query,
                        input.path.as_deref(),
                        input.max_results.unwrap_or(20),
                    )
                    .map_err(tool_error_from_environment)?;
                serde_json::to_string(&output)
                    .map_err(|error| ToolError::Execution(error.to_string()))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FilesystemOperation {
    Read,
    Write,
    List,
    Search,
}

impl FilesystemOperation {
    fn definition(self) -> ToolDefinition {
        match self {
            Self::Read => ToolDefinition::new(
                READ_FILE_TOOL,
                "Read a UTF-8 text file from the configured environment root.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path inside the environment root."}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            ),
            Self::Write => ToolDefinition::new(
                WRITE_FILE_TOOL,
                "Write UTF-8 text to a file inside the configured environment root.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path inside the environment root."},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }),
            ),
            Self::List => ToolDefinition::new(
                LIST_FILES_TOOL,
                "List files and directories inside the configured environment root.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory path inside the environment root."}
                    },
                    "additionalProperties": false
                }),
            ),
            Self::Search => ToolDefinition::new(
                SEARCH_FILES_TOOL,
                "Search UTF-8 text files under the configured environment root.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "description": "Optional relative directory path inside the environment root."},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 200}
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
            ),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReadFileInput {
    path: String,
}

#[derive(Debug, Deserialize)]
struct WriteFileInput {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ListFilesInput {
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchFilesInput {
    query: String,
    path: Option<String>,
    max_results: Option<usize>,
}

fn parse_arguments<T>(arguments: Value) -> ToolResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(arguments).map_err(|error| {
        ToolError::result_content(
            json!({
                "error": format!("invalid tool arguments: {error}")
            })
            .to_string(),
        )
    })
}

fn tool_error_from_environment(error: EnvironmentError) -> ToolError {
    if error.is_security_failure() {
        ToolError::Execution(error.to_string())
    } else {
        ToolError::result_content(
            json!({
                "error": error.to_string()
            })
            .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use serde_json::json;

    use super::*;
    use crate::{environment::LocalEnvironment, tool::ToolRegistry};

    #[tokio::test]
    async fn filesystem_tools_read_write_list_and_search() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("notes")).unwrap();
        let environment = Arc::new(LocalEnvironment::new(temp.path()).unwrap());
        let mut registry = ToolRegistry::new();
        register_filesystem_tools(&mut registry, environment);

        let write = registry
            .execute(
                WRITE_FILE_TOOL,
                json!({"path": "notes/todo.txt", "content": "alpha\nbeta"}),
            )
            .await
            .unwrap();
        assert_eq!(write, r#"{"path":"notes/todo.txt","bytes_written":10}"#);

        let read = registry
            .execute(READ_FILE_TOOL, json!({"path": "notes/todo.txt"}))
            .await
            .unwrap();
        assert_eq!(read, r#"{"content":"alpha\nbeta","path":"notes/todo.txt"}"#);

        let list = registry
            .execute(LIST_FILES_TOOL, json!({"path": "notes"}))
            .await
            .unwrap();
        assert!(list.contains(r#""path":"notes/todo.txt""#));

        let search = registry
            .execute(SEARCH_FILES_TOOL, json!({"query": "beta"}))
            .await
            .unwrap();
        assert!(search.contains(r#""text":"beta""#));
    }

    #[tokio::test]
    async fn filesystem_tools_return_missing_files_as_tool_content() {
        let temp = tempfile::tempdir().unwrap();
        let environment = Arc::new(LocalEnvironment::new(temp.path()).unwrap());
        let mut registry = ToolRegistry::new();
        register_filesystem_tools(&mut registry, environment);

        let error = registry
            .execute(READ_FILE_TOOL, json!({"path": "missing.txt"}))
            .await
            .unwrap_err();

        assert!(matches!(error, ToolError::ResultContent(_)));
    }

    #[tokio::test]
    async fn filesystem_tools_fail_hard_on_root_escape_attempts() {
        let temp = tempfile::tempdir().unwrap();
        let environment = Arc::new(LocalEnvironment::new(temp.path()).unwrap());
        let mut registry = ToolRegistry::new();
        register_filesystem_tools(&mut registry, environment);

        let error = registry
            .execute(READ_FILE_TOOL, json!({"path": "../secret.txt"}))
            .await
            .unwrap_err();

        assert!(matches!(error, ToolError::Execution(_)));
    }
}
