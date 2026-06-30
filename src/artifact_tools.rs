use std::{
    fs,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    tool::{Tool, ToolError, ToolRegistry, ToolResult},
    types::ToolDefinition,
};

pub const PUBLISH_ARTIFACT_TOOL: &str = "artifact_publish";
pub const READ_ARTIFACT_TOOL: &str = "artifact_read";
pub const LIST_ARTIFACTS_TOOL: &str = "artifact_list";

pub fn register_artifact_tools(registry: &mut ToolRegistry, store: Arc<dyn ArtifactStore>) {
    registry.register(ArtifactTool::new(
        ArtifactOperation::Publish,
        Arc::clone(&store),
    ));
    registry.register(ArtifactTool::new(
        ArtifactOperation::Read,
        Arc::clone(&store),
    ));
    registry.register(ArtifactTool::new(ArtifactOperation::List, store));
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub id: String,
    pub title: String,
    pub path: String,
    pub media_type: String,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactContent {
    pub id: String,
    pub title: String,
    pub media_type: String,
    pub content: String,
}

pub trait ArtifactStore: Send + Sync {
    fn publish(&self, input: PublishArtifact) -> ArtifactResult<ArtifactMetadata>;

    fn read(&self, id: &str) -> ArtifactResult<ArtifactContent>;

    fn list(&self) -> ArtifactResult<Vec<ArtifactMetadata>>;
}

#[derive(Debug, Clone)]
pub struct LocalArtifactStore {
    root: PathBuf,
    index: Arc<Mutex<Vec<ArtifactMetadata>>>,
}

impl LocalArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> ArtifactResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|source| ArtifactError::Io {
            path: root.display().to_string(),
            source,
        })?;
        let root = root.canonicalize().map_err(|source| ArtifactError::Io {
            path: root.display().to_string(),
            source,
        })?;

        if !root.is_dir() {
            return Err(ArtifactError::Configuration(format!(
                "artifact root `{}` is not a directory",
                root.display()
            )));
        }

        Ok(Self {
            root,
            index: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve_for_write(
        &self,
        requested: Option<&str>,
        fallback: &str,
    ) -> ArtifactResult<PathBuf> {
        let path = requested.unwrap_or(fallback);
        let path = normalize_artifact_path(path)?;
        let mut target = self.root.clone();

        for component in Path::new(&path).components() {
            match component {
                Component::Normal(part) => target.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(ArtifactError::OutsideRoot(path));
                }
            }
        }

        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|source| ArtifactError::Io {
                path: path.clone(),
                source,
            })?;
            let parent = parent.canonicalize().map_err(|source| ArtifactError::Io {
                path: path.clone(),
                source,
            })?;
            self.ensure_in_root(&parent, &path)?;
        }

        if target.exists() {
            let resolved = target.canonicalize().map_err(|source| ArtifactError::Io {
                path: path.clone(),
                source,
            })?;
            self.ensure_in_root(&resolved, &path)?;
        }

        Ok(target)
    }

    fn resolve_existing(&self, path: &str) -> ArtifactResult<PathBuf> {
        let target = self.resolve_for_write(Some(path), path)?;
        let resolved = target.canonicalize().map_err(|source| ArtifactError::Io {
            path: path.to_owned(),
            source,
        })?;
        self.ensure_in_root(&resolved, path)?;
        Ok(resolved)
    }

    fn ensure_in_root(&self, path: &Path, requested: &str) -> ArtifactResult<()> {
        if path.starts_with(&self.root) {
            Ok(())
        } else {
            Err(ArtifactError::OutsideRoot(requested.to_owned()))
        }
    }
}

impl ArtifactStore for LocalArtifactStore {
    fn publish(&self, input: PublishArtifact) -> ArtifactResult<ArtifactMetadata> {
        let id = new_artifact_id();
        let fallback = format!(
            "{}-{}",
            id,
            default_file_name(&input.title, &input.media_type)
        );
        let target = self.resolve_for_write(input.path.as_deref(), &fallback)?;
        fs::write(&target, &input.content).map_err(|source| ArtifactError::Io {
            path: target.display().to_string(),
            source,
        })?;
        let path = target
            .strip_prefix(&self.root)
            .unwrap_or(&target)
            .to_string_lossy()
            .to_string();
        let metadata = ArtifactMetadata {
            id,
            title: input.title,
            path,
            media_type: input.media_type,
            bytes: input.content.len(),
        };
        self.index.lock().unwrap().push(metadata.clone());
        Ok(metadata)
    }

    fn read(&self, id: &str) -> ArtifactResult<ArtifactContent> {
        let metadata = self
            .index
            .lock()
            .unwrap()
            .iter()
            .find(|artifact| artifact.id == id)
            .cloned()
            .ok_or_else(|| ArtifactError::NotFound(id.to_owned()))?;
        let path = self.resolve_existing(&metadata.path)?;
        let content = fs::read_to_string(&path).map_err(|source| ArtifactError::Io {
            path: metadata.path.clone(),
            source,
        })?;
        Ok(ArtifactContent {
            id: metadata.id,
            title: metadata.title,
            media_type: metadata.media_type,
            content,
        })
    }

    fn list(&self) -> ArtifactResult<Vec<ArtifactMetadata>> {
        Ok(self.index.lock().unwrap().clone())
    }
}

pub type ArtifactResult<T> = Result<T, ArtifactError>;

#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("artifact configuration is invalid: {0}")]
    Configuration(String),
    #[error("path `{0}` is outside the artifact root")]
    OutsideRoot(String),
    #[error("artifact `{0}` was not found")]
    NotFound(String),
    #[error("artifact operation failed for `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl ArtifactError {
    pub fn is_security_failure(&self) -> bool {
        matches!(self, Self::Configuration(_) | Self::OutsideRoot(_))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PublishArtifact {
    pub title: String,
    pub content: String,
    #[serde(default = "default_media_type")]
    pub media_type: String,
    pub path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReadArtifact {
    id: String,
}

struct ArtifactTool {
    operation: ArtifactOperation,
    store: Arc<dyn ArtifactStore>,
}

impl ArtifactTool {
    fn new(operation: ArtifactOperation, store: Arc<dyn ArtifactStore>) -> Self {
        Self { operation, store }
    }
}

#[async_trait]
impl Tool for ArtifactTool {
    fn definition(&self) -> ToolDefinition {
        self.operation.definition()
    }

    async fn execute(&self, arguments: Value) -> ToolResult<String> {
        match self.operation {
            ArtifactOperation::Publish => {
                let input: PublishArtifact = parse_arguments(arguments)?;
                serde_json::to_string(
                    &self
                        .store
                        .publish(input)
                        .map_err(tool_error_from_artifact)?,
                )
                .map_err(|error| ToolError::Execution(error.to_string()))
            }
            ArtifactOperation::Read => {
                let input: ReadArtifact = parse_arguments(arguments)?;
                serde_json::to_string(
                    &self
                        .store
                        .read(&input.id)
                        .map_err(tool_error_from_artifact)?,
                )
                .map_err(|error| ToolError::Execution(error.to_string()))
            }
            ArtifactOperation::List => Ok(json!({
                "artifacts": self.store.list().map_err(tool_error_from_artifact)?
            })
            .to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ArtifactOperation {
    Publish,
    Read,
    List,
}

impl ArtifactOperation {
    fn definition(self) -> ToolDefinition {
        match self {
            Self::Publish => ToolDefinition::new(
                PUBLISH_ARTIFACT_TOOL,
                "Publish a text artifact to the configured artifact store.",
                json!({
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "media_type": {"type": "string", "default": "text/plain"},
                        "path": {"type": "string", "description": "Optional relative artifact path."}
                    },
                    "required": ["title", "content"],
                    "additionalProperties": false
                }),
            ),
            Self::Read => ToolDefinition::new(
                READ_ARTIFACT_TOOL,
                "Read a text artifact by id from the configured artifact store.",
                json!({
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    },
                    "required": ["id"],
                    "additionalProperties": false
                }),
            ),
            Self::List => ToolDefinition::new(
                LIST_ARTIFACTS_TOOL,
                "List artifacts published during this run.",
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            ),
        }
    }
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

fn tool_error_from_artifact(error: ArtifactError) -> ToolError {
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

fn default_media_type() -> String {
    "text/plain".to_owned()
}

fn normalize_artifact_path(path: &str) -> ArtifactResult<String> {
    let path = path.trim();
    if path.is_empty() || Path::new(path).is_absolute() {
        return Err(ArtifactError::OutsideRoot(path.to_owned()));
    }
    Ok(path.to_owned())
}

fn default_file_name(title: &str, media_type: &str) -> String {
    let stem = title
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    let stem = if stem.is_empty() { "artifact" } else { &stem };
    format!("{stem}.{}", extension_for_media_type(media_type))
}

fn extension_for_media_type(media_type: &str) -> &'static str {
    match media_type {
        "text/markdown" => "md",
        "application/json" => "json",
        "text/html" => "html",
        _ => "txt",
    }
}

fn new_artifact_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("artifact_{nanos}")
}
