use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub type EnvironmentResult<T> = Result<T, EnvironmentError>;

pub trait FileEnvironment: Send + Sync {
    fn read_text(&self, path: &str) -> EnvironmentResult<String>;

    fn write_text(&self, path: &str, content: &str) -> EnvironmentResult<WriteFileOutput>;

    fn list(&self, path: &str) -> EnvironmentResult<Vec<DirectoryEntry>>;

    fn search_text(
        &self,
        query: &str,
        path: Option<&str>,
        max_results: usize,
    ) -> EnvironmentResult<SearchOutput>;
}

#[derive(Debug, Clone)]
pub struct LocalEnvironment {
    root: PathBuf,
}

impl LocalEnvironment {
    pub fn new(root: impl Into<PathBuf>) -> EnvironmentResult<Self> {
        let root = root.into();
        let root = root.canonicalize().map_err(|source| {
            EnvironmentError::Configuration(format!(
                "environment root `{}` is not accessible: {source}",
                root.display()
            ))
        })?;

        if !root.is_dir() {
            return Err(EnvironmentError::Configuration(format!(
                "environment root `{}` is not a directory",
                root.display()
            )));
        }

        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve_existing(&self, path: &str) -> EnvironmentResult<PathBuf> {
        let requested = self.join_agent_path(path)?;
        let resolved = requested
            .canonicalize()
            .map_err(|source| EnvironmentError::Io {
                path: path.to_owned(),
                source,
            })?;
        self.ensure_in_root(&resolved, path)?;
        Ok(resolved)
    }

    fn resolve_for_write(&self, path: &str) -> EnvironmentResult<PathBuf> {
        let requested = self.join_agent_path(path)?;

        if requested.exists() {
            let resolved = requested
                .canonicalize()
                .map_err(|source| EnvironmentError::Io {
                    path: path.to_owned(),
                    source,
                })?;
            self.ensure_in_root(&resolved, path)?;
            return Ok(requested);
        }

        let parent = requested.parent().unwrap_or(&self.root);
        let resolved_parent = parent
            .canonicalize()
            .map_err(|source| EnvironmentError::Io {
                path: path.to_owned(),
                source,
            })?;
        self.ensure_in_root(&resolved_parent, path)?;
        Ok(requested)
    }

    fn join_agent_path(&self, path: &str) -> EnvironmentResult<PathBuf> {
        let path = path.trim();
        if path.is_empty() {
            return Ok(self.root.clone());
        }

        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(EnvironmentError::OutsideRoot(path.to_owned()));
        }

        let mut joined = self.root.clone();
        for component in candidate.components() {
            match component {
                Component::Normal(part) => joined.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(EnvironmentError::OutsideRoot(path.to_owned()));
                }
            }
        }
        Ok(joined)
    }

    fn ensure_in_root(&self, path: &Path, requested: &str) -> EnvironmentResult<()> {
        if path.starts_with(&self.root) {
            Ok(())
        } else {
            Err(EnvironmentError::OutsideRoot(requested.to_owned()))
        }
    }

    fn relative_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .trim_start_matches(std::path::MAIN_SEPARATOR)
            .to_owned()
    }
}

impl FileEnvironment for LocalEnvironment {
    fn read_text(&self, path: &str) -> EnvironmentResult<String> {
        let resolved = self.resolve_existing(path)?;
        fs::read_to_string(&resolved).map_err(|source| EnvironmentError::Io {
            path: path.to_owned(),
            source,
        })
    }

    fn write_text(&self, path: &str, content: &str) -> EnvironmentResult<WriteFileOutput> {
        let resolved = self.resolve_for_write(path)?;
        fs::write(&resolved, content).map_err(|source| EnvironmentError::Io {
            path: path.to_owned(),
            source,
        })?;
        Ok(WriteFileOutput {
            path: path.to_owned(),
            bytes_written: content.len(),
        })
    }

    fn list(&self, path: &str) -> EnvironmentResult<Vec<DirectoryEntry>> {
        let resolved = self.resolve_existing(path)?;
        let mut entries = Vec::new();

        for entry in fs::read_dir(&resolved).map_err(|source| EnvironmentError::Io {
            path: path.to_owned(),
            source,
        })? {
            let entry = entry.map_err(|source| EnvironmentError::Io {
                path: path.to_owned(),
                source,
            })?;
            let file_type = entry.file_type().map_err(|source| EnvironmentError::Io {
                path: path.to_owned(),
                source,
            })?;
            entries.push(DirectoryEntry {
                path: self.relative_path(&entry.path()),
                kind: EntryKind::from_file_type(file_type),
            });
        }

        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(entries)
    }

    fn search_text(
        &self,
        query: &str,
        path: Option<&str>,
        max_results: usize,
    ) -> EnvironmentResult<SearchOutput> {
        let root_path = path.unwrap_or("");
        let start = self.resolve_existing(root_path)?;
        let limit = max_results.clamp(1, 200);
        let mut matches = Vec::new();
        let mut stack = vec![start];
        let mut truncated = false;

        while let Some(dir) = stack.pop() {
            let mut entries = fs::read_dir(&dir)
                .map_err(|source| EnvironmentError::Io {
                    path: root_path.to_owned(),
                    source,
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| EnvironmentError::Io {
                    path: root_path.to_owned(),
                    source,
                })?;
            entries.sort_by_key(|entry| entry.path());

            for entry in entries {
                let file_type = entry.file_type().map_err(|source| EnvironmentError::Io {
                    path: root_path.to_owned(),
                    source,
                })?;

                if file_type.is_symlink() {
                    continue;
                }

                if file_type.is_dir() {
                    stack.push(entry.path());
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }

                let Ok(content) = fs::read_to_string(entry.path()) else {
                    continue;
                };

                for (line_index, line) in content.lines().enumerate() {
                    if line.contains(query) {
                        matches.push(SearchMatch {
                            path: self.relative_path(&entry.path()),
                            line: line_index + 1,
                            text: line.to_owned(),
                        });
                        if matches.len() == limit {
                            truncated = true;
                            return Ok(SearchOutput { matches, truncated });
                        }
                    }
                }
            }
        }

        matches.sort_by(|left, right| {
            left.path
                .cmp(&right.path)
                .then_with(|| left.line.cmp(&right.line))
        });
        Ok(SearchOutput { matches, truncated })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub path: String,
    pub kind: EntryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    File,
    Directory,
    Symlink,
    Other,
}

impl EntryKind {
    fn from_file_type(file_type: fs::FileType) -> Self {
        if file_type.is_file() {
            Self::File
        } else if file_type.is_dir() {
            Self::Directory
        } else if file_type.is_symlink() {
            Self::Symlink
        } else {
            Self::Other
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteFileOutput {
    pub path: String,
    pub bytes_written: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchOutput {
    pub matches: Vec<SearchMatch>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchMatch {
    pub path: String,
    pub line: usize,
    pub text: String,
}

#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("environment configuration is invalid: {0}")]
    Configuration(String),
    #[error("path `{0}` is outside the configured environment root")]
    OutsideRoot(String),
    #[error("filesystem operation failed for `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl EnvironmentError {
    pub fn is_security_failure(&self) -> bool {
        matches!(self, Self::Configuration(_) | Self::OutsideRoot(_))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn local_environment_reads_writes_and_lists_inside_root() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("notes")).unwrap();
        let environment = LocalEnvironment::new(temp.path()).unwrap();

        let written = environment
            .write_text("notes/todo.txt", "ship the tiny filesystem")
            .unwrap();

        assert_eq!(written.bytes_written, 24);
        assert_eq!(
            environment.read_text("notes/todo.txt").unwrap(),
            "ship the tiny filesystem"
        );
        assert_eq!(
            environment.list("notes").unwrap(),
            vec![DirectoryEntry {
                path: "notes/todo.txt".to_owned(),
                kind: EntryKind::File,
            }]
        );
    }

    #[test]
    fn local_environment_rejects_parent_and_absolute_paths() {
        let temp = tempfile::tempdir().unwrap();
        let environment = LocalEnvironment::new(temp.path()).unwrap();

        assert!(matches!(
            environment.read_text("../secret.txt"),
            Err(EnvironmentError::OutsideRoot(_))
        ));
        assert!(matches!(
            environment.write_text("/tmp/secret.txt", "nope"),
            Err(EnvironmentError::OutsideRoot(_))
        ));
    }

    #[test]
    fn local_environment_searches_text_without_following_missing_files() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("src")).unwrap();
        fs::write(temp.path().join("src/lib.rs"), "alpha\nbeta alpha\n").unwrap();
        fs::write(temp.path().join("README.md"), "alpha docs\n").unwrap();
        let environment = LocalEnvironment::new(temp.path()).unwrap();

        let output = environment.search_text("alpha", None, 2).unwrap();

        assert_eq!(output.matches.len(), 2);
        assert!(output.truncated);
        assert!(output.matches.iter().any(|found| found.path == "README.md"));
    }

    #[cfg(unix)]
    #[test]
    fn local_environment_rejects_symlinks_that_escape_root() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "nope").unwrap();
        symlink(
            outside.path().join("secret.txt"),
            temp.path().join("secret-link"),
        )
        .unwrap();
        let environment = LocalEnvironment::new(temp.path()).unwrap();

        assert!(matches!(
            environment.read_text("secret-link"),
            Err(EnvironmentError::OutsideRoot(_))
        ));
    }
}
