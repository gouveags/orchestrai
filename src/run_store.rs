use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::usage::UsageSnapshot;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RunId(String);

impl RunId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Started,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunEvent {
    RunStarted {
        run_id: RunId,
    },
    ModelCallStarted {
        run_id: RunId,
        model: String,
    },
    ModelCallFinished {
        run_id: RunId,
        model: String,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    ToolCallStarted {
        run_id: RunId,
        tool_call_id: String,
        name: String,
    },
    ToolCallFinished {
        run_id: RunId,
        tool_call_id: String,
        name: String,
        is_error: bool,
        recoverable: bool,
    },
    RunFinished {
        run_id: RunId,
        status: RunStatus,
        usage: UsageSnapshot,
    },
}

pub trait RunStore: Send + Sync {
    fn start_run(&self) -> RunId;

    fn record(&self, event: RunEvent);
}

#[derive(Clone, Default)]
pub struct NoopRunStore;

impl RunStore for NoopRunStore {
    fn start_run(&self) -> RunId {
        RunId::new(new_run_id())
    }

    fn record(&self, _event: RunEvent) {}
}

#[derive(Clone, Default)]
pub struct InMemoryRunStore {
    events: Arc<Mutex<Vec<RunEvent>>>,
}

impl InMemoryRunStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<RunEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl RunStore for InMemoryRunStore {
    fn start_run(&self) -> RunId {
        RunId::new(new_run_id())
    }

    fn record(&self, event: RunEvent) {
        self.events.lock().unwrap().push(event);
    }
}

fn new_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("run_{nanos}")
}
