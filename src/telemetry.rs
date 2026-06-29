use std::sync::Arc;

use crate::types::Usage;

#[derive(Clone, Default)]
pub struct TelemetryConfig {
    sink: Option<Arc<dyn TelemetrySink>>,
}

impl TelemetryConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sink<T>(mut self, sink: T) -> Self
    where
        T: TelemetrySink + 'static,
    {
        self.sink = Some(Arc::new(sink));
        self
    }

    pub(crate) fn record(&self, event: TelemetryEvent) {
        if let Some(sink) = &self.sink {
            sink.record(event);
        }
    }
}

pub trait TelemetrySink: Send + Sync {
    fn record(&self, event: TelemetryEvent);
}

#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryEvent {
    RunStarted,
    RunFinished {
        success: bool,
    },
    ModelCallStarted {
        model: String,
    },
    ModelCallFinished {
        model: String,
        usage: Option<Usage>,
    },
    ToolCallStarted {
        tool_call_id: String,
        name: String,
    },
    ToolCallFinished {
        tool_call_id: String,
        name: String,
        is_error: bool,
        recoverable: bool,
    },
}
