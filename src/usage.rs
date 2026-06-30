use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::types::Usage;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageSnapshot {
    pub runs: u64,
    /// Logical orchestrai model requests made by the loop. One call is one
    /// `ModelProvider::complete` or `ModelProvider::stream` invocation, not a
    /// provider adapter's internal retry or fallback attempt.
    pub model_calls: u64,
    pub tool_calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl UsageSnapshot {
    pub fn total_tokens(self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub(crate) fn add_assign(&mut self, other: UsageSnapshot) {
        self.runs += other.runs;
        self.model_calls += other.model_calls;
        self.tool_calls += other.tool_calls;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

#[derive(Debug, Clone, Default)]
pub struct UsageMeter {
    inner: Arc<Mutex<UsageSnapshot>>,
}

impl UsageMeter {
    pub fn from_snapshot(snapshot: UsageSnapshot) -> Self {
        Self {
            inner: Arc::new(Mutex::new(snapshot)),
        }
    }

    pub fn snapshot(&self) -> UsageSnapshot {
        *self.inner.lock().unwrap()
    }

    pub(crate) fn reserve_run(
        &self,
        limits: &UsageLimits,
    ) -> Result<UsageSnapshot, UsageLimitError> {
        let mut snapshot = self.inner.lock().unwrap();
        limits.check_run_start(*snapshot)?;

        let delta = UsageSnapshot {
            runs: 1,
            ..UsageSnapshot::default()
        };
        snapshot.add_assign(delta);
        Ok(delta)
    }

    pub(crate) fn reserve_model_call(
        &self,
        limits: &UsageLimits,
    ) -> Result<UsageSnapshot, UsageLimitError> {
        let mut snapshot = self.inner.lock().unwrap();
        limits.check_model_call_start(*snapshot)?;

        let delta = UsageSnapshot {
            model_calls: 1,
            ..UsageSnapshot::default()
        };
        snapshot.add_assign(delta);
        Ok(delta)
    }

    pub(crate) fn reserve_tool_call(
        &self,
        limits: &UsageLimits,
    ) -> Result<UsageSnapshot, UsageLimitError> {
        let mut snapshot = self.inner.lock().unwrap();
        limits.check_tool_call_start(*snapshot)?;

        let delta = UsageSnapshot {
            tool_calls: 1,
            ..UsageSnapshot::default()
        };
        snapshot.add_assign(delta);
        Ok(delta)
    }

    pub(crate) fn record_model_usage(&self, usage: Option<&Usage>) -> UsageSnapshot {
        let delta = UsageSnapshot {
            input_tokens: usage.and_then(|usage| usage.input_tokens).unwrap_or(0),
            output_tokens: usage.and_then(|usage| usage.output_tokens).unwrap_or(0),
            ..UsageSnapshot::default()
        };
        self.record(delta);
        delta
    }

    fn record(&self, delta: UsageSnapshot) {
        self.inner.lock().unwrap().add_assign(delta);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageLimits {
    pub max_runs: Option<u64>,
    pub max_model_calls: Option<u64>,
    pub max_tool_calls: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub max_total_tokens: Option<u64>,
}

impl UsageLimits {
    pub fn with_max_runs(mut self, max_runs: u64) -> Self {
        self.max_runs = Some(max_runs);
        self
    }

    pub fn with_max_model_calls(mut self, max_model_calls: u64) -> Self {
        self.max_model_calls = Some(max_model_calls);
        self
    }

    pub fn with_max_tool_calls(mut self, max_tool_calls: u64) -> Self {
        self.max_tool_calls = Some(max_tool_calls);
        self
    }

    pub fn with_max_input_tokens(mut self, max_input_tokens: u64) -> Self {
        self.max_input_tokens = Some(max_input_tokens);
        self
    }

    pub fn with_max_output_tokens(mut self, max_output_tokens: u64) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn with_max_total_tokens(mut self, max_total_tokens: u64) -> Self {
        self.max_total_tokens = Some(max_total_tokens);
        self
    }

    fn check_run_start(&self, snapshot: UsageSnapshot) -> Result<(), UsageLimitError> {
        self.check_limit(UsageLimitKind::Runs, snapshot.runs, self.max_runs)?;
        self.check_token_limits(snapshot)
    }

    fn check_model_call_start(&self, snapshot: UsageSnapshot) -> Result<(), UsageLimitError> {
        self.check_limit(
            UsageLimitKind::ModelCalls,
            snapshot.model_calls,
            self.max_model_calls,
        )?;
        self.check_token_limits(snapshot)
    }

    fn check_tool_call_start(&self, snapshot: UsageSnapshot) -> Result<(), UsageLimitError> {
        self.check_limit(
            UsageLimitKind::ToolCalls,
            snapshot.tool_calls,
            self.max_tool_calls,
        )?;
        self.check_token_limits(snapshot)
    }

    fn check_token_limits(&self, snapshot: UsageSnapshot) -> Result<(), UsageLimitError> {
        self.check_limit(
            UsageLimitKind::InputTokens,
            snapshot.input_tokens,
            self.max_input_tokens,
        )?;
        self.check_limit(
            UsageLimitKind::OutputTokens,
            snapshot.output_tokens,
            self.max_output_tokens,
        )?;
        self.check_limit(
            UsageLimitKind::TotalTokens,
            snapshot.total_tokens(),
            self.max_total_tokens,
        )
    }

    fn check_limit(
        &self,
        kind: UsageLimitKind,
        current: u64,
        limit: Option<u64>,
    ) -> Result<(), UsageLimitError> {
        let Some(limit) = limit else {
            return Ok(());
        };
        if current >= limit {
            Err(UsageLimitError {
                kind,
                limit,
                current,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageLimitKind {
    Runs,
    ModelCalls,
    ToolCalls,
    InputTokens,
    OutputTokens,
    TotalTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UsageLimitError {
    pub kind: UsageLimitKind,
    pub limit: u64,
    pub current: u64,
}
