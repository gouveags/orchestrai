use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunState {
    entries: Map<String, Value>,
}

impl RunState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json(value: Value) -> Result<Self, RunStateError> {
        match value {
            Value::Object(entries) => Ok(Self { entries }),
            _ => Err(RunStateError::ExpectedObject),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    pub fn get_string(&self, key: &str) -> Option<String> {
        self.entries
            .get(key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        self.entries.insert(key.into(), value)
    }

    pub fn rendered_instructions(&self, policy: &StateInstructionPolicy) -> Option<String> {
        let lines = policy
            .keys
            .iter()
            .filter_map(|key| {
                self.entries
                    .get(key)
                    .map(|value| format!("- {key}: {}", render_value(value)))
            })
            .collect::<Vec<_>>();

        if lines.is_empty() {
            None
        } else {
            Some(format!("Run state:\n{}", lines.join("\n")))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateInstructionPolicy {
    keys: Vec<String>,
}

impl StateInstructionPolicy {
    pub fn selected<I, K>(keys: I) -> Self
    where
        I: IntoIterator<Item = K>,
        K: Into<String>,
    {
        Self {
            keys: keys.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCallConfig {
    pub model: String,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeforeModelCall {
    pub state: RunState,
    pub config: ModelCallConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunOptions {
    pub input: String,
    pub state: RunState,
    pub model_mode: Option<String>,
}

impl RunOptions {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            state: RunState::new(),
            model_mode: None,
        }
    }

    pub fn with_state(mut self, state: RunState) -> Self {
        self.state = state;
        self
    }

    pub fn with_model_mode(mut self, model_mode: impl Into<String>) -> Self {
        self.model_mode = Some(model_mode.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunStateError {
    #[error("run state must be a JSON object")]
    ExpectedObject,
}

fn render_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value).unwrap_or_default(),
    }
}
