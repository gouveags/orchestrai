use pyo3::prelude::*;

use crate::{
    Agent, AgentConfig, LoopError, create_agent,
    providers::{AnthropicProvider, OpenAiProvider},
};

#[pyclass(name = "Agent")]
pub struct PyAgent {
    agent: PyAgentInner,
}

enum PyAgentInner {
    Anthropic(Agent<AnthropicProvider>),
    OpenAi(Agent<OpenAiProvider>),
}

#[pymethods]
impl PyAgent {
    pub fn run(&self, prompt: &str) -> PyResult<String> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let input = prompt.to_owned();
        let output = match &self.agent {
            PyAgentInner::Anthropic(agent) => runtime.block_on(agent.run(input)),
            PyAgentInner::OpenAi(agent) => runtime.block_on(agent.run(input)),
        }
        .map_err(loop_error)?;
        Ok(output.final_message)
    }
}

#[pyfunction]
#[pyo3(name = "create_agent", signature = (model, api_key=None, instructions=None, max_tool_rounds=8, max_tokens=None, provider="openai"))]
pub fn create_agent_py(
    model: String,
    api_key: Option<String>,
    instructions: Option<String>,
    max_tool_rounds: usize,
    max_tokens: Option<u32>,
    provider: &str,
) -> PyResult<PyAgent> {
    let agent = match provider {
        "anthropic" => PyAgentInner::Anthropic(build_agent(
            AnthropicProvider::new(resolve_api_key(
                api_key,
                "ANTHROPIC_API_KEY",
                "api_key or ANTHROPIC_API_KEY is required",
            )?),
            model,
            instructions,
            max_tool_rounds,
            max_tokens,
        )),
        "openai" => PyAgentInner::OpenAi(build_agent(
            OpenAiProvider::new(resolve_api_key(
                api_key,
                "OPENAI_API_KEY",
                "api_key or OPENAI_API_KEY is required",
            )?),
            model,
            instructions,
            max_tool_rounds,
            max_tokens,
        )),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "provider must be 'anthropic' or 'openai'",
            ));
        }
    };

    Ok(PyAgent { agent })
}

#[pymodule]
pub fn orchestrai(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(create_agent_py, module)?)?;
    module.add_class::<PyAgent>()?;
    Ok(())
}

fn loop_error(error: LoopError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}

fn build_agent<P>(
    provider: P,
    model: String,
    instructions: Option<String>,
    max_tool_rounds: usize,
    max_tokens: Option<u32>,
) -> Agent<P>
where
    P: crate::ModelProvider,
{
    let mut config = AgentConfig::new(provider, model).with_max_tool_rounds(max_tool_rounds);
    if let Some(instructions) = instructions {
        config = config.with_instructions(instructions);
    }
    if let Some(max_tokens) = max_tokens {
        config = config.with_max_tokens(max_tokens);
    }
    create_agent(config)
}

fn resolve_api_key(
    api_key: Option<String>,
    env_var: &'static str,
    error_message: &'static str,
) -> PyResult<String> {
    api_key
        .or_else(|| std::env::var(env_var).ok())
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(error_message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_api_key_uses_explicit_key_first() {
        let key = resolve_api_key(
            Some("explicit-key".to_owned()),
            "ORCHESTRAI_MISSING_TEST_KEY",
            "missing",
        )
        .unwrap();

        assert_eq!(key, "explicit-key");
    }

    #[test]
    fn resolve_api_key_rejects_missing_key() {
        pyo3::prepare_freethreaded_python();

        let error = resolve_api_key(None, "ORCHESTRAI_MISSING_TEST_KEY", "missing").unwrap_err();

        assert_eq!(error.to_string(), "ValueError: missing");
    }
}
