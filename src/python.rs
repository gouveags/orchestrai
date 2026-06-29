use pyo3::prelude::*;

use crate::{Agent, AgentConfig, LoopError, create_agent, providers::OpenAiProvider};

#[pyclass(name = "Agent")]
pub struct PyAgent {
    agent: Agent<OpenAiProvider>,
}

#[pymethods]
impl PyAgent {
    pub fn run(&self, prompt: &str) -> PyResult<String> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let output = runtime
            .block_on(self.agent.run(prompt.to_owned()))
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
    if provider != "openai" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "only provider='openai' is currently supported by the Python bindings",
        ));
    }

    let api_key = api_key
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("api_key or OPENAI_API_KEY is required")
        })?;

    let mut config =
        AgentConfig::new(OpenAiProvider::new(api_key), model).with_max_tool_rounds(max_tool_rounds);
    if let Some(instructions) = instructions {
        config = config.with_instructions(instructions);
    }
    if let Some(max_tokens) = max_tokens {
        config = config.with_max_tokens(max_tokens);
    }

    Ok(PyAgent {
        agent: create_agent(config),
    })
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
