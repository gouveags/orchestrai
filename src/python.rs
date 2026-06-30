use pyo3::prelude::*;

use crate::{
    Agent, AgentConfig, CapabilityBundle, CapabilityBundleSet, CapabilitySelection, FnTool,
    LocalArtifactStore, LocalEnvironment, LoopError, LoopOutput, PlanToolSet, RunOptions, RunState,
    StateInstructionPolicy, ToolDefinition, ToolError, ToolRegistry, create_agent,
    providers::{AnthropicProvider, OpenAiProvider},
    register_artifact_tools, register_filesystem_tools,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};

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

    #[pyo3(signature = (prompt, state=None, capabilities=None, model_mode=None))]
    pub fn run_output(
        &self,
        py: Python<'_>,
        prompt: &str,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let options = run_options_from_py(py, prompt, state, model_mode)?;
        let selection = CapabilitySelection::new(capabilities.unwrap_or_default());
        let output = match &self.agent {
            PyAgentInner::Anthropic(agent) => {
                runtime.block_on(agent.run_with_options_and_capabilities(options, selection))
            }
            PyAgentInner::OpenAi(agent) => {
                runtime.block_on(agent.run_with_options_and_capabilities(options, selection))
            }
        }
        .map_err(loop_error)?;

        loop_output_to_py(py, output)
    }

    #[pyo3(signature = (prompt, on_delta=None, state=None, capabilities=None, model_mode=None))]
    pub fn run_stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        on_delta: Option<Py<PyAny>>,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let callback_error = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let error_slot = std::sync::Arc::clone(&callback_error);
        let options = run_options_from_py(py, prompt, state, model_mode)?;
        let selection = CapabilitySelection::new(capabilities.unwrap_or_default());
        let callback = on_delta.map(Arc::new);
        let output = match &self.agent {
            PyAgentInner::Anthropic(agent) => runtime.block_on(
                agent.run_stream_with_options_and_capabilities(options, selection, move |event| {
                    let callback = callback.clone();
                    let error_slot = std::sync::Arc::clone(&error_slot);
                    async move {
                        if let (Some(callback), crate::LoopEvent::MessageDelta(delta)) =
                            (callback, event)
                        {
                            if let Err(error) = Python::with_gil(|py| callback.call1(py, (delta,)))
                            {
                                *error_slot.lock().unwrap() = Some(error.to_string());
                            }
                        }
                    }
                }),
            ),
            PyAgentInner::OpenAi(agent) => runtime.block_on(
                agent.run_stream_with_options_and_capabilities(options, selection, move |event| {
                    let callback = callback.clone();
                    let error_slot = std::sync::Arc::clone(&error_slot);
                    async move {
                        if let (Some(callback), crate::LoopEvent::MessageDelta(delta)) =
                            (callback, event)
                        {
                            if let Err(error) = Python::with_gil(|py| callback.call1(py, (delta,)))
                            {
                                *error_slot.lock().unwrap() = Some(error.to_string());
                            }
                        }
                    }
                }),
            ),
        }
        .map_err(loop_error)?;

        if let Some(error) = callback_error.lock().unwrap().take() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(error));
        }

        loop_output_to_py(py, output)
    }
}

#[pyclass(name = "ToolRegistry")]
#[derive(Clone, Default)]
pub struct PyToolRegistry {
    registry: ToolRegistry,
}

#[pymethods]
impl PyToolRegistry {
    #[new]
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::new(),
        }
    }

    #[pyo3(signature = (name, description, handler, input_schema=None))]
    pub fn register_tool(
        &mut self,
        py: Python<'_>,
        name: String,
        description: String,
        handler: Py<PyAny>,
        input_schema: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let schema = match input_schema {
            Some(schema) => py_to_json_value(py, schema)?,
            None => json!({"type": "object"}),
        };
        let handler = Arc::new(handler);
        self.registry.register(FnTool::new(
            ToolDefinition::new(name.clone(), description, schema),
            move |arguments| {
                let handler = Arc::clone(&handler);
                let name = name.clone();
                Box::pin(async move {
                    call_python_tool(&handler, arguments).map_err(|error| {
                        ToolError::result_content(format!(
                            "python tool `{name}` returned an error: {error}"
                        ))
                    })
                })
            },
        ));
        Ok(())
    }

    pub fn register_planning_tools(&mut self) {
        PlanToolSet::new().register(&mut self.registry);
    }

    pub fn register_filesystem_tools(&mut self, root: String) -> PyResult<()> {
        let environment = LocalEnvironment::new(root)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        register_filesystem_tools(&mut self.registry, std::sync::Arc::new(environment));
        Ok(())
    }

    pub fn register_artifact_tools(&mut self, root: String) -> PyResult<()> {
        let store = LocalArtifactStore::new(root)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        register_artifact_tools(&mut self.registry, std::sync::Arc::new(store));
        Ok(())
    }
}

#[pyfunction]
#[pyo3(name = "create_agent", signature = (model, api_key=None, instructions=None, max_tool_rounds=8, max_tokens=None, provider="openai", tools=None, default_prompts=None, capability_prompts=None, state_keys=None, model_modes=None))]
pub fn create_agent_py(
    model: String,
    api_key: Option<String>,
    instructions: Option<String>,
    max_tool_rounds: usize,
    max_tokens: Option<u32>,
    provider: &str,
    tools: Option<PyToolRegistry>,
    default_prompts: Option<Vec<String>>,
    capability_prompts: Option<BTreeMap<String, Vec<String>>>,
    state_keys: Option<Vec<String>>,
    model_modes: Option<BTreeMap<String, String>>,
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
            tools,
            default_prompts,
            capability_prompts,
            state_keys,
            model_modes,
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
            tools,
            default_prompts,
            capability_prompts,
            state_keys,
            model_modes,
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
    module.add_class::<PyToolRegistry>()?;
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
    tools: Option<PyToolRegistry>,
    default_prompts: Option<Vec<String>>,
    capability_prompts: Option<BTreeMap<String, Vec<String>>>,
    state_keys: Option<Vec<String>>,
    model_modes: Option<BTreeMap<String, String>>,
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
    if let Some(tools) = tools {
        config = config.with_tools(tools.registry);
    }
    if let Some(capability_bundles) = build_capability_bundles(default_prompts, capability_prompts)
    {
        config = config.with_capability_bundles(capability_bundles);
    }
    if let Some(keys) = state_keys {
        config = config.with_run_state_instructions(StateInstructionPolicy::selected(keys));
    }
    if let Some(model_modes) = model_modes {
        config = config.with_model_modes(model_modes);
    }
    create_agent(config)
}

fn build_capability_bundles(
    default_prompts: Option<Vec<String>>,
    capability_prompts: Option<BTreeMap<String, Vec<String>>>,
) -> Option<CapabilityBundleSet> {
    if default_prompts.is_none() && capability_prompts.is_none() {
        return None;
    }

    let mut bundles = CapabilityBundleSet::new();
    for (index, prompt) in default_prompts.unwrap_or_default().into_iter().enumerate() {
        bundles = bundles.with_default(
            CapabilityBundle::new(format!("python_default_{index}")).with_prompt(prompt),
        );
    }
    for (name, prompts) in capability_prompts.unwrap_or_default() {
        let mut bundle = CapabilityBundle::new(name.clone());
        for prompt in prompts {
            bundle = bundle.with_prompt(prompt);
        }
        bundles = bundles.with_bundle(name, bundle);
    }

    Some(bundles)
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

fn run_options_from_py(
    py: Python<'_>,
    prompt: &str,
    state: Option<Py<PyAny>>,
    model_mode: Option<String>,
) -> PyResult<RunOptions> {
    let mut options = RunOptions::new(prompt);
    if let Some(state) = state {
        options = options.with_state(
            RunState::from_json(py_to_json_value(py, state)?)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?,
        );
    }
    if let Some(model_mode) = model_mode {
        options = options.with_model_mode(model_mode);
    }
    Ok(options)
}

fn py_to_json_value(py: Python<'_>, value: Py<PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let text: String = json.call_method1("dumps", (value,))?.extract()?;
    serde_json::from_str(&text)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn json_value_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    let json = py.import("json")?;
    Ok(json.call_method1("loads", (value.to_string(),))?.unbind())
}

fn call_python_tool(handler: &Py<PyAny>, arguments: Value) -> Result<String, PyErr> {
    Python::with_gil(|py| {
        let json = py.import("json")?;
        let arguments = json.call_method1("loads", (arguments.to_string(),))?;
        let result = handler.call1(py, (arguments,))?;
        if let Ok(text) = result.extract::<String>(py) {
            return Ok(text);
        }
        json.call_method1("dumps", (result,))?.extract()
    })
}

fn loop_output_to_py(py: Python<'_>, output: LoopOutput) -> PyResult<Py<PyAny>> {
    let usage = output.usage_snapshot();
    json_value_to_py(
        py,
        json!({
            "run_id": output.run_id.as_str(),
            "final_message": output.final_message,
            "tool_results": output.tool_results,
            "usage": {
                "runs": usage.runs,
                "model_calls": usage.model_calls,
                "tool_calls": usage.tool_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens(),
            }
        }),
    )
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

    #[test]
    fn python_tool_registry_executes_python_callable_with_json_arguments() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let code = pyo3::ffi::c_str!(
                "def handle(arguments):\n    return {'echo': arguments['value']}\n"
            );
            let module =
                PyModule::from_code_bound(py, code.to_str().unwrap(), "test.py", "test").unwrap();
            let handler = module.getattr("handle").unwrap().unbind();
            let mut registry = PyToolRegistry::new();
            registry
                .register_tool(
                    py,
                    "echo".to_owned(),
                    "Echo test.".to_owned(),
                    handler,
                    None,
                )
                .unwrap();

            let runtime = tokio::runtime::Runtime::new().unwrap();
            let output = runtime
                .block_on(registry.registry.execute("echo", json!({"value": "ok"})))
                .unwrap();

            assert_eq!(output, r#"{"echo": "ok"}"#);
        });
    }
}
