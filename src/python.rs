use pyo3::prelude::*;

use crate::{
    Agent, AgentConfig, CapabilityBundle, CapabilityBundleSet, CapabilitySelection, FallbackPolicy,
    FnTool, InMemoryRunStore, LocalArtifactStore, LocalEnvironment, LoopError, LoopEvent,
    LoopOutput, ModelCatalog, PlanToolSet, ProviderModel, ProviderRegistry, RoutedModelProvider,
    RunOptions, RunState, StateInstructionPolicy, ToolDefinition, ToolError, ToolRegistry,
    UsageLimits, UsageMeter, UsageSnapshot, create_agent,
    providers::{AnthropicProvider, AwsCredentials, BedrockProvider, OpenAiProvider},
    register_artifact_tools, register_filesystem_tools,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

#[pyclass(name = "Agent")]
pub struct PyAgent {
    agent: PyAgentInner,
    usage_meter: UsageMeter,
    run_store: Option<InMemoryRunStore>,
}

enum PyAgentInner {
    Anthropic(Agent<AnthropicProvider>),
    Bedrock(Agent<BedrockProvider>),
    OpenAi(Agent<OpenAiProvider>),
    Routed(Agent<RoutedModelProvider>),
}

impl PyAgent {
    fn new(agent: PyAgentInner, runtime: PyAgentRuntime) -> Self {
        Self {
            agent,
            usage_meter: runtime.usage_meter,
            run_store: runtime.run_store,
        }
    }
}

#[derive(Clone)]
struct PyAgentRuntime {
    usage_meter: UsageMeter,
    usage_limits: UsageLimits,
    run_store: Option<InMemoryRunStore>,
}

impl PyAgentRuntime {
    fn new(usage_limits: Option<BTreeMap<String, u64>>, track_runs: bool) -> PyResult<Self> {
        Ok(Self {
            usage_meter: UsageMeter::default(),
            usage_limits: build_usage_limits(usage_limits)?,
            run_store: track_runs.then(InMemoryRunStore::new),
        })
    }

    fn apply<P>(&self, mut config: AgentConfig<P>) -> AgentConfig<P> {
        config = config
            .with_usage_meter(self.usage_meter.clone())
            .with_usage_limits(self.usage_limits);
        if let Some(run_store) = &self.run_store {
            config = config.with_run_store(run_store.clone());
        }
        config
    }
}

#[pyclass(name = "RunResult")]
#[derive(Clone)]
pub struct PyRunResult {
    value: Value,
}

#[pymethods]
impl PyRunResult {
    #[getter]
    pub fn run_id(&self) -> PyResult<String> {
        self.string_field("run_id")
    }

    #[getter]
    pub fn final_message(&self) -> PyResult<String> {
        self.string_field("final_message")
    }

    #[getter]
    pub fn text(&self) -> PyResult<String> {
        self.final_message()
    }

    #[getter]
    pub fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value_field(py, "usage")
    }

    #[getter]
    pub fn tool_results(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value_field(py, "tool_results")
    }

    #[getter]
    pub fn messages(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value_field(py, "messages")
    }

    #[getter]
    pub fn injected_summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value_field(py, "injected_summary")
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_value_to_py(py, self.value.clone())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "RunResult(run_id={:?}, text={:?})",
            self.run_id()?,
            self.text()?
        ))
    }
}

impl PyRunResult {
    fn from_output(output: LoopOutput) -> PyResult<Self> {
        Ok(Self {
            value: loop_output_to_value(output)?,
        })
    }

    fn string_field(&self, field: &str) -> PyResult<String> {
        self.value
            .get(field)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(format!("missing {field}")))
    }

    fn value_field(&self, py: Python<'_>, field: &str) -> PyResult<Py<PyAny>> {
        json_value_to_py(py, self.value.get(field).cloned().unwrap_or(Value::Null))
    }
}

impl PyAgent {
    fn run_result(
        &self,
        py: Python<'_>,
        prompt: &str,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<PyRunResult> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let options = run_options_from_py(py, prompt, state, model_mode)?;
        let selection = CapabilitySelection::new(capabilities.unwrap_or_default());
        let output = match &self.agent {
            PyAgentInner::Anthropic(agent) => {
                runtime.block_on(agent.run_with_options_and_capabilities(options, selection))
            }
            PyAgentInner::Bedrock(agent) => {
                runtime.block_on(agent.run_with_options_and_capabilities(options, selection))
            }
            PyAgentInner::OpenAi(agent) => {
                runtime.block_on(agent.run_with_options_and_capabilities(options, selection))
            }
            PyAgentInner::Routed(agent) => {
                runtime.block_on(agent.run_with_options_and_capabilities(options, selection))
            }
        }
        .map_err(loop_error)?;

        PyRunResult::from_output(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn stream_result(
        &self,
        py: Python<'_>,
        prompt: &str,
        on_delta: Option<Py<PyAny>>,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
        on_event: Option<Py<PyAny>>,
    ) -> PyResult<PyRunResult> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        let callback_error = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
        let error_slot = std::sync::Arc::clone(&callback_error);
        let options = run_options_from_py(py, prompt, state, model_mode)?;
        let selection = CapabilitySelection::new(capabilities.unwrap_or_default());
        let delta_callback = on_delta.map(Arc::new);
        let event_callback = on_event.map(Arc::new);
        let output = match &self.agent {
            PyAgentInner::Anthropic(agent) => runtime.block_on(
                agent.run_stream_with_options_and_capabilities(options, selection, move |event| {
                    let delta_callback = delta_callback.clone();
                    let event_callback = event_callback.clone();
                    let error_slot = std::sync::Arc::clone(&error_slot);
                    async move {
                        call_python_stream_callbacks(
                            delta_callback,
                            event_callback,
                            error_slot,
                            event,
                        );
                    }
                }),
            ),
            PyAgentInner::Bedrock(agent) => runtime.block_on(
                agent.run_stream_with_options_and_capabilities(options, selection, move |event| {
                    let delta_callback = delta_callback.clone();
                    let event_callback = event_callback.clone();
                    let error_slot = std::sync::Arc::clone(&error_slot);
                    async move {
                        call_python_stream_callbacks(
                            delta_callback,
                            event_callback,
                            error_slot,
                            event,
                        );
                    }
                }),
            ),
            PyAgentInner::OpenAi(agent) => runtime.block_on(
                agent.run_stream_with_options_and_capabilities(options, selection, move |event| {
                    let delta_callback = delta_callback.clone();
                    let event_callback = event_callback.clone();
                    let error_slot = std::sync::Arc::clone(&error_slot);
                    async move {
                        call_python_stream_callbacks(
                            delta_callback,
                            event_callback,
                            error_slot,
                            event,
                        );
                    }
                }),
            ),
            PyAgentInner::Routed(agent) => runtime.block_on(
                agent.run_stream_with_options_and_capabilities(options, selection, move |event| {
                    let delta_callback = delta_callback.clone();
                    let event_callback = event_callback.clone();
                    let error_slot = std::sync::Arc::clone(&error_slot);
                    async move {
                        call_python_stream_callbacks(
                            delta_callback,
                            event_callback,
                            error_slot,
                            event,
                        );
                    }
                }),
            ),
        }
        .map_err(loop_error)?;

        if let Some(error) = callback_error.lock().unwrap().take() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(error));
        }

        PyRunResult::from_output(output)
    }
}

#[pymethods]
impl PyAgent {
    #[pyo3(signature = (prompt, state=None, capabilities=None, model_mode=None))]
    pub fn run(
        &self,
        py: Python<'_>,
        prompt: &str,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<String> {
        self.run_result(py, prompt, state, capabilities, model_mode)?
            .text()
    }

    #[pyo3(name = "run_text", signature = (prompt, state=None, capabilities=None, model_mode=None))]
    pub fn run_text(
        &self,
        py: Python<'_>,
        prompt: &str,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<String> {
        self.run(py, prompt, state, capabilities, model_mode)
    }

    #[pyo3(name = "ask", signature = (prompt, state=None, capabilities=None, model_mode=None))]
    pub fn ask(
        &self,
        py: Python<'_>,
        prompt: &str,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<String> {
        self.run_text(py, prompt, state, capabilities, model_mode)
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
        self.run_result(py, prompt, state, capabilities, model_mode)?
            .to_dict(py)
    }

    #[pyo3(name = "run_full", signature = (prompt, state=None, capabilities=None, model_mode=None))]
    pub fn run_full(
        &self,
        py: Python<'_>,
        prompt: &str,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
    ) -> PyResult<PyRunResult> {
        self.run_result(py, prompt, state, capabilities, model_mode)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (prompt, on_delta=None, state=None, capabilities=None, model_mode=None, on_event=None))]
    pub fn run_stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        on_delta: Option<Py<PyAny>>,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
        on_event: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.stream_result(
            py,
            prompt,
            on_delta,
            state,
            capabilities,
            model_mode,
            on_event,
        )?
        .to_dict(py)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "stream", signature = (prompt, on_delta=None, state=None, capabilities=None, model_mode=None, on_event=None))]
    pub fn stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        on_delta: Option<Py<PyAny>>,
        state: Option<Py<PyAny>>,
        capabilities: Option<Vec<String>>,
        model_mode: Option<String>,
        on_event: Option<Py<PyAny>>,
    ) -> PyResult<PyRunResult> {
        self.stream_result(
            py,
            prompt,
            on_delta,
            state,
            capabilities,
            model_mode,
            on_event,
        )
    }

    pub fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        usage_to_py(py, self.usage_meter.snapshot())
    }

    pub fn run_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let events = self
            .run_store
            .as_ref()
            .map(InMemoryRunStore::events)
            .unwrap_or_default();
        let value = serde_json::to_value(events)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        json_value_to_py(py, value)
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

    #[pyo3(name = "register", signature = (name, description, handler, input_schema=None))]
    pub fn register(
        &mut self,
        py: Python<'_>,
        name: String,
        description: String,
        handler: Py<PyAny>,
        input_schema: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.register_tool(py, name, description, handler, input_schema)
    }

    pub fn register_planning_tools(&mut self) {
        PlanToolSet::new().register(&mut self.registry);
    }

    #[pyo3(name = "planning")]
    pub fn planning(&mut self) {
        self.register_planning_tools();
    }

    pub fn register_filesystem_tools(&mut self, root: String) -> PyResult<()> {
        let environment = LocalEnvironment::new(root)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        register_filesystem_tools(&mut self.registry, std::sync::Arc::new(environment));
        Ok(())
    }

    #[pyo3(name = "filesystem")]
    pub fn filesystem(&mut self, root: String) -> PyResult<()> {
        self.register_filesystem_tools(root)
    }

    pub fn register_artifact_tools(&mut self, root: String) -> PyResult<()> {
        let store = LocalArtifactStore::new(root)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        register_artifact_tools(&mut self.registry, std::sync::Arc::new(store));
        Ok(())
    }

    #[pyo3(name = "artifacts")]
    pub fn artifacts(&mut self, root: String) -> PyResult<()> {
        self.register_artifact_tools(root)
    }

    pub fn definitions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(self.registry.definitions())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        json_value_to_py(py, value)
    }
}

#[pyfunction]
#[pyo3(name = "tools")]
pub fn tools_py() -> PyToolRegistry {
    PyToolRegistry::new()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "create_agent", signature = (model, api_key=None, instructions=None, max_tool_rounds=8, max_tokens=None, provider="openai", tools=None, default_prompts=None, capability_prompts=None, state_keys=None, model_modes=None, usage_limits=None, track_runs=false, providers=None, models=None, fallback="disabled", prompts=None, bundles=None))]
pub fn create_agent_py(
    py: Python<'_>,
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
    usage_limits: Option<BTreeMap<String, u64>>,
    track_runs: bool,
    providers: Option<Py<PyAny>>,
    models: Option<Py<PyAny>>,
    fallback: &str,
    prompts: Option<Vec<String>>,
    bundles: Option<BTreeMap<String, Vec<String>>>,
) -> PyResult<PyAgent> {
    let default_prompts = merge_prompt_lists(default_prompts, prompts);
    let capability_prompts = merge_prompt_maps(capability_prompts, bundles);
    let telemetry = PyAgentRuntime::new(usage_limits, track_runs)?;

    if providers.is_some() || models.is_some() {
        let provider = build_routed_provider(py, providers, models, api_key, fallback)?;
        let agent = PyAgentInner::Routed(build_agent(
            provider,
            model,
            instructions,
            max_tool_rounds,
            max_tokens,
            tools,
            default_prompts,
            capability_prompts,
            state_keys,
            model_modes,
            telemetry.clone(),
        ));
        return Ok(PyAgent::new(agent, telemetry));
    }

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
            telemetry.clone(),
        )),
        "bedrock" => PyAgentInner::Bedrock(build_agent(
            BedrockProvider::from_env(std::env::var("AWS_REGION").unwrap_or_else(|_| {
                std::env::var("AWS_DEFAULT_REGION").unwrap_or_else(|_| "us-east-1".to_owned())
            }))
            .map_err(provider_config_error)?,
            model,
            instructions,
            max_tool_rounds,
            max_tokens,
            tools,
            default_prompts,
            capability_prompts,
            state_keys,
            model_modes,
            telemetry.clone(),
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
            telemetry.clone(),
        )),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "provider must be 'anthropic', 'bedrock', or 'openai'",
            ));
        }
    };

    Ok(PyAgent::new(agent, telemetry))
}

#[pymodule]
pub fn orchestrai(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(create_agent_py, module)?)?;
    module.add_function(wrap_pyfunction!(tools_py, module)?)?;
    module.add_class::<PyAgent>()?;
    module.add_class::<PyRunResult>()?;
    module.add_class::<PyToolRegistry>()?;
    Ok(())
}

fn loop_error(error: LoopError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}

fn provider_config_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

fn build_usage_limits(limits: Option<BTreeMap<String, u64>>) -> PyResult<UsageLimits> {
    let mut output = UsageLimits::default();
    for (key, value) in limits.unwrap_or_default() {
        output = match key.as_str() {
            "max_runs" => output.with_max_runs(value),
            "max_model_calls" => output.with_max_model_calls(value),
            "max_tool_calls" => output.with_max_tool_calls(value),
            "max_input_tokens" => output.with_max_input_tokens(value),
            "max_output_tokens" => output.with_max_output_tokens(value),
            "max_total_tokens" => output.with_max_total_tokens(value),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown usage limit `{key}`"
                )));
            }
        };
    }
    Ok(output)
}

fn build_routed_provider(
    py: Python<'_>,
    providers: Option<Py<PyAny>>,
    models: Option<Py<PyAny>>,
    api_key: Option<String>,
    fallback: &str,
) -> PyResult<RoutedModelProvider> {
    let models = models.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "models= is required when configuring routed providers",
        )
    })?;
    let (catalog, provider_names) = parse_model_catalog(py_to_json_value(py, models)?)?;
    let provider_configs = match providers {
        Some(providers) => py_to_json_value(py, providers)?,
        None => json!({}),
    };
    let provider_configs = provider_configs.as_object().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("providers= must be a dictionary")
    })?;
    for (provider_name, config) in provider_configs {
        if !provider_names.contains(provider_name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "providers['{provider_name}'] is not referenced by models="
            )));
        }
        if !config.is_object() && !config.is_string() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "providers['{provider_name}'] must be a dictionary or API key string"
            )));
        }
        if provider_name == "bedrock" && config.is_string() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "providers['bedrock'] must be a dictionary with AWS credential fields or omitted to use AWS environment credentials",
            ));
        }
    }
    let only_provider = (provider_names.len() == 1).then(|| {
        provider_names
            .iter()
            .next()
            .expect("len checked above")
            .to_owned()
    });

    let mut registry = ProviderRegistry::new();
    for provider_name in provider_names {
        let config = provider_configs.get(&provider_name).unwrap_or(&Value::Null);
        registry = match provider_name.as_str() {
            "anthropic" => registry.with_provider(
                provider_name,
                Arc::new(AnthropicProvider::new(resolve_provider_api_key(
                    config,
                    api_key.as_deref(),
                    only_provider.as_deref(),
                    "anthropic",
                    "ANTHROPIC_API_KEY",
                )?)),
            ),
            "bedrock" => {
                let region = config
                    .get("region")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| std::env::var("AWS_REGION").ok())
                    .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
                    .unwrap_or_else(|| "us-east-1".to_owned());
                let provider = if config_has_aws_credentials(config) {
                    BedrockProvider::new(region, aws_credentials_from_config(config)?)
                } else {
                    BedrockProvider::from_env(region).map_err(provider_config_error)?
                };
                registry.with_provider(provider_name, Arc::new(provider))
            }
            "openai" => registry.with_provider(
                provider_name,
                Arc::new(OpenAiProvider::new(resolve_provider_api_key(
                    config,
                    api_key.as_deref(),
                    only_provider.as_deref(),
                    "openai",
                    "OPENAI_API_KEY",
                )?)),
            ),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported provider `{provider_name}`"
                )));
            }
        };
    }

    let fallback_policy = match fallback {
        "disabled" | "none" => FallbackPolicy::disabled(),
        "transient" | "transient_provider_errors" => FallbackPolicy::transient_provider_errors(),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fallback must be 'disabled' or 'transient'",
            ));
        }
    };

    Ok(RoutedModelProvider::new(registry, catalog).with_fallback_policy(fallback_policy))
}

fn parse_model_catalog(value: Value) -> PyResult<(ModelCatalog, BTreeSet<String>)> {
    let object = value
        .as_object()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("models= must be a dictionary"))?;
    if object.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "models= must define at least one alias",
        ));
    }
    let mut catalog = ModelCatalog::new();
    let mut provider_names = BTreeSet::new();

    for (alias, route) in object {
        let route = route.as_array().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "models['{alias}'] must be a list of provider model entries"
            ))
        })?;
        if route.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "models['{alias}'] must include at least one provider model"
            )));
        }
        let mut provider_models = Vec::new();
        for entry in route {
            let provider_model = parse_provider_model(alias, entry)?;
            provider_names.insert(provider_model.provider.clone());
            provider_models.push(provider_model);
        }
        catalog = catalog.with_alias(alias, provider_models);
    }

    Ok((catalog, provider_names))
}

fn parse_provider_model(alias: &str, entry: &Value) -> PyResult<ProviderModel> {
    if let Some(object) = entry.as_object() {
        let provider = object
            .get("provider")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "models['{alias}'] entry is missing provider"
                ))
            })?;
        let model = object.get("model").and_then(Value::as_str).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "models['{alias}'] entry is missing model"
            ))
        })?;
        return Ok(ProviderModel::new(provider, model));
    }

    if let Some(array) = entry.as_array()
        && array.len() == 2
        && let (Some(provider), Some(model)) = (array[0].as_str(), array[1].as_str())
    {
        return Ok(ProviderModel::new(provider, model));
    }

    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "models['{alias}'] entries must be {{'provider': ..., 'model': ...}} or (provider, model)"
    )))
}

fn resolve_provider_api_key(
    config: &Value,
    top_level_api_key: Option<&str>,
    only_provider: Option<&str>,
    provider: &str,
    env_var: &'static str,
) -> PyResult<String> {
    if let Some(api_key) = config.as_str() {
        return Ok(api_key.to_owned());
    }

    config
        .get("api_key")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            (only_provider == Some(provider))
                .then(|| top_level_api_key.map(ToOwned::to_owned))
                .flatten()
        })
        .or_else(|| std::env::var(env_var).ok())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "providers['{provider}']['api_key'] or {env_var} is required"
            ))
        })
}

fn config_has_aws_credentials(config: &Value) -> bool {
    config.get("access_key_id").is_some() || config.get("secret_access_key").is_some()
}

fn aws_credentials_from_config(config: &Value) -> PyResult<AwsCredentials> {
    let access_key_id = config
        .get("access_key_id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("bedrock access_key_id is required")
        })?;
    let secret_access_key = config
        .get("secret_access_key")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("bedrock secret_access_key is required")
        })?;
    let session_token = config
        .get("session_token")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    Ok(AwsCredentials::new(
        access_key_id,
        secret_access_key,
        session_token,
    ))
}

fn merge_prompt_lists(
    primary: Option<Vec<String>>,
    alias: Option<Vec<String>>,
) -> Option<Vec<String>> {
    match (primary, alias) {
        (None, None) => None,
        (Some(prompts), None) | (None, Some(prompts)) => Some(prompts),
        (Some(mut primary), Some(alias)) => {
            primary.extend(alias);
            Some(primary)
        }
    }
}

fn merge_prompt_maps(
    primary: Option<BTreeMap<String, Vec<String>>>,
    alias: Option<BTreeMap<String, Vec<String>>>,
) -> Option<BTreeMap<String, Vec<String>>> {
    match (primary, alias) {
        (None, None) => None,
        (Some(map), None) | (None, Some(map)) => Some(map),
        (Some(mut primary), Some(alias)) => {
            for (key, prompts) in alias {
                primary.entry(key).or_default().extend(prompts);
            }
            Some(primary)
        }
    }
}

#[allow(clippy::too_many_arguments)]
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
    runtime: PyAgentRuntime,
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
    config = runtime.apply(config);
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

fn call_python_stream_callbacks(
    delta_callback: Option<Arc<Py<PyAny>>>,
    event_callback: Option<Arc<Py<PyAny>>>,
    error_slot: Arc<Mutex<Option<String>>>,
    event: LoopEvent,
) {
    if let Some(callback) = event_callback {
        let event_value = loop_event_to_value(&event);
        if let Err(error) =
            Python::with_gil(|py| callback.call1(py, (json_value_to_py(py, event_value)?,)))
        {
            *error_slot.lock().unwrap() = Some(error.to_string());
            return;
        }
    }

    if let (Some(callback), LoopEvent::MessageDelta(delta)) = (delta_callback, event)
        && let Err(error) = Python::with_gil(|py| callback.call1(py, (delta,)))
    {
        *error_slot.lock().unwrap() = Some(error.to_string());
    }
}

fn loop_event_to_value(event: &LoopEvent) -> Value {
    match event {
        LoopEvent::MessageDelta(delta) => {
            json!({"type": "message_delta", "delta": delta})
        }
        LoopEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments_delta,
        } => json!({
            "type": "tool_call_delta",
            "index": index,
            "id": id,
            "name": name,
            "arguments_delta": arguments_delta,
        }),
        LoopEvent::ToolStarted { tool_call_id, name } => json!({
            "type": "tool_started",
            "tool_call_id": tool_call_id,
            "name": name,
        }),
        LoopEvent::ToolFinished {
            tool_call_id,
            name,
            content,
        } => json!({
            "type": "tool_finished",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }),
        LoopEvent::Usage {
            input_tokens,
            output_tokens,
        } => json!({
            "type": "usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }),
        LoopEvent::Done => json!({"type": "done"}),
    }
}

fn usage_to_value(usage: UsageSnapshot) -> Value {
    json!({
        "runs": usage.runs,
        "model_calls": usage.model_calls,
        "tool_calls": usage.tool_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens(),
    })
}

fn usage_to_py(py: Python<'_>, usage: UsageSnapshot) -> PyResult<Py<PyAny>> {
    json_value_to_py(py, usage_to_value(usage))
}

fn loop_output_to_value(output: LoopOutput) -> PyResult<Value> {
    let usage = output.usage_snapshot();
    let final_message = output.final_message;
    Ok(json!({
        "run_id": output.run_id.as_str(),
        "final_message": final_message.clone(),
        "text": final_message,
        "messages": output.messages,
        "tool_results": output.tool_results,
        "injected_summary": output.injected_summary,
        "usage": usage_to_value(usage),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ModelRequest, ModelResponse, ModelStream, ModelStreamEvent, ProviderError, ProviderResult,
        Usage,
    };
    use async_trait::async_trait;
    use futures_util::stream;
    use std::{collections::VecDeque, sync::Mutex};

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
            let module = PyModule::from_code(
                py,
                code,
                pyo3::ffi::c_str!("test.py"),
                pyo3::ffi::c_str!("test"),
            )
            .unwrap();
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

    #[test]
    fn python_tool_registry_exposes_ergonomic_aliases_and_definitions() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let code = pyo3::ffi::c_str!("def handle(arguments):\n    return 'ok'\n");
            let module = PyModule::from_code(
                py,
                code,
                pyo3::ffi::c_str!("test.py"),
                pyo3::ffi::c_str!("test"),
            )
            .unwrap();
            let mut registry = PyToolRegistry::new();

            registry
                .register(
                    py,
                    "echo".to_owned(),
                    "Echo test.".to_owned(),
                    module.getattr("handle").unwrap().unbind(),
                    None,
                )
                .unwrap();
            registry.planning();

            let definitions = py_to_json_value(py, registry.definitions(py).unwrap()).unwrap();
            let definitions = definitions.as_array().unwrap();
            let names = definitions
                .iter()
                .filter_map(|definition| definition.get("name").and_then(Value::as_str))
                .collect::<Vec<_>>();

            assert!(names.contains(&"echo"));
            assert!(names.contains(&crate::PLAN_CREATE_TOOL));
        });
    }

    #[test]
    fn python_usage_limits_accept_only_known_keys() {
        let limits = build_usage_limits(Some(BTreeMap::from([
            ("max_runs".to_owned(), 5),
            ("max_total_tokens".to_owned(), 100),
        ])))
        .unwrap();
        assert_eq!(limits.max_runs, Some(5));
        assert_eq!(limits.max_total_tokens, Some(100));

        let error =
            build_usage_limits(Some(BTreeMap::from([("surprise".to_owned(), 1)]))).unwrap_err();
        assert!(error.to_string().contains("unknown usage limit"));
    }

    #[test]
    fn python_model_catalog_accepts_simple_dict_routes() {
        let (_catalog, providers) = parse_model_catalog(json!({
            "regular": [
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                ["openai", "gpt-4.1-mini"]
            ]
        }))
        .unwrap();

        assert_eq!(
            providers,
            BTreeSet::from(["anthropic".to_owned(), "openai".to_owned()])
        );
    }

    #[test]
    fn python_routed_provider_rejects_unused_provider_configs_and_invalid_fallback() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let models =
                json_value_to_py(py, json!({"regular": [["openai", "gpt-4.1-mini"]]})).unwrap();
            let providers =
                json_value_to_py(py, json!({"openai": "test-key", "open_ai": "typo-key"})).unwrap();
            let error =
                match build_routed_provider(py, Some(providers), Some(models), None, "disabled") {
                    Ok(_) => panic!("unused provider config should fail"),
                    Err(error) => error,
                };
            assert!(error.to_string().contains("not referenced by models"));

            let models =
                json_value_to_py(py, json!({"regular": [["openai", "gpt-4.1-mini"]]})).unwrap();
            let providers = json_value_to_py(py, json!({"openai": "test-key"})).unwrap();
            let error =
                match build_routed_provider(py, Some(providers), Some(models), None, "surprise") {
                    Ok(_) => panic!("invalid fallback should fail"),
                    Err(error) => error,
                };
            assert!(error.to_string().contains("fallback must be"));

            let models = json_value_to_py(
                py,
                json!({"regular": [["bedrock", "anthropic.claude-sonnet-4-6"]]}),
            )
            .unwrap();
            let providers = json_value_to_py(py, json!({"bedrock": "not-an-api-key"})).unwrap();
            let error =
                match build_routed_provider(py, Some(providers), Some(models), None, "disabled") {
                    Ok(_) => panic!("bedrock string config should fail"),
                    Err(error) => error,
                };
            assert!(error.to_string().contains("providers['bedrock'] must be"));
        });
    }

    #[test]
    fn python_routed_provider_rejects_empty_model_routes() {
        let error = parse_model_catalog(json!({"regular": []})).unwrap_err();
        assert!(error.to_string().contains("at least one provider model"));
    }

    #[test]
    fn python_build_agent_translates_prompts_state_capabilities_and_model_modes() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let code = pyo3::ffi::c_str!("def handle(arguments):\n    return {'ok': True}\n");
            let module = PyModule::from_code(
                py,
                code,
                pyo3::ffi::c_str!("test.py"),
                pyo3::ffi::c_str!("test"),
            )
            .unwrap();
            let mut registry = PyToolRegistry::new();
            registry
                .register(
                    py,
                    "lookup".to_owned(),
                    "Lookup test.".to_owned(),
                    module.getattr("handle").unwrap().unbind(),
                    None,
                )
                .unwrap();

            let provider = RecordingProvider::new([ModelResponse {
                message: "ready".to_owned(),
                tool_calls: Vec::new(),
                usage: Some(Usage {
                    input_tokens: Some(8),
                    output_tokens: Some(3),
                }),
            }]);
            let requests = Arc::clone(&provider.requests);
            let agent = build_agent(
                provider,
                "logical".to_owned(),
                Some("Instruction sentinel.".to_owned()),
                2,
                Some(42),
                Some(registry),
                Some(vec!["Base prompt sentinel.".to_owned()]),
                Some(BTreeMap::from([(
                    "data".to_owned(),
                    vec!["Capability prompt sentinel.".to_owned()],
                )])),
                Some(vec!["tenant".to_owned()]),
                Some(BTreeMap::from([(
                    "fast".to_owned(),
                    "real-model".to_owned(),
                )])),
                PyAgentRuntime::new(
                    Some(BTreeMap::from([("max_total_tokens".to_owned(), 100)])),
                    true,
                )
                .unwrap(),
            );

            let options = RunOptions::new("Hello")
                .with_state(
                    RunState::from_json(json!({
                        "tenant": "Acme",
                        "private_runtime_secret": "not rendered"
                    }))
                    .unwrap(),
                )
                .with_model_mode("fast");
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let output =
                runtime
                    .block_on(agent.run_with_options_and_capabilities(
                        options,
                        CapabilitySelection::new(["data"]),
                    ))
                    .unwrap();

            assert_eq!(output.final_message, "ready");
            assert_eq!(
                output.usage_snapshot(),
                UsageSnapshot {
                    runs: 1,
                    model_calls: 1,
                    tool_calls: 0,
                    input_tokens: 8,
                    output_tokens: 3,
                }
            );

            let requests = requests.lock().unwrap();
            let request = requests.first().unwrap();
            assert_eq!(request.model, "real-model");
            assert_eq!(request.max_tokens, Some(42));
            assert_eq!(request.tools[0].name, "lookup");
            let rendered = request
                .messages
                .iter()
                .map(|message| message.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(rendered.contains("Instruction sentinel."));
            assert!(rendered.contains("Base prompt sentinel."));
            assert!(rendered.contains("Capability prompt sentinel."));
            assert!(rendered.contains("tenant"));
            assert!(rendered.contains("Acme"));
            assert!(!rendered.contains("private_runtime_secret"));
            assert!(!rendered.contains("not rendered"));
        });
    }

    #[test]
    fn python_run_result_is_easy_to_use_and_convert_to_dict() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let provider = RecordingProvider::new([ModelResponse {
                message: "hello from result".to_owned(),
                tool_calls: Vec::new(),
                usage: Some(Usage {
                    input_tokens: Some(2),
                    output_tokens: Some(4),
                }),
            }]);
            let agent = build_agent(
                provider,
                "fake".to_owned(),
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                PyAgentRuntime::new(None, true).unwrap(),
            );
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let output = runtime.block_on(agent.run("Hello")).unwrap();
            let result = PyRunResult::from_output(output).unwrap();
            assert_eq!(result.text().unwrap(), "hello from result");
            let dict = py_to_json_value(py, result.to_dict(py).unwrap()).unwrap();
            assert_eq!(dict["final_message"], "hello from result");
            assert_eq!(dict["text"], "hello from result");
            assert_eq!(dict["usage"]["total_tokens"], 6);
            assert!(dict["messages"].is_array());
        });
    }

    #[derive(Clone)]
    struct RecordingProvider {
        responses: Arc<Mutex<VecDeque<ModelResponse>>>,
        requests: Arc<Mutex<Vec<ModelRequest>>>,
    }

    impl RecordingProvider {
        fn new<const N: usize>(responses: [ModelResponse; N]) -> Self {
            Self {
                responses: Arc::new(Mutex::new(VecDeque::from(responses))),
                requests: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl crate::ModelProvider for RecordingProvider {
        async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
            self.requests.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| ProviderError::Config("missing fake response".to_owned()))
        }

        async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
            self.requests.lock().unwrap().push(request);
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| ProviderError::Config("missing fake response".to_owned()))?;
            Ok(Box::pin(stream::iter([
                Ok(ModelStreamEvent::MessageDelta(response.message)),
                Ok(ModelStreamEvent::Done),
            ])))
        }
    }
}
