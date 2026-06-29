use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use crate::tool::{Tool, ToolRegistry};

#[derive(Clone, Default)]
pub struct CapabilityBundleSet {
    defaults: Vec<CapabilityBundle>,
    bundles: BTreeMap<String, CapabilityBundle>,
}

impl CapabilityBundleSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default(mut self, bundle: CapabilityBundle) -> Self {
        self.defaults.push(bundle);
        self
    }

    pub fn with_bundle(mut self, name: impl Into<String>, bundle: CapabilityBundle) -> Self {
        self.bundles.insert(name.into(), bundle);
        self
    }

    pub(crate) fn resolve(
        &self,
        selection: &CapabilitySelection,
        base_tools: &ToolRegistry,
    ) -> Result<ResolvedCapabilities, CapabilityError> {
        let mut tools = base_tools.clone();
        let mut seen_tools = tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<HashSet<_>>();
        let mut prompts = Vec::new();

        for bundle in &self.defaults {
            add_bundle(bundle, &mut prompts, &mut tools, &mut seen_tools)?;
        }

        for name in &selection.bundle_names {
            let bundle = self
                .bundles
                .get(name)
                .ok_or_else(|| CapabilityError::UnknownBundle { name: name.clone() })?;
            add_bundle(bundle, &mut prompts, &mut tools, &mut seen_tools)?;
        }

        Ok(ResolvedCapabilities { prompts, tools })
    }
}

#[derive(Clone)]
pub struct CapabilityBundle {
    name: String,
    prompts: Vec<String>,
    tools: Vec<Arc<dyn Tool>>,
}

impl CapabilityBundle {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            prompts: Vec::new(),
            tools: Vec::new(),
        }
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompts.push(prompt.into());
        self
    }

    pub fn with_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySelection {
    bundle_names: Vec<String>,
}

impl CapabilitySelection {
    pub fn new<I, S>(bundle_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            bundle_names: bundle_names.into_iter().map(Into::into).collect(),
        }
    }
}

pub(crate) struct ResolvedCapabilities {
    pub prompts: Vec<String>,
    pub tools: ToolRegistry,
}

#[derive(Debug, thiserror::Error)]
pub enum CapabilityError {
    #[error("capability bundle `{name}` is not configured")]
    UnknownBundle { name: String },
    #[error("tool `{name}` is provided by more than one active capability")]
    DuplicateTool { name: String },
}

fn add_bundle(
    bundle: &CapabilityBundle,
    prompts: &mut Vec<String>,
    tools: &mut ToolRegistry,
    seen_tools: &mut HashSet<String>,
) -> Result<(), CapabilityError> {
    prompts.extend(bundle.prompts.iter().cloned());

    for tool in &bundle.tools {
        let name = tool.definition().name;
        if !seen_tools.insert(name.clone()) {
            return Err(CapabilityError::DuplicateTool { name });
        }
        tools.register_arc(Arc::clone(tool));
    }

    Ok(())
}
