use std::{collections::HashMap, sync::Arc};

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::StatusCode;

use crate::provider::{
    ModelProvider, ModelRequest, ModelResponse, ModelStream, ProviderError, ProviderResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderModel {
    pub provider: String,
    pub model: String,
}

impl ProviderModel {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn ModelProvider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_provider<P>(mut self, name: impl Into<String>, provider: Arc<P>) -> Self
    where
        P: ModelProvider + 'static,
    {
        self.providers.insert(name.into(), provider);
        self
    }

    fn provider(&self, name: &str) -> Option<Arc<dyn ModelProvider>> {
        self.providers.get(name).cloned()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelCatalog {
    aliases: HashMap<String, Vec<ProviderModel>>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_alias(
        mut self,
        alias: impl Into<String>,
        models: impl IntoIterator<Item = ProviderModel>,
    ) -> Self {
        self.aliases
            .insert(alias.into(), models.into_iter().collect());
        self
    }

    fn models_for(&self, alias: &str) -> Option<&[ProviderModel]> {
        self.aliases.get(alias).map(Vec::as_slice)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FallbackPolicy {
    retry_transient_provider_errors: bool,
}

impl FallbackPolicy {
    pub fn disabled() -> Self {
        Self {
            retry_transient_provider_errors: false,
        }
    }

    pub fn transient_provider_errors() -> Self {
        Self {
            retry_transient_provider_errors: true,
        }
    }

    fn should_try_next(&self, error: &ProviderError) -> bool {
        self.retry_transient_provider_errors && is_transient_provider_error(error)
    }
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

pub struct RoutedModelProvider {
    registry: ProviderRegistry,
    catalog: ModelCatalog,
    fallback_policy: FallbackPolicy,
}

impl RoutedModelProvider {
    pub fn new(registry: ProviderRegistry, catalog: ModelCatalog) -> Self {
        Self {
            registry,
            catalog,
            fallback_policy: FallbackPolicy::default(),
        }
    }

    pub fn with_fallback_policy(mut self, fallback_policy: FallbackPolicy) -> Self {
        self.fallback_policy = fallback_policy;
        self
    }

    fn route_for(&self, alias: &str) -> ProviderResult<&[ProviderModel]> {
        match self.catalog.models_for(alias) {
            Some([]) => Err(ProviderError::Config(format!(
                "model alias `{alias}` has no provider models"
            ))),
            Some(models) => Ok(models),
            None => Err(ProviderError::Config(format!(
                "model alias `{alias}` is not configured"
            ))),
        }
    }

    fn provider(&self, provider_name: &str) -> ProviderResult<Arc<dyn ModelProvider>> {
        self.registry.provider(provider_name).ok_or_else(|| {
            ProviderError::Config(format!("provider `{provider_name}` is not registered"))
        })
    }
}

#[async_trait]
impl ModelProvider for RoutedModelProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        let route = self.route_for(&request.model)?;
        let last_index = route.len() - 1;

        for (index, provider_model) in route.iter().enumerate() {
            let provider = self.provider(&provider_model.provider)?;
            let routed_request = request_for_model(&request, &provider_model.model);

            match provider.complete(routed_request).await {
                Ok(response) => return Ok(response),
                Err(error)
                    if self.fallback_policy.should_try_next(&error) && index < last_index =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }

        unreachable!("non-empty routes return from the loop")
    }

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
        let route = self.route_for(&request.model)?;
        let route = route.to_vec();
        let registry = self.registry.clone();
        let last_index = route.len() - 1;
        let fallback_policy = self.fallback_policy;

        Ok(Box::pin(try_stream! {
            for (index, provider_model) in route.iter().enumerate() {
                let provider = registry.provider(&provider_model.provider).ok_or_else(|| {
                    ProviderError::Config(format!(
                        "provider `{}` is not registered",
                        provider_model.provider
                    ))
                })?;
                let routed_request = request_for_model(&request, &provider_model.model);
                let mut stream = match provider.stream(routed_request).await {
                    Ok(stream) => stream,
                    Err(error)
                        if fallback_policy.should_try_next(&error) && index < last_index =>
                    {
                        continue;
                    }
                    Err(error) => Err(error)?,
                };
                let mut emitted = false;
                let mut should_try_next = false;

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(event) => {
                            emitted = true;
                            yield event;
                        }
                        Err(error)
                            if !emitted && fallback_policy.should_try_next(&error) && index < last_index =>
                        {
                            should_try_next = true;
                            break;
                        }
                        Err(error) => Err(error)?,
                    }
                }

                if should_try_next {
                    continue;
                }
                break;
            }
        }))
    }
}

fn request_for_model(request: &ModelRequest, model: &str) -> ModelRequest {
    let mut routed = request.clone();
    routed.model = model.to_owned();
    routed
}

fn is_transient_provider_error(error: &ProviderError) -> bool {
    match error {
        ProviderError::Http(error) => error.is_timeout() || error.is_connect(),
        ProviderError::Status { status, .. } => is_transient_status(*status),
        ProviderError::MissingField(_)
        | ProviderError::Parse(_)
        | ProviderError::Config(_)
        | ProviderError::UnsupportedFeature(_) => false,
    }
}

fn is_transient_status(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}
