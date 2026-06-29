use std::collections::HashMap;

use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{Value, json};

use crate::{
    provider::{
        ModelProvider, ModelRequest, ModelResponse, ModelStream, ModelStreamEvent, ProviderError,
        ProviderResult, error_for_status,
    },
    types::{Message, Role, ToolCall, Usage},
};

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    anthropic_version: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".to_owned(),
            anthropic_version: "2023-06-01".to_owned(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/messages", self.base_url)
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        let response = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .json(&anthropic_request(&request, false))
            .send()
            .await?;
        let body: Value = error_for_status(response).await?.json().await?;
        parse_anthropic_response(&body)
    }

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
        let response = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .json(&anthropic_request(&request, true))
            .send()
            .await?;
        let response = error_for_status(response).await?;

        Ok(Box::pin(try_stream! {
            let mut buffer = String::new();
            let mut chunks = response.bytes_stream();
            let mut content_to_tool_index = HashMap::<usize, usize>::new();
            let mut next_tool_index = 0usize;

            while let Some(chunk) = chunks.next().await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(boundary) = buffer.find("\n\n") {
                    let frame = buffer[..boundary].to_owned();
                    buffer = buffer[boundary + 2..].to_owned();

                    for event in parse_anthropic_sse_frame(
                        &frame,
                        &mut content_to_tool_index,
                        &mut next_tool_index,
                    )? {
                        yield event;
                    }
                }
            }
        }))
    }
}

fn anthropic_request(request: &ModelRequest, stream: bool) -> Value {
    let (system, messages) = anthropic_messages(&request.messages);
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens.unwrap_or(4096),
        "stream": stream,
    });

    if !system.is_empty() {
        body["system"] = json!(system);
    }

    if !request.tools.is_empty() {
        body["tools"] = json!(
            request
                .tools
                .iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }))
                .collect::<Vec<_>>()
        );
    }

    body
}

fn anthropic_messages(messages: &[Message]) -> (String, Vec<Value>) {
    let mut system = Vec::new();
    let mut output = Vec::new();

    for message in messages {
        match message.role {
            Role::System => system.push(message.content.clone()),
            Role::User => output.push(json!({"role": "user", "content": message.content})),
            Role::Assistant => {
                let mut content = Vec::new();
                if !message.content.is_empty() {
                    content.push(json!({"type": "text", "text": message.content}));
                }
                for call in &message.tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }));
                }
                output.push(json!({"role": "assistant", "content": content}));
            }
            Role::Tool => output.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id.as_deref().unwrap_or_default(),
                    "content": message.content,
                }],
            })),
        }
    }

    (system.join("\n\n"), output)
}

fn parse_anthropic_response(body: &Value) -> ProviderResult<ModelResponse> {
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or(ProviderError::MissingField("content"))?;
    let mut message = String::new();
    let mut tool_calls = Vec::new();

    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => message.push_str(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
            Some("tool_use") => tool_calls.push(ToolCall {
                id: block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or(ProviderError::MissingField("content[].id"))?
                    .to_owned(),
                name: block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or(ProviderError::MissingField("content[].name"))?
                    .to_owned(),
                arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
            }),
            _ => {}
        }
    }

    Ok(ModelResponse {
        message,
        tool_calls,
        usage: body.get("usage").map(|usage| Usage {
            input_tokens: usage.get("input_tokens").and_then(Value::as_u64),
            output_tokens: usage.get("output_tokens").and_then(Value::as_u64),
        }),
    })
}

fn parse_anthropic_sse_frame(
    frame: &str,
    content_to_tool_index: &mut HashMap<usize, usize>,
    next_tool_index: &mut usize,
) -> ProviderResult<Vec<ModelStreamEvent>> {
    let mut events = Vec::new();

    for line in frame.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let body: Value = serde_json::from_str(data).map_err(|error| {
            ProviderError::Parse(format!("invalid Anthropic SSE frame: {error}"))
        })?;
        let event_type = body.get("type").and_then(Value::as_str);

        match event_type {
            Some("content_block_start") => {
                let index = body.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let block = body.get("content_block").unwrap_or(&Value::Null);
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    let tool_index = *next_tool_index;
                    *next_tool_index += 1;
                    content_to_tool_index.insert(index, tool_index);
                    events.push(ModelStreamEvent::ToolCallDelta {
                        index: tool_index,
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                        arguments_delta: String::new(),
                    });
                }
            }
            Some("content_block_delta") => {
                let index = body.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let delta = body.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            events.push(ModelStreamEvent::MessageDelta(text.to_owned()));
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(tool_index) = content_to_tool_index.get(&index) {
                            events.push(ModelStreamEvent::ToolCallDelta {
                                index: *tool_index,
                                id: None,
                                name: None,
                                arguments_delta: delta
                                    .get("partial_json")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_owned(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(usage) = body.get("usage") {
                    events.push(ModelStreamEvent::Usage(Usage {
                        input_tokens: usage.get("input_tokens").and_then(Value::as_u64),
                        output_tokens: usage.get("output_tokens").and_then(Value::as_u64),
                    }));
                }
            }
            Some("message_stop") => events.push(ModelStreamEvent::Done),
            _ => {}
        }
    }

    Ok(events)
}
