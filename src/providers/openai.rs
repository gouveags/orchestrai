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

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_owned(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_owned();
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        let response = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .json(&openai_request(&request, false))
            .send()
            .await?;
        let body: Value = error_for_status(response).await?.json().await?;
        parse_openai_response(&body)
    }

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
        let response = self
            .client
            .post(self.endpoint())
            .bearer_auth(&self.api_key)
            .json(&openai_request(&request, true))
            .send()
            .await?;
        let response = error_for_status(response).await?;

        Ok(Box::pin(try_stream! {
            let mut buffer = String::new();
            let mut chunks = response.bytes_stream();

            while let Some(chunk) = chunks.next().await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(boundary) = buffer.find("\n\n") {
                    let frame = buffer[..boundary].to_owned();
                    buffer = buffer[boundary + 2..].to_owned();

                    for event in parse_openai_sse_frame(&frame)? {
                        yield event;
                    }
                }
            }
        }))
    }
}

fn openai_request(request: &ModelRequest, stream: bool) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": request.messages.iter().map(openai_message).collect::<Vec<_>>(),
        "stream": stream,
    });

    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }

    if !request.tools.is_empty() {
        body["tools"] = json!(
            request
                .tools
                .iter()
                .map(|tool| json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                }))
                .collect::<Vec<_>>()
        );
        body["tool_choice"] = json!("auto");
    }

    body
}

fn openai_message(message: &Message) -> Value {
    match message.role {
        Role::System => json!({"role": "system", "content": message.content}),
        Role::User => json!({"role": "user", "content": message.content}),
        Role::Assistant => {
            let mut value = json!({"role": "assistant", "content": message.content});
            if !message.tool_calls.is_empty() {
                value["tool_calls"] = json!(
                    message
                        .tool_calls
                        .iter()
                        .map(|call| json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments.to_string(),
                            }
                        }))
                        .collect::<Vec<_>>()
                );
            }
            value
        }
        Role::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id.as_deref().unwrap_or_default(),
            "content": message.content,
        }),
    }
}

fn parse_openai_response(body: &Value) -> ProviderResult<ModelResponse> {
    let message = body
        .pointer("/choices/0/message")
        .ok_or(ProviderError::MissingField("choices[0].message"))?;

    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().map(parse_openai_tool_call).collect())
        .transpose()?
        .unwrap_or_default();

    Ok(ModelResponse {
        message: content,
        tool_calls,
        usage: parse_openai_usage(body.get("usage")),
    })
}

fn parse_openai_tool_call(call: &Value) -> ProviderResult<ToolCall> {
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .ok_or(ProviderError::MissingField("tool_calls[].id"))?;
    let function = call
        .get("function")
        .ok_or(ProviderError::MissingField("tool_calls[].function"))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or(ProviderError::MissingField("tool_calls[].function.name"))?;
    let arguments = function
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let arguments = serde_json::from_str(arguments)
        .map_err(|error| ProviderError::Parse(format!("invalid tool arguments: {error}")))?;

    Ok(ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments,
    })
}

fn parse_openai_usage(usage: Option<&Value>) -> Option<Usage> {
    usage.map(|usage| Usage {
        input_tokens: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_tokens: usage.get("completion_tokens").and_then(Value::as_u64),
    })
}

fn parse_openai_sse_frame(frame: &str) -> ProviderResult<Vec<ModelStreamEvent>> {
    let mut events = Vec::new();

    for line in frame.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };

        if data == "[DONE]" {
            events.push(ModelStreamEvent::Done);
            continue;
        }

        let body: Value = serde_json::from_str(data)
            .map_err(|error| ProviderError::Parse(format!("invalid OpenAI SSE frame: {error}")))?;
        let Some(delta) = body.pointer("/choices/0/delta") else {
            continue;
        };

        if let Some(content) = delta.get("content").and_then(Value::as_str) {
            events.push(ModelStreamEvent::MessageDelta(content.to_owned()));
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in tool_calls {
                let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let function = call.get("function");
                events.push(ModelStreamEvent::ToolCallDelta {
                    index,
                    id: call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    name: function
                        .and_then(|value| value.get("name"))
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    arguments_delta: function
                        .and_then(|value| value.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            }
        }

        if let Some(usage) = parse_openai_usage(body.get("usage")) {
            events.push(ModelStreamEvent::Usage(usage));
        }
    }

    Ok(events)
}
