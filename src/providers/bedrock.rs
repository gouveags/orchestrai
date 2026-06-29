use std::collections::{BTreeMap, HashMap};

use async_stream::try_stream;
use async_trait::async_trait;
use bytes::{Buf, Bytes};
use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use time::{OffsetDateTime, macros::format_description};

use crate::{
    provider::{
        ModelProvider, ModelRequest, ModelResponse, ModelStream, ModelStreamEvent, ProviderError,
        ProviderResult, error_for_status,
    },
    types::{Message, Role, ToolCall, Usage},
};

type HmacSha256 = Hmac<Sha256>;

pub struct BedrockProvider {
    client: Client,
    region: String,
    credentials: AwsCredentials,
}

impl BedrockProvider {
    pub fn new(region: impl Into<String>, credentials: AwsCredentials) -> Self {
        Self {
            client: Client::new(),
            region: region.into(),
            credentials,
        }
    }

    pub fn from_env(region: impl Into<String>) -> ProviderResult<Self> {
        Ok(Self::new(region, AwsCredentials::from_env()?))
    }

    fn endpoint(&self, model: &str, operation: &str) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/{}",
            self.region,
            urlencoding::encode(model),
            operation
        )
    }
}

#[async_trait]
impl ModelProvider for BedrockProvider {
    async fn complete(&self, request: ModelRequest) -> ProviderResult<ModelResponse> {
        let body = bedrock_request(&request).to_string();
        let url = self.endpoint(&request.model, "converse");
        let headers = sign_bedrock_request(&self.region, &self.credentials, &url, &body)?;
        let response = self
            .client
            .post(url)
            .headers(headers)
            .body(body)
            .send()
            .await?;
        let body: Value = error_for_status(response).await?.json().await?;
        parse_bedrock_response(&body)
    }

    async fn stream(&self, request: ModelRequest) -> ProviderResult<ModelStream> {
        let body = bedrock_request(&request).to_string();
        let url = self.endpoint(&request.model, "converse-stream");
        let headers = sign_bedrock_request(&self.region, &self.credentials, &url, &body)?;
        let response = self
            .client
            .post(url)
            .headers(headers)
            .body(body)
            .send()
            .await?;
        let response = error_for_status(response).await?;

        Ok(Box::pin(try_stream! {
            let mut buffer = Vec::<u8>::new();
            let mut chunks = response.bytes_stream();
            let mut content_to_tool_index = HashMap::<usize, usize>::new();
            let mut next_tool_index = 0usize;

            while let Some(chunk) = chunks.next().await {
                let chunk = chunk?;
                buffer.extend_from_slice(&chunk);

                while let Some(message) = take_eventstream_message(&mut buffer)? {
                    if let Some(payload) = message.payload_json()? {
                        for event in parse_bedrock_stream_payload(
                            &payload,
                            &mut content_to_tool_index,
                            &mut next_tool_index,
                        )? {
                            yield event;
                        }
                    }
                }
            }
        }))
    }
}

#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl AwsCredentials {
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
        }
    }

    pub fn from_env() -> ProviderResult<Self> {
        let access_key_id = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| ProviderError::Config("AWS_ACCESS_KEY_ID is required".to_owned()))?;
        let secret_access_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| ProviderError::Config("AWS_SECRET_ACCESS_KEY is required".to_owned()))?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        Ok(Self::new(access_key_id, secret_access_key, session_token))
    }
}

fn bedrock_request(request: &ModelRequest) -> Value {
    let (system, messages) = bedrock_messages(&request.messages);
    let mut body = json!({
        "messages": messages,
    });

    if !system.is_empty() {
        body["system"] = json!(
            system
                .into_iter()
                .map(|text| json!({"text": text}))
                .collect::<Vec<_>>()
        );
    }

    if let Some(max_tokens) = request.max_tokens {
        body["inferenceConfig"] = json!({"maxTokens": max_tokens});
    }

    if !request.tools.is_empty() {
        body["toolConfig"] = json!({
            "tools": request.tools.iter().map(|tool| json!({
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": tool.input_schema,
                    },
                }
            })).collect::<Vec<_>>()
        });
    }

    body
}

fn bedrock_messages(messages: &[Message]) -> (Vec<String>, Vec<Value>) {
    let mut system = Vec::new();
    let mut output = Vec::new();

    for message in messages {
        match message.role {
            Role::System => system.push(message.content.clone()),
            Role::User => output.push(json!({
                "role": "user",
                "content": [{"text": message.content}],
            })),
            Role::Assistant => {
                let mut content = Vec::new();
                if !message.content.is_empty() {
                    content.push(json!({"text": message.content}));
                }
                for call in &message.tool_calls {
                    content.push(json!({
                        "toolUse": {
                            "toolUseId": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    }));
                }
                output.push(json!({"role": "assistant", "content": content}));
            }
            Role::Tool => output.push(json!({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": message.tool_call_id.as_deref().unwrap_or_default(),
                        "content": [{"text": message.content}],
                    }
                }],
            })),
        }
    }

    (system, output)
}

fn parse_bedrock_response(body: &Value) -> ProviderResult<ModelResponse> {
    let content = body
        .pointer("/output/message/content")
        .and_then(Value::as_array)
        .ok_or(ProviderError::MissingField("output.message.content"))?;
    let mut message = String::new();
    let mut tool_calls = Vec::new();

    for block in content {
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            message.push_str(text);
        }
        if let Some(tool_use) = block.get("toolUse") {
            tool_calls.push(ToolCall {
                id: tool_use
                    .get("toolUseId")
                    .and_then(Value::as_str)
                    .ok_or(ProviderError::MissingField("toolUse.toolUseId"))?
                    .to_owned(),
                name: tool_use
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or(ProviderError::MissingField("toolUse.name"))?
                    .to_owned(),
                arguments: tool_use.get("input").cloned().unwrap_or_else(|| json!({})),
            });
        }
    }

    Ok(ModelResponse {
        message,
        tool_calls,
        usage: body.get("usage").map(|usage| Usage {
            input_tokens: usage.get("inputTokens").and_then(Value::as_u64),
            output_tokens: usage.get("outputTokens").and_then(Value::as_u64),
        }),
    })
}

fn parse_bedrock_stream_payload(
    payload: &Value,
    content_to_tool_index: &mut HashMap<usize, usize>,
    next_tool_index: &mut usize,
) -> ProviderResult<Vec<ModelStreamEvent>> {
    let mut events = Vec::new();

    if let Some(start) = payload.get("contentBlockStart") {
        let content_index = start
            .get("contentBlockIndex")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if let Some(tool_use) = start.pointer("/start/toolUse") {
            let tool_index = *next_tool_index;
            *next_tool_index += 1;
            content_to_tool_index.insert(content_index, tool_index);
            events.push(ModelStreamEvent::ToolCallDelta {
                index: tool_index,
                id: tool_use
                    .get("toolUseId")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                name: tool_use
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                arguments_delta: String::new(),
            });
        }
    }

    if let Some(delta) = payload.get("contentBlockDelta") {
        let content_index = delta
            .get("contentBlockIndex")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if let Some(text) = delta.pointer("/delta/text").and_then(Value::as_str) {
            events.push(ModelStreamEvent::MessageDelta(text.to_owned()));
        }
        if let Some(partial_json) = delta
            .pointer("/delta/toolUse/input")
            .and_then(Value::as_str)
            && let Some(tool_index) = content_to_tool_index.get(&content_index)
        {
            events.push(ModelStreamEvent::ToolCallDelta {
                index: *tool_index,
                id: None,
                name: None,
                arguments_delta: partial_json.to_owned(),
            });
        }
    }

    if let Some(metadata) = payload.get("metadata")
        && let Some(usage) = metadata.get("usage")
    {
        events.push(ModelStreamEvent::Usage(Usage {
            input_tokens: usage.get("inputTokens").and_then(Value::as_u64),
            output_tokens: usage.get("outputTokens").and_then(Value::as_u64),
        }));
    }

    if payload.get("messageStop").is_some() {
        events.push(ModelStreamEvent::Done);
    }

    Ok(events)
}

fn sign_bedrock_request(
    region: &str,
    credentials: &AwsCredentials,
    url: &str,
    body: &str,
) -> ProviderResult<reqwest::header::HeaderMap> {
    let parsed_url =
        reqwest::Url::parse(url).map_err(|error| ProviderError::Config(error.to_string()))?;
    let host = parsed_url
        .host_str()
        .ok_or_else(|| ProviderError::Config("Bedrock endpoint is missing host".to_owned()))?;
    let path = parsed_url.path();
    let now = OffsetDateTime::now_utc();
    let date_stamp = now
        .format(format_description!("[year][month][day]"))
        .map_err(|error| ProviderError::Config(error.to_string()))?;
    let amz_date = now
        .format(format_description!(
            "[year][month][day]T[hour][minute][second]Z"
        ))
        .map_err(|error| ProviderError::Config(error.to_string()))?;
    let payload_hash = hex::encode(Sha256::digest(body.as_bytes()));

    let mut headers = BTreeMap::<String, String>::new();
    headers.insert("content-type".to_owned(), "application/json".to_owned());
    headers.insert("host".to_owned(), host.to_owned());
    headers.insert("x-amz-date".to_owned(), amz_date.clone());
    if let Some(token) = &credentials.session_token {
        headers.insert("x-amz-security-token".to_owned(), token.clone());
    }

    let canonical_headers = headers
        .iter()
        .map(|(name, value)| format!("{name}:{}\n", value.trim()))
        .collect::<String>();
    let signed_headers = headers.keys().cloned().collect::<Vec<_>>().join(";");
    let canonical_request =
        format!("POST\n{path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}");
    let credential_scope = format!("{date_stamp}/{region}/bedrock/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );
    let signing_key = signing_key(
        &credentials.secret_access_key,
        &date_stamp,
        region,
        "bedrock",
    )?;
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes())?);
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        credentials.access_key_id, credential_scope, signed_headers, signature
    );

    let mut header_map = reqwest::header::HeaderMap::new();
    for (name, value) in headers {
        header_map.insert(
            reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| ProviderError::Config(error.to_string()))?,
            reqwest::header::HeaderValue::from_str(&value)
                .map_err(|error| ProviderError::Config(error.to_string()))?,
        );
    }
    header_map.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&authorization)
            .map_err(|error| ProviderError::Config(error.to_string()))?,
    );

    Ok(header_map)
}

fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> ProviderResult<Vec<u8>> {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes())?;
    let k_region = hmac_sha256(&k_date, region.as_bytes())?;
    let k_service = hmac_sha256(&k_region, service.as_bytes())?;
    hmac_sha256(&k_service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> ProviderResult<Vec<u8>> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|error| ProviderError::Config(error.to_string()))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

struct EventStreamMessage {
    payload: Bytes,
}

impl EventStreamMessage {
    fn payload_json(&self) -> ProviderResult<Option<Value>> {
        if self.payload.is_empty() {
            return Ok(None);
        }
        serde_json::from_slice(&self.payload)
            .map(Some)
            .map_err(|error| {
                ProviderError::Parse(format!("invalid Bedrock stream payload: {error}"))
            })
    }
}

fn take_eventstream_message(buffer: &mut Vec<u8>) -> ProviderResult<Option<EventStreamMessage>> {
    if buffer.len() < 12 {
        return Ok(None);
    }

    let total_len = u32::from_be_bytes(buffer[0..4].try_into().unwrap()) as usize;
    if total_len < 16 {
        return Err(ProviderError::Parse(
            "invalid AWS event-stream frame length".to_owned(),
        ));
    }
    if buffer.len() < total_len {
        return Ok(None);
    }

    let mut frame = Bytes::copy_from_slice(&buffer[..total_len]);
    buffer.drain(..total_len);

    let frame_total_len = frame.get_u32() as usize;
    let headers_len = frame.get_u32() as usize;
    let _prelude_crc = frame.get_u32();

    if frame_total_len != total_len || frame.remaining() < headers_len + 4 {
        return Err(ProviderError::Parse(
            "invalid AWS event-stream frame".to_owned(),
        ));
    }

    frame.advance(headers_len);
    let payload_len = total_len - 12 - headers_len - 4;
    let payload = frame.copy_to_bytes(payload_len);
    let _message_crc = frame.get_u32();

    Ok(Some(EventStreamMessage { payload }))
}
