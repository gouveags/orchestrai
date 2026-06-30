use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    tool::{BoxToolFuture, FnTool, ToolError, ToolRegistry, ToolResult as RegistryToolResult},
    types::ToolDefinition,
};

type BuiltInPlanTool = FnTool<Box<dyn Fn(Value) -> BoxToolFuture + Send + Sync>>;

pub const PLAN_CREATE_TOOL: &str = "plan_create";
pub const PLAN_UPDATE_TOOL: &str = "plan_update";
pub const PLAN_READ_TOOL: &str = "plan_read";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub items: Vec<PlanItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanItem {
    pub id: String,
    pub text: String,
    pub status: PlanItemStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanItemStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

impl PlanItemStatus {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            "blocked" => Some(Self::Blocked),
            _ => None,
        }
    }
}

#[derive(Clone, Default)]
pub struct PlanToolSet {
    state: Arc<Mutex<Plan>>,
}

impl PlanToolSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn current_plan(&self) -> Plan {
        self.state.lock().unwrap().clone()
    }

    pub fn register(&self, registry: &mut ToolRegistry) {
        registry.register(self.create_tool());
        registry.register(self.update_tool());
        registry.register(self.read_tool());
    }

    fn create_tool(&self) -> BuiltInPlanTool {
        let state = Arc::clone(&self.state);
        FnTool::new(
            ToolDefinition::new(
                PLAN_CREATE_TOOL,
                "Create or replace the current plan with ordered items.",
                json!({
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["items"]
                }),
            ),
            Box::new(move |arguments| {
                let state = Arc::clone(&state);
                Box::pin(async move {
                    let items = read_string_array(&arguments, "items")?;
                    let title = arguments
                        .get("title")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    let plan = Plan {
                        title,
                        items: items
                            .into_iter()
                            .enumerate()
                            .map(|(index, text)| PlanItem {
                                id: (index + 1).to_string(),
                                text,
                                status: PlanItemStatus::Pending,
                            })
                            .collect(),
                    };
                    *state.lock().unwrap() = plan.clone();
                    Ok(render_plan_result("created", &plan))
                })
            }),
        )
    }

    fn update_tool(&self) -> BuiltInPlanTool {
        let state = Arc::clone(&self.state);
        FnTool::new(
            ToolDefinition::new(
                PLAN_UPDATE_TOOL,
                "Update one item in the current plan.",
                json!({
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "blocked"]
                        },
                        "text": {"type": "string"}
                    },
                    "required": ["id"]
                }),
            ),
            Box::new(move |arguments| {
                let state = Arc::clone(&state);
                Box::pin(async move {
                    let id = read_string(&arguments, "id")?;
                    let status = arguments
                        .get("status")
                        .and_then(Value::as_str)
                        .map(|value| {
                            PlanItemStatus::parse(value).ok_or_else(|| {
                                ToolError::Execution(format!(
                                    "status must be one of pending, in_progress, completed, blocked; got `{value}`"
                                ))
                            })
                        })
                        .transpose()?;
                    let text = arguments
                        .get("text")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);

                    let mut plan = state.lock().unwrap();
                    let item = plan
                        .items
                        .iter_mut()
                        .find(|item| item.id == id)
                        .ok_or_else(|| {
                            ToolError::Execution(format!("plan item `{id}` was not found"))
                        })?;

                    if let Some(status) = status {
                        item.status = status;
                    }
                    if let Some(text) = text {
                        item.text = text;
                    }

                    Ok(render_plan_result("updated", &plan))
                })
            }),
        )
    }

    fn read_tool(&self) -> BuiltInPlanTool {
        let state = Arc::clone(&self.state);
        FnTool::new(
            ToolDefinition::new(
                PLAN_READ_TOOL,
                "Read the current plan.",
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            ),
            Box::new(move |_arguments| {
                let state = Arc::clone(&state);
                Box::pin(async move {
                    let plan = state.lock().unwrap().clone();
                    Ok(render_plan_result("read", &plan))
                })
            }),
        )
    }
}

fn read_string(arguments: &Value, field: &str) -> RegistryToolResult<String> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| ToolError::Execution(format!("`{field}` must be a string")))
}

fn read_string_array(arguments: &Value, field: &str) -> RegistryToolResult<Vec<String>> {
    let values = arguments
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| ToolError::Execution(format!("`{field}` must be an array of strings")))?;

    values
        .iter()
        .map(|value| {
            value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                ToolError::Execution(format!("`{field}` must be an array of strings"))
            })
        })
        .collect()
}

fn render_plan_result(action: &str, plan: &Plan) -> String {
    json!({
        "action": action,
        "plan": plan,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn plan_tools_create_update_and_read_the_shared_plan() {
        let plan_tools = PlanToolSet::new();
        let mut registry = ToolRegistry::new();
        plan_tools.register(&mut registry);

        registry
            .execute(
                PLAN_CREATE_TOOL,
                json!({
                    "title": "Ship the feature",
                    "items": ["Write tests", "Implement code"]
                }),
            )
            .await
            .unwrap();
        registry
            .execute(PLAN_UPDATE_TOOL, json!({"id": "1", "status": "completed"}))
            .await
            .unwrap();

        let output = registry.execute(PLAN_READ_TOOL, json!({})).await.unwrap();
        let output: Value = serde_json::from_str(&output).unwrap();

        assert_eq!(output["plan"]["title"], "Ship the feature");
        assert_eq!(output["plan"]["items"][0]["text"], "Write tests");
        assert_eq!(output["plan"]["items"][0]["status"], "completed");
        assert_eq!(output["plan"]["items"][1]["status"], "pending");
        assert_eq!(
            plan_tools.current_plan().items[0].status,
            PlanItemStatus::Completed
        );
    }
}
