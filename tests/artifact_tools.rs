use std::sync::Arc;

use orchestrai::{
    LIST_ARTIFACTS_TOOL, LocalArtifactStore, PUBLISH_ARTIFACT_TOOL, READ_ARTIFACT_TOOL,
    ToolRegistry, register_artifact_tools,
};
use serde_json::{Value, json};

#[tokio::test]
async fn artifact_tools_publish_list_and_read_from_local_store() {
    let temp = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalArtifactStore::new(temp.path()).unwrap());
    let mut registry = ToolRegistry::new();
    register_artifact_tools(&mut registry, store);

    let published = registry
        .execute(
            PUBLISH_ARTIFACT_TOOL,
            json!({
                "title": "Analysis Summary",
                "content": "# Summary\nAll good.",
                "media_type": "text/markdown"
            }),
        )
        .await
        .unwrap();
    let published = serde_json::from_str::<Value>(&published).unwrap();
    let id = published["id"].as_str().unwrap();

    assert_eq!(published["title"], "Analysis Summary");
    assert_eq!(published["media_type"], "text/markdown");
    assert_eq!(published["bytes"], 19);
    assert!(published["path"].as_str().unwrap().ends_with(".md"));

    let listed = registry
        .execute(LIST_ARTIFACTS_TOOL, json!({}))
        .await
        .unwrap();
    let listed = serde_json::from_str::<Value>(&listed).unwrap();
    assert_eq!(listed["artifacts"].as_array().unwrap().len(), 1);
    assert_eq!(listed["artifacts"][0]["id"], id);

    let read = registry
        .execute(READ_ARTIFACT_TOOL, json!({"id": id}))
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&read).unwrap(),
        json!({
            "id": id,
            "title": "Analysis Summary",
            "media_type": "text/markdown",
            "content": "# Summary\nAll good."
        })
    );
}

#[tokio::test]
async fn artifact_publish_rejects_root_escape_paths_as_hard_failures() {
    let temp = tempfile::tempdir().unwrap();
    let store = Arc::new(LocalArtifactStore::new(temp.path()).unwrap());
    let mut registry = ToolRegistry::new();
    register_artifact_tools(&mut registry, store);

    let error = registry
        .execute(
            PUBLISH_ARTIFACT_TOOL,
            json!({
                "title": "Secret",
                "content": "nope",
                "path": "../secret.md"
            }),
        )
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("path `../secret.md` is outside the artifact root")
    );
}
