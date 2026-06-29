use std::io::{self, Write};

use orchestrai::{
    AgentLoop, AgentLoopConfig, LoopEvent, Message, ToolRegistry, providers::AnthropicProvider,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let provider = AnthropicProvider::new(api_key);
    let tools = ToolRegistry::new();
    let mut config = AgentLoopConfig::new("claude-sonnet-4-6");
    config.max_tool_rounds = 0;
    config.max_tokens = Some(256);
    let agent_loop = AgentLoop::new(provider, tools, config);

    let output = agent_loop
        .run_stream(vec![Message::user("Hi")], |event| async move {
            if let LoopEvent::MessageDelta(delta) = event {
                print!("{delta}");
                let _ = io::stdout().flush();
            }
        })
        .await?;

    println!("\n\n--- final ---\n{}", output.final_message);

    Ok(())
}
