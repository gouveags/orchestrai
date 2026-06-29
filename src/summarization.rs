use serde::{Deserialize, Serialize};

use crate::types::Message;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub content: String,
}

impl ConversationSummary {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SummaryPolicy {
    summary: ConversationSummary,
    minimum_messages: usize,
}

impl SummaryPolicy {
    pub fn always(summary: ConversationSummary) -> Self {
        Self {
            summary,
            minimum_messages: 0,
        }
    }

    pub fn after_messages(summary: ConversationSummary, minimum_messages: usize) -> Self {
        Self {
            summary,
            minimum_messages,
        }
    }

    pub fn prepare(&self, messages: &[Message]) -> PreparedMessages {
        let summary = (messages.len() >= self.minimum_messages).then(|| self.summary.clone());
        let mut prepared = Vec::with_capacity(messages.len() + usize::from(summary.is_some()));

        if let Some(summary) = &summary {
            prepared.push(Message::system(format!(
                "Conversation summary derived from earlier context:\n{}",
                summary.content
            )));
        }
        prepared.extend(messages.iter().cloned());

        PreparedMessages {
            messages: prepared,
            injected_summary: summary,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedMessages {
    pub messages: Vec<Message>,
    pub injected_summary: Option<ConversationSummary>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_policy_prepends_derived_context_without_mutating_messages() {
        let policy = SummaryPolicy::always(ConversationSummary::new("The user picked Rust."));
        let messages = vec![Message::user("Keep going."), Message::assistant("On it.")];

        let prepared = policy.prepare(&messages);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], Message::user("Keep going."));
        assert_eq!(prepared.messages.len(), 3);
        assert!(
            prepared.messages[0]
                .content
                .contains("The user picked Rust.")
        );
        assert_eq!(prepared.messages[1..], messages);
        assert_eq!(
            prepared.injected_summary,
            Some(ConversationSummary::new("The user picked Rust."))
        );
    }

    #[test]
    fn summary_policy_can_wait_until_history_is_large_enough() {
        let policy = SummaryPolicy::after_messages(ConversationSummary::new("Older context."), 3);
        let messages = vec![Message::user("One"), Message::assistant("Two")];

        let prepared = policy.prepare(&messages);

        assert_eq!(prepared.messages, messages);
        assert_eq!(prepared.injected_summary, None);
    }
}
