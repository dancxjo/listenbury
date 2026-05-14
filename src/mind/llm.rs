use std::collections::HashMap;

use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenerationId(pub Uuid);

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum LlmEvent {
    Token { text: String },
    Completed,
    Cancelled,
    Error { message: String },
}

pub trait LlmEngine {
    fn start(&mut self, request: GenerationRequest) -> anyhow::Result<GenerationId>;
    fn poll(&mut self, id: GenerationId) -> anyhow::Result<Vec<LlmEvent>>;
    fn cancel(&mut self, id: GenerationId) -> anyhow::Result<()>;
}

#[derive(Debug)]
pub struct MockLlmEngine {
    response_tokens: Vec<String>,
    active: HashMap<GenerationId, usize>,
}

impl MockLlmEngine {
    pub fn with_response(response_tokens: Vec<String>) -> Self {
        Self {
            response_tokens,
            active: HashMap::new(),
        }
    }
}

impl LlmEngine for MockLlmEngine {
    fn start(&mut self, _request: GenerationRequest) -> anyhow::Result<GenerationId> {
        let id = GenerationId(Uuid::new_v4());
        self.active.insert(id, 0);
        Ok(id)
    }

    fn poll(&mut self, id: GenerationId) -> anyhow::Result<Vec<LlmEvent>> {
        let Some(index) = self.active.get_mut(&id) else {
            return Ok(vec![LlmEvent::Error {
                message: "generation not found".to_string(),
            }]);
        };

        if *index < self.response_tokens.len() {
            let event = LlmEvent::Token {
                text: self.response_tokens[*index].clone(),
            };
            *index += 1;
            return Ok(vec![event]);
        }

        self.active.remove(&id);
        Ok(vec![LlmEvent::Completed])
    }

    fn cancel(&mut self, id: GenerationId) -> anyhow::Result<()> {
        if self.active.remove(&id).is_some() {
            Ok(())
        } else {
            anyhow::bail!("generation not found")
        }
    }
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: None,
        }
    }
}

impl Default for MockLlmEngine {
    fn default() -> Self {
        Self::with_response(vec!["I ".into(), "heard ".into(), "you.".into()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_llm_streams_tokens_before_completion() {
        let mut engine = MockLlmEngine::default();
        let id = engine
            .start(GenerationRequest {
                prompt: "hello".to_string(),
                max_tokens: None,
            })
            .expect("start should succeed");

        let first = engine.poll(id).expect("first poll should succeed");
        assert!(first.iter().any(|ev| matches!(ev, LlmEvent::Token { .. })));

        let mut saw_completed = false;
        for _ in 0..5 {
            let events = engine.poll(id).expect("poll should succeed");
            if events.iter().any(|ev| matches!(ev, LlmEvent::Completed)) {
                saw_completed = true;
                break;
            }
        }

        assert!(saw_completed);
    }
}
