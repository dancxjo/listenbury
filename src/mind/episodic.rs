use crate::mind::controller::ConversationRole;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodicSpeaker {
    User,
    Pete,
}

impl EpisodicSpeaker {
    fn label(&self) -> &'static str {
        match self {
            Self::User => "User",
            Self::Pete => "Pete",
        }
    }
}

impl From<ConversationRole> for EpisodicSpeaker {
    fn from(value: ConversationRole) -> Self {
        match value {
            ConversationRole::User => Self::User,
            ConversationRole::Pete => Self::Pete,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicTurn {
    pub speaker: EpisodicSpeaker,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageInstruction {
    pub text: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicScene {
    pub number: usize,
    pub topic: String,
    pub summary: String,
    pub stage_instruction: StageInstruction,
    pub turns: Vec<EpisodicTurn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicEpisode {
    pub number: usize,
    pub title: String,
    pub summary: String,
    pub scenes: Vec<EpisodicScene>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodicMemory {
    pub current_stage_instruction: StageInstruction,
    pub episodes: Vec<EpisodicEpisode>,
}

impl EpisodicMemory {
    pub fn empty() -> Self {
        let stage = StageInstruction {
            text: "Pete is waiting for the next live event.".to_string(),
            summary: "No active screenplay beat yet.".to_string(),
        };
        Self {
            current_stage_instruction: stage,
            episodes: Vec::new(),
        }
    }

    pub fn from_turns(
        conversation_tail: impl IntoIterator<Item = EpisodicTurn>,
        current_user_message: &str,
    ) -> Self {
        let mut turns = conversation_tail
            .into_iter()
            .filter(|turn| !turn.text.trim().is_empty())
            .collect::<Vec<_>>();
        let current = current_user_message.trim();
        if !current.is_empty() {
            turns.push(EpisodicTurn {
                speaker: EpisodicSpeaker::User,
                text: current.to_string(),
            });
        }
        if turns.is_empty() {
            return Self::empty();
        }

        let scenes = build_scenes(turns);
        let episode_summary = summarize_episode(&scenes);
        let current_stage_instruction = scenes
            .last()
            .map(|scene| scene.stage_instruction.clone())
            .unwrap_or_else(|| Self::empty().current_stage_instruction);
        Self {
            current_stage_instruction,
            episodes: vec![EpisodicEpisode {
                number: 1,
                title: "Live session so far".to_string(),
                summary: episode_summary,
                scenes,
            }],
        }
    }

    pub fn render_prompt_summary(&self) -> String {
        if self.episodes.is_empty() {
            return format!(
                "Current screenplay beat: {}\nScene timeline: no scenes yet.",
                self.current_stage_instruction.text
            );
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "Current screenplay beat: {}",
            self.current_stage_instruction.text
        ));
        if !self.current_stage_instruction.summary.trim().is_empty()
            && self.current_stage_instruction.summary != self.current_stage_instruction.text
        {
            lines.push(format!(
                "Current action summary: {}",
                self.current_stage_instruction.summary
            ));
        }
        lines.push("Scene timeline:".to_string());
        for episode in &self.episodes {
            lines.push(format!(
                "- Episode {}: {}. {}",
                episode.number, episode.title, episode.summary
            ));
            for scene in &episode.scenes {
                lines.push(format!(
                    "  - Scene {}: Action: {} Screenplay beat: {}",
                    scene.number, scene.summary, scene.stage_instruction.text
                ));
            }
        }
        lines.join("\n")
    }
}

fn build_scenes(turns: Vec<EpisodicTurn>) -> Vec<EpisodicScene> {
    let topic = "active scene".to_string();
    let summary = summarize_scene(&topic, &turns);
    let stage_instruction = StageInstruction {
        text: stage_instruction_for_scene(&topic, &turns),
        summary: summary.clone(),
    };
    vec![EpisodicScene {
        number: 1,
        topic,
        summary,
        stage_instruction,
        turns,
    }]
}

fn summarize_scene(_topic: &str, turns: &[EpisodicTurn]) -> String {
    let lead = turns
        .iter()
        .rev()
        .find(|turn| turn.speaker == EpisodicSpeaker::User)
        .or_else(|| turns.last())
        .map(|turn| compact_text(&turn.text, 120))
        .unwrap_or_else(|| "the live exchange continues".to_string());
    format!(
        "{} turn{} of action around {}",
        turns.len(),
        if turns.len() == 1 { "" } else { "s" },
        lead
    )
}

fn summarize_episode(scenes: &[EpisodicScene]) -> String {
    if scenes.is_empty() {
        return "No scenes yet.".to_string();
    }
    format!(
        "{} active scene{}",
        scenes.len(),
        if scenes.len() == 1 { "" } else { "s" }
    )
}

fn stage_instruction_for_scene(_topic: &str, turns: &[EpisodicTurn]) -> String {
    let speaker_list = turns
        .iter()
        .map(|turn| turn.speaker.label())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(" and ");
    let verb = if speaker_list.contains(" and ") {
        "are"
    } else {
        "is"
    };
    let latest = turns
        .last()
        .map(|turn| compact_text(&turn.text, 140))
        .unwrap_or_else(|| "the room is quiet".to_string());
    format!(
        "Setting: a live spoken session. Action: {speaker_list} {verb} advancing the current scene; latest beat: {latest}"
    )
}

fn compact_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let mut truncated = compact
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episodic_memory_renders_current_stage_instruction_and_timeline() {
        let memory = EpisodicMemory::from_turns(
            [
                EpisodicTurn {
                    speaker: EpisodicSpeaker::User,
                    text: "We need Pete to remember scenes and episodes.".to_string(),
                },
                EpisodicTurn {
                    speaker: EpisodicSpeaker::Pete,
                    text: "I can keep an episodic summary.".to_string(),
                },
            ],
            "This should be available to the LLM as what's going on.",
        );

        let rendered = memory.render_prompt_summary();
        assert!(rendered.contains("Current screenplay beat:"));
        assert!(rendered.contains("Current action summary:"));
        assert!(rendered.contains("Scene timeline:"));
        assert!(rendered.contains("Action:"));
        assert!(rendered.contains("Screenplay beat:"));
        assert!(rendered.contains("Scene 1"));
        assert!(!rendered.contains("memory and continuity"));
        assert!(!rendered.contains("hearing and voice"));
    }

    #[test]
    fn compact_text_truncates_on_character_boundaries() {
        let compact = compact_text(
            "All right, I can summarize Unicode safely with curly quotes and ellipses: “listen…”",
            72,
        );

        assert!(compact.ends_with("..."));
        assert!(compact.chars().count() <= 72);
    }
}
