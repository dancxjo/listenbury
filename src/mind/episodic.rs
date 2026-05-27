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
            summary: "No active pericope yet.".to_string(),
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
                "Current stage instruction: {}\nTimeline: no scenes yet.",
                self.current_stage_instruction.text
            );
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "Current stage instruction: {}",
            self.current_stage_instruction.text
        ));
        if !self.current_stage_instruction.summary.trim().is_empty()
            && self.current_stage_instruction.summary != self.current_stage_instruction.text
        {
            lines.push(format!(
                "Current stage summary: {}",
                self.current_stage_instruction.summary
            ));
        }
        lines.push("Episodic timeline:".to_string());
        for episode in &self.episodes {
            lines.push(format!(
                "- Episode {}: {}. {}",
                episode.number, episode.title, episode.summary
            ));
            for scene in &episode.scenes {
                lines.push(format!(
                    "  - Scene {} [{}]: {} Stage: {}",
                    scene.number, scene.topic, scene.summary, scene.stage_instruction.text
                ));
            }
        }
        lines.join("\n")
    }
}

fn build_scenes(turns: Vec<EpisodicTurn>) -> Vec<EpisodicScene> {
    let mut groups: Vec<(String, Vec<EpisodicTurn>)> = Vec::new();
    for turn in turns {
        let topic = classify_topic(&turn.text);
        let current = groups.last_mut();
        match current {
            Some((current_topic, current_turns))
                if *current_topic == topic
                    || (current_turns.len() < 3 && topic == "live session") =>
            {
                if *current_topic == "live session" && topic != "live session" {
                    *current_topic = topic;
                }
                current_turns.push(turn);
            }
            _ => groups.push((topic, vec![turn])),
        }
    }

    groups
        .into_iter()
        .enumerate()
        .map(|(index, (topic, turns))| {
            let summary = summarize_scene(&topic, &turns);
            let stage_instruction = StageInstruction {
                text: stage_instruction_for_scene(&topic, &turns),
                summary: summary.clone(),
            };
            EpisodicScene {
                number: index + 1,
                topic,
                summary,
                stage_instruction,
                turns,
            }
        })
        .collect()
}

fn classify_topic(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if contains_any(
        &lower,
        &[
            "memory", "remember", "episode", "scene", "timeline", "summar",
        ],
    ) {
        "memory and continuity".to_string()
    } else if contains_any(&lower, &["hear", "listening", "microphone", "asr", "voice"]) {
        "hearing and voice".to_string()
    } else if contains_any(&lower, &["stop", "sleep", "shutdown", "goodnight"]) {
        "session shutdown".to_string()
    } else {
        "live session".to_string()
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
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
        "{} turn{} around {}",
        turns.len(),
        if turns.len() == 1 { "" } else { "s" },
        lead
    )
}

fn summarize_episode(scenes: &[EpisodicScene]) -> String {
    if scenes.is_empty() {
        return "No scenes yet.".to_string();
    }
    scenes
        .iter()
        .map(|scene| scene.topic.as_str())
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn stage_instruction_for_scene(topic: &str, turns: &[EpisodicTurn]) -> String {
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
    format!("{speaker_list} {verb} in a {topic} pericope: {latest}")
}

fn compact_text(text: &str, max_chars: usize) -> String {
    let mut compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= max_chars {
        return compact;
    }
    compact.truncate(max_chars.saturating_sub(3));
    compact.push_str("...");
    compact
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
        assert!(rendered.contains("Current stage instruction:"));
        assert!(rendered.contains("Episodic timeline:"));
        assert!(rendered.contains("memory and continuity"));
        assert!(rendered.contains("Scene 1"));
    }
}
