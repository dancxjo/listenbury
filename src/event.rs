use crate::hearing::breath::{BreathGroupEndReason, BreathGroupId};

#[derive(Debug, Clone)]
pub enum PeteEvent {
    Audio(AudioEvent),
    Hearing(HearingEvent),
    Transcript(TranscriptEvent),
    Mind(MindEvent),
    Mouth(MouthEvent),
    Vision(VisionEvent),
}

#[derive(Debug, Clone)]
pub enum AudioEvent {
    FrameReceived,
    FrameDropped,
}

#[derive(Debug, Clone)]
pub enum HearingEvent {
    SpeechStarted,
    SpeechContinued {
        speech_prob: f32,
    },
    PauseStarted,
    BreathGroupOpened {
        id: BreathGroupId,
    },
    BreathGroupClosed {
        id: BreathGroupId,
        reason: BreathGroupEndReason,
    },
}

#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    Partial { text: String },
    Final { text: String },
}

#[derive(Debug, Clone)]
pub enum MindEvent {
    GenerationStarted,
    Token { text: String },
    GenerationCompleted,
}

#[derive(Debug, Clone)]
pub enum MouthEvent {
    SpeakRequested,
    SpeakStarted,
    SpeakFinished,
}

#[derive(Debug, Clone)]
pub enum VisionEvent {
    FrameCaptured,
}
