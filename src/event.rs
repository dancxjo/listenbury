use crate::hearing::breath::{BreathGroupEndReason, BreathGroupId};
pub use crate::speech_timeline::UtteranceId;

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
    Partial {
        utterance_id: UtteranceId,
        text: String,
    },
    Final {
        utterance_id: UtteranceId,
        text: String,
    },
}

#[derive(Debug, Clone)]
pub enum MindEvent {
    GenerationStarted {
        utterance_id: UtteranceId,
    },
    Token {
        utterance_id: UtteranceId,
        text: String,
    },
    GenerationCompleted {
        utterance_id: UtteranceId,
    },
}

#[derive(Debug, Clone)]
pub enum MouthEvent {
    SpeakRequested {
        utterance_id: UtteranceId,
    },
    SpeakStarted {
        utterance_id: UtteranceId,
    },
    SpeakInterrupted {
        utterance_id: UtteranceId,
    },
    SpeakResumed {
        utterance_id: UtteranceId,
    },
    SpeakAborted {
        utterance_id: UtteranceId,
        reason: String,
    },
    SpeakFinished {
        utterance_id: UtteranceId,
    },
}

#[derive(Debug, Clone)]
pub enum VisionEvent {
    FrameCaptured,
}
