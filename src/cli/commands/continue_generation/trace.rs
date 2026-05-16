#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
use super::*;

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) fn wrap_live_input(text: &str) -> String {
    format!(
        "\n\n--- LIVE EVENT: user ---\n{}\n--- END LIVE EVENT ---\n\n",
        text.trim()
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) fn wrap_time_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: clock ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) fn wrap_ear_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: ear ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(test)]
pub(super) fn wrap_mouth_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: mouth ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(test)]
pub(super) fn wrap_runtime_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: runtime ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) fn wrap_source_event(message: &str) -> String {
    format!("\n\n--- LIVE EVENT: source ---\n{message}\n--- END LIVE EVENT ---\n\n")
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) fn current_time_message() -> String {
    let now = Local::now();
    format!(
        "The current local time is {}. Unix time is {}.{:03} seconds.",
        now.to_rfc3339_opts(SecondsFormat::Millis, false),
        now.timestamp(),
        now.timestamp_subsec_millis()
    )
}

#[cfg(any(
    test,
    all(
        feature = "audio-cpal",
        feature = "asr-whisper",
        feature = "llm-llama-cpp",
        feature = "tts-piper"
    )
))]
pub(super) fn next_time_event_interval(jitter_state: &mut u64) -> Duration {
    *jitter_state ^= *jitter_state << 7;
    *jitter_state ^= *jitter_state >> 9;
    *jitter_state ^= *jitter_state << 8;
    if *jitter_state == 0 {
        *jitter_state = 0x9e37_79b9_7f4a_7c15;
    }

    let span = TIME_EVENT_INTERVAL_JITTER_MS * 2 + 1;
    let jitter = (*jitter_state % span) as i64 - TIME_EVENT_INTERVAL_JITTER_MS as i64;
    Duration::from_millis((TIME_EVENT_INTERVAL_BASE_MS as i64 + jitter) as u64)
}
