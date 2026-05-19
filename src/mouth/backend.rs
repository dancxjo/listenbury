use anyhow::Result;

use crate::audio::frame::AudioFrame;

/// Lower-level trait for TTS synthesis backends.
///
/// Implementors provide synchronous, blocking text-to-audio synthesis.  The
/// higher-level [`TextToSpeech`] trait wraps a backend with queueing,
/// async-style polling, and (optionally) caching.
///
/// [`TextToSpeech`]: crate::mouth::tts::TextToSpeech
pub trait TtsBackend: Send {
    /// Synthesize `text` into a sequence of [`AudioFrame`]s.
    ///
    /// Implementations should strip or reject unsafe fragments (emoji, control
    /// characters) before forwarding text to the underlying model or process.
    /// The [`PiperTextToSpeech`] wrapper already strips emoji via
    /// [`strip_emoji`] before passing text here.
    ///
    /// [`PiperTextToSpeech`]: crate::mouth::piper::PiperTextToSpeech
    /// [`strip_emoji`]: crate::mouth::planner::strip_emoji
    fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>>;

    /// Notify the backend that any in-progress or pending synthesis should
    /// stop.
    ///
    /// The default implementation is a no-op; stateful backends (e.g. a
    /// persistent worker process) may override it to flush queued work.
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

    /// A [`TtsBackend`] that returns predictable, deterministic audio for
    /// use in unit tests.  It records every call so that tests can assert on
    /// the text that was synthesized and how many times `stop` was invoked.
    pub(crate) struct MockTtsBackend {
        pub synthesize_calls: Vec<String>,
        pub stop_calls: usize,
    }

    impl MockTtsBackend {
        pub(crate) fn new() -> Self {
            Self {
                synthesize_calls: Vec::new(),
                stop_calls: 0,
            }
        }
    }

    impl TtsBackend for MockTtsBackend {
        fn synthesize(&mut self, text: &str) -> Result<Vec<AudioFrame>> {
            self.synthesize_calls.push(text.to_string());
            Ok(vec![AudioFrame {
                captured_at: ExactTimestamp::now(),
                sample_rate_hz: 22_050,
                channels: 1,
                samples: vec![0.1, 0.2, 0.3],
                voice_signatures: Vec::new(),
            }])
        }

        fn stop(&mut self) -> Result<()> {
            self.stop_calls += 1;
            Ok(())
        }
    }

    #[test]
    fn mock_backend_records_synthesize_calls() {
        let mut backend = MockTtsBackend::new();
        let frames = backend.synthesize("hello world").expect("synthesize");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples, vec![0.1, 0.2, 0.3]);
        assert_eq!(backend.synthesize_calls, vec!["hello world".to_string()]);
    }

    #[test]
    fn mock_backend_stop_is_counted() {
        let mut backend = MockTtsBackend::new();
        backend.stop().expect("stop");
        backend.stop().expect("stop again");
        assert_eq!(backend.stop_calls, 2);
    }

    #[test]
    fn mock_backend_accumulates_multiple_calls() {
        let mut backend = MockTtsBackend::new();
        backend.synthesize("first").expect("first");
        backend.synthesize("second").expect("second");
        assert_eq!(backend.synthesize_calls.len(), 2);
        assert_eq!(backend.synthesize_calls[0], "first");
        assert_eq!(backend.synthesize_calls[1], "second");
    }
}
