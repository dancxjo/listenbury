use std::sync::mpsc;

use super::trace::MemoryTrace;

/// Receives [`MemoryTrace`] events emitted by the Listenbury runtime.
///
/// Implementations **must not** block the caller: a slow or failing sink must
/// not stall audio processing, LLM inference, or TTS synthesis.
///
/// The trait is object-safe and requires `Send + Sync` so that a single sink
/// can be shared across threads via `Arc<dyn MemorySink>`.
pub trait MemorySink: Send + Sync {
    /// Submit a trace for recording.
    ///
    /// Implementations should return immediately; any I/O or heavy processing
    /// must happen on a background worker.  A failed submission (e.g. a full
    /// channel) must be handled silently — callers do not inspect the result.
    fn submit(&self, trace: MemoryTrace);
}

// ---------------------------------------------------------------------------
// NoopMemorySink
// ---------------------------------------------------------------------------

/// A [`MemorySink`] that silently discards every trace.
///
/// This is the default implementation used when no persistent memory backend
/// has been configured.  It has zero overhead beyond the `submit` call itself.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopMemorySink;

impl MemorySink for NoopMemorySink {
    #[inline]
    fn submit(&self, _trace: MemoryTrace) {}
}

// ---------------------------------------------------------------------------
// ChannelMemorySink
// ---------------------------------------------------------------------------

/// A [`MemorySink`] that forwards traces to a standard-library channel.
///
/// Construct it with [`ChannelMemorySink::new`], keep the returned
/// [`mpsc::Receiver`] on a background worker thread, and drop both when
/// the system shuts down.
///
/// Sending on the channel is non-blocking: if the channel is disconnected the
/// trace is silently dropped so that memory failures never break conversation.
#[derive(Debug)]
pub struct ChannelMemorySink {
    tx: mpsc::SyncSender<MemoryTrace>,
}

impl ChannelMemorySink {
    /// Create a new sink backed by a bounded synchronous channel.
    ///
    /// `capacity` is the maximum number of [`MemoryTrace`] events that can be
    /// buffered.  Once the channel is full, additional calls to `submit` drop
    /// the trace immediately rather than blocking.  A value of `256` is a
    /// reasonable default for most workloads.
    pub fn new(capacity: usize) -> (Self, mpsc::Receiver<MemoryTrace>) {
        let (tx, rx) = mpsc::sync_channel(capacity);
        (Self { tx }, rx)
    }
}

impl MemorySink for ChannelMemorySink {
    fn submit(&self, trace: MemoryTrace) {
        // `try_send` never blocks.  A full channel or a disconnected receiver
        // is silently ignored to preserve real-time conversational behaviour.
        let _ = self.tx.try_send(trace);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::trace::{MemoryImageVector, MemoryTrace, MemoryVoiceVector, SpeakerRole};
    use crate::time::ExactTimestamp;

    fn sample_trace() -> MemoryTrace {
        MemoryTrace::ConversationTurnFinalized {
            speaker: SpeakerRole::UnknownVoice { ordinal: 1 },
            text: "hello".to_string(),
            occurred_at: ExactTimestamp::now(),
        }
    }

    // --- NoopMemorySink ---

    #[test]
    fn noop_sink_does_not_block() {
        let sink = NoopMemorySink;
        // submitting many traces must not block or panic
        for _ in 0..1_000 {
            sink.submit(sample_trace());
        }
    }

    #[test]
    fn noop_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoopMemorySink>();
    }

    // --- ChannelMemorySink ---

    #[test]
    fn channel_sink_delivers_trace_to_receiver() {
        let (sink, rx) = ChannelMemorySink::new(8);
        sink.submit(MemoryTrace::ConversationTurnFinalized {
            speaker: SpeakerRole::Pete,
            text: "hi there".to_string(),
            occurred_at: ExactTimestamp::now(),
        });
        let received = rx.recv().expect("trace should be received");
        match received {
            MemoryTrace::ConversationTurnFinalized { text, .. } => {
                assert_eq!(text, "hi there");
            }
            other => panic!("unexpected trace variant: {:?}", other),
        }
    }

    #[test]
    fn channel_sink_does_not_block_when_full() {
        let capacity = 4;
        let (sink, _rx) = ChannelMemorySink::new(capacity);
        // Fill the channel beyond capacity — submit must not block or panic.
        for _ in 0..(capacity * 2) {
            sink.submit(sample_trace());
        }
    }

    #[test]
    fn channel_sink_drops_traces_when_receiver_is_dropped() {
        let (sink, rx) = ChannelMemorySink::new(8);
        drop(rx);
        // Must not panic even though the receiver is gone.
        sink.submit(sample_trace());
    }

    #[test]
    fn channel_sink_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ChannelMemorySink>();
    }

    #[test]
    fn channel_sink_delivers_all_trace_variants() {
        let (sink, rx) = ChannelMemorySink::new(16);
        let now = ExactTimestamp::now();

        sink.submit(MemoryTrace::TimedWordStreamFinalized {
            stream_id: "stream-1".to_string(),
            summary: "word summary".to_string(),
            occurred_at: now,
        });
        sink.submit(MemoryTrace::MouthPlaybackStarted {
            utterance_id: 42,
            text: "saying something".to_string(),
            occurred_at: now,
        });
        sink.submit(MemoryTrace::MouthPlaybackCompleted {
            utterance_id: 42,
            text: "saying something".to_string(),
            occurred_at: now,
        });
        sink.submit(MemoryTrace::AuditorySceneObservation {
            description: "dog barking".to_string(),
            salience: 0.8,
            occurred_at: now,
        });
        sink.submit(MemoryTrace::OverlapDetected {
            description: "both speakers active".to_string(),
            occurred_at: now,
        });
        sink.submit(MemoryTrace::RecallResultUsed {
            query: "what did we discuss?".to_string(),
            result_summary: "we discussed weather".to_string(),
            occurred_at: now,
        });
        sink.submit(MemoryTrace::ImageVectorCaptured {
            image: MemoryImageVector {
                image_id: "image:test".to_string(),
                source: "test".to_string(),
                width: 1,
                height: 1,
                vector: vec![1.0],
                content_node_id: None,
                retained_image: false,
            },
            captured_at: now,
        });
        sink.submit(MemoryTrace::VoiceVectorCaptured {
            voice: MemoryVoiceVector {
                voice_signature_id: "voice-sig:test".to_string(),
                voice_node_id: "voice:test".to_string(),
                source: "test".to_string(),
                span_id: None,
                vector: vec![1.0],
                confidence: 0.7,
            },
            captured_at: now,
        });

        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::TimedWordStreamFinalized { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::MouthPlaybackStarted { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::MouthPlaybackCompleted { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::AuditorySceneObservation { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::OverlapDetected { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::RecallResultUsed { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::ImageVectorCaptured { .. }
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            MemoryTrace::VoiceVectorCaptured { .. }
        ));
    }
}
