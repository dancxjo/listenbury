use anyhow::Result;
use listenbury::soundscape::{
    SoundscapeDebugView, SoundscapeFrame, SourceAttributedTranscript, SourceHypothesis, SourceId,
    SourceKind, SourceLabel, SoundSource, TimePoint, TimeRange, VoiceCount, VoiceSignatureId,
    AttributionEvidence, detect_overlaps,
};

use crate::cli::SoundscapeDebugCommand;

/// JSON input accepted by the `soundscape-debug` command.
#[derive(Debug, serde::Deserialize)]
struct SoundscapeDebugInput {
    pub frame: SoundscapeFrame,
    pub voice_count: VoiceCount,
    #[serde(default)]
    pub hypotheses: Vec<SourceHypothesis>,
    #[serde(default)]
    pub transcripts: Vec<SourceAttributedTranscript>,
}

pub(crate) fn run_soundscape_debug(cmd: SoundscapeDebugCommand) -> Result<()> {
    let view = if cmd.sample {
        build_sample_view()
    } else if let Some(input_path) = cmd.input {
        let json = std::fs::read_to_string(&input_path)
            .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", input_path.display()))?;
        let input: SoundscapeDebugInput = serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("failed to parse soundscape debug input: {e}"))?;
        let overlaps = detect_overlaps(&input.hypotheses);
        SoundscapeDebugView::from_components(
            &input.frame,
            input.voice_count,
            &input.hypotheses,
            &overlaps,
            &input.transcripts,
        )
    } else {
        eprintln!("No input provided.");
        eprintln!("  --sample      Print a demo debug view showing the output format.");
        eprintln!("  --input FILE  Load a JSON soundscape debug input and print its debug view.");
        eprintln!("  --pretty      Pretty-print the JSON output (default: compact).");
        return Ok(());
    };

    let json = if cmd.pretty {
        serde_json::to_string_pretty(&view)?
    } else {
        serde_json::to_string(&view)?
    };
    println!("{json}");
    Ok(())
}

fn build_sample_view() -> SoundscapeDebugView {
    let pete_id = SourceId::new();
    let range = TimeRange::new(TimePoint::from_millis(12_000), TimePoint::from_millis(15_000));

    let frame = SoundscapeFrame {
        range,
        sources: vec![
            SoundSource {
                id: pete_id,
                kind: SourceKind::KnownSelfVoice,
                label: SourceLabel::NamedVoice("Pete".into()),
                confidence: 0.96,
            },
            SoundSource {
                id: SourceId::new(),
                kind: SourceKind::Voice,
                label: SourceLabel::UnknownVoice { ordinal: 1 },
                confidence: 0.68,
            },
        ],
        events: vec![],
        mixtures: vec![],
    };

    let hypotheses = vec![
        SourceHypothesis {
            source_id: Some(pete_id),
            kind: SourceKind::KnownSelfVoice,
            range: TimeRange::new(
                TimePoint::from_millis(12_000),
                TimePoint::from_millis(13_500),
            ),
            confidence: 0.96,
            evidence: vec![
                AttributionEvidence::MatchesExpectedPlayback {
                    source_id: pete_id,
                    confidence: 0.94,
                },
                AttributionEvidence::MatchesVoiceSignature {
                    signature_id: VoiceSignatureId::new(),
                    confidence: 0.91,
                },
            ],
        },
        SourceHypothesis {
            source_id: None,
            kind: SourceKind::Voice,
            range: TimeRange::new(
                TimePoint::from_millis(14_000),
                TimePoint::from_millis(15_000),
            ),
            confidence: 0.68,
            evidence: vec![AttributionEvidence::PitchContinuity { confidence: 0.65 }],
        },
    ];

    let overlaps = detect_overlaps(&hypotheses);

    let transcripts = vec![SourceAttributedTranscript {
        range: TimeRange::new(TimePoint::from_millis(14_000), TimePoint::from_millis(15_000)),
        source_hypothesis: hypotheses[1].clone(),
        source_label: SourceLabel::UnknownVoice { ordinal: 1 },
        text: "wait, what?".to_string(),
        transcript_confidence: 0.71,
        attribution_confidence: 0.62,
        overlap: None,
    }];

    let voice_count = VoiceCount {
        active_now: 2,
        recently_heard: 3,
        known: 1,
        unknown: 2,
        confidence: 0.74,
    };

    SoundscapeDebugView::from_components(&frame, voice_count, &hypotheses, &overlaps, &transcripts)
}
