use super::*;
use crate::audio::hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisKind,
};
use serde_json::json;

fn make_word_candidate(label: &str, start_ms: u64, end_ms: u64, confidence: f32) -> SpanHypothesis {
    SpanHypothesis::new(
        SpanHypothesisKind::WordCandidate,
        label,
        start_ms,
        end_ms,
        confidence,
        confidence,
        HypothesisSource::Manual,
        vec![],
        json!(null),
    )
}

fn make_word_candidate_with_provenance(
    label: &str,
    start_ms: u64,
    end_ms: u64,
    confidence: f32,
    provenance: serde_json::Value,
) -> SpanHypothesis {
    let mut hypothesis = make_word_candidate(label, start_ms, end_ms, confidence);
    hypothesis.provenance = provenance;
    hypothesis
}

fn make_boundary(
    label: &str,
    ms: u64,
    confidence: f32,
    source: HypothesisSource,
    features: Vec<String>,
) -> SpanHypothesis {
    SpanHypothesis::new(
        SpanHypothesisKind::SpeechBoundary,
        label,
        ms,
        ms,
        confidence,
        confidence,
        source,
        features,
        json!(null),
    )
}

#[test]
fn competing_word_candidates_coexist_in_lattice() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
    lattice.add(make_word_candidate("texting", 1000, 1300, 0.19));
    lattice.add(make_word_candidate("test in", 1000, 1300, 0.07));
    assert_eq!(lattice.active_hypotheses().len(), 3);
    assert_eq!(lattice.all_hypotheses().len(), 3);
}

#[test]
fn lattice_preserves_all_hypotheses_after_revision() {
    let mut lattice = HypothesisLattice::new();
    let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
    let h1_id = h1.id.clone();
    lattice.add(h1);

    let h2 = make_word_candidate("texting", 1000, 1300, 0.85);
    lattice.revise(&h1_id, h2);

    assert_eq!(lattice.all_hypotheses().len(), 2);
    let old = lattice
        .all_hypotheses()
        .iter()
        .find(|h| h.id == h1_id)
        .unwrap();
    assert_eq!(old.status, HypothesisStatus::Revised);
}

#[test]
fn active_hypotheses_excludes_revised() {
    let mut lattice = HypothesisLattice::new();
    let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
    let h1_id = h1.id.clone();
    lattice.add(h1);
    let h2 = make_word_candidate("texting", 1000, 1300, 0.85);
    lattice.revise(&h1_id, h2);

    let active = lattice.active_hypotheses();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].label, "texting");
}

#[test]
fn revision_adds_revision_of_edge() {
    let mut lattice = HypothesisLattice::new();
    let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
    let h1_id = h1.id.clone();
    lattice.add(h1);
    let h2 = make_word_candidate("texting", 1000, 1300, 0.85);
    let h2_id = lattice.revise(&h1_id, h2);

    let edges = lattice.edges_from(&h2_id);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].kind, HypothesisEdgeKind::RevisionOf);
    assert_eq!(edges[0].to, h1_id);
}

#[test]
fn hypotheses_can_support_each_other() {
    let mut lattice = HypothesisLattice::new();
    let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
    let h2 = make_boundary(
        "speech_start",
        1000,
        0.65,
        HypothesisSource::EndpointDetector,
        vec!["energy.onset".to_string()],
    );
    let h1_id = h1.id.clone();
    let h2_id = h2.id.clone();
    lattice.add(h1);
    lattice.add(h2);
    lattice.connect(
        h2_id.clone(),
        h1_id.clone(),
        HypothesisEdgeKind::Supports,
        0.8,
    );
    let edges = lattice.edges_from(&h2_id);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].kind, HypothesisEdgeKind::Supports);
}

#[test]
fn conflicting_boundary_hypotheses_are_distinguishable() {
    let mut lattice = HypothesisLattice::new();
    let b1 = make_boundary(
        "speech_start_asr",
        1000,
        0.8,
        HypothesisSource::Manual,
        vec!["asr.timing".to_string()],
    );
    let b2 = make_boundary(
        "speech_start_energy",
        1050,
        0.7,
        HypothesisSource::EndpointDetector,
        vec!["energy.onset".to_string()],
    );
    let b1_id = b1.id.clone();
    let b2_id = b2.id.clone();
    lattice.add(b1);
    lattice.add(b2);
    lattice.connect(
        b1_id.clone(),
        b2_id.clone(),
        HypothesisEdgeKind::Contradicts,
        1.0,
    );
    assert_eq!(lattice.active_hypotheses().len(), 2);
    let result = fuse_hypotheses(&lattice, &[], &FusionWeights::default()).unwrap();
    assert!(!result.conflicting_ids.is_empty());
}

#[test]
fn fusion_resolves_highest_confidence_candidate() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
    lattice.add(make_word_candidate("texting", 1000, 1300, 0.19));
    lattice.add(make_word_candidate("test in", 1000, 1300, 0.07));

    let result = fuse_hypotheses(&lattice, &[], &FusionWeights::default()).unwrap();
    assert_eq!(result.resolved.label, "testing");
    assert!(result.confidence > 0.5);
}

#[test]
fn fusion_boosted_by_asr_and_energy_evidence_can_flip_winner() {
    let mut lattice = HypothesisLattice::new();
    let h_low = make_word_candidate("texting", 1000, 1300, 0.19);
    let h_high = make_word_candidate("testing", 1000, 1300, 0.72);
    let low_id = h_low.id.clone();
    lattice.add(h_low);
    lattice.add(h_high);

    let evidence = vec![(
        low_id,
        FusionInput {
            asr_confidence: Some(0.95),
            energy_alignment_quality: Some(0.90),
            mechanical_recognizer_score: Some(0.88),
            ..Default::default()
        },
    )];
    let result = fuse_hypotheses(&lattice, &evidence, &FusionWeights::default()).unwrap();
    assert_eq!(result.resolved.label, "texting");
}

#[test]
fn fusion_classifies_conflicting_and_supporting_correctly() {
    let mut lattice = HypothesisLattice::new();
    let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
    let h2 = make_word_candidate("texting", 1000, 1300, 0.19);
    let h2_id = h2.id.clone();
    let h1_id = h1.id.clone();
    lattice.add(h1);
    lattice.add(h2);
    lattice.connect(
        h2_id.clone(),
        h1_id.clone(),
        HypothesisEdgeKind::Contradicts,
        1.0,
    );

    let result = fuse_hypotheses(&lattice, &[], &FusionWeights::default()).unwrap();
    assert!(result.conflicting_ids.contains(&h2_id));
    assert!(!result.conflicting_summary.contains("no conflicting"));
}

#[test]
fn fusion_result_preserves_provenance_json() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
    let result = fuse_hypotheses(&lattice, &[], &FusionWeights::default()).unwrap();
    assert_eq!(result.provenance["fusion"], "first_pass_weighted_average");
    assert!(result.provenance["candidate_count"].as_u64().unwrap() >= 1);
}

#[test]
fn fusion_on_empty_lattice_returns_none() {
    let lattice = HypothesisLattice::new();
    assert!(fuse_hypotheses(&lattice, &[], &FusionWeights::default()).is_none());
}

#[test]
fn fusion_input_weighted_confidence_uses_all_signals() {
    let input = FusionInput {
        asr_confidence: Some(1.0),
        energy_alignment_quality: Some(1.0),
        phone_segmentation_agreement: Some(1.0),
        pronunciation_fit: Some(1.0),
        spectral_evidence: Some(1.0),
        prosody_consistency: Some(1.0),
        timing_coherence: Some(1.0),
        mechanical_recognizer_score: Some(1.0),
        visual_speech_evidence: Some(1.0),
    };
    assert!((input.weighted_confidence() - 1.0).abs() < 1e-5);
}

#[test]
fn fusion_input_zero_signals_returns_zero_confidence() {
    let input = FusionInput::default();
    assert_eq!(input.weighted_confidence(), 0.0);
}

#[test]
fn fusion_result_serializes_to_json() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate("testing", 1000, 1300, 0.72));
    let result = fuse_hypotheses(&lattice, &[], &FusionWeights::default()).unwrap();
    let json = serde_json::to_string(&result).expect("serialise");
    assert!(json.contains("resolved"));
    assert!(json.contains("confidence"));
    assert!(json.contains("provenance"));
}

#[test]
fn speech_hypothesis_engine_uses_multiple_default_evidence_sources() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate_with_provenance(
        "testing",
        1000,
        1300,
        0.40,
        json!({
            "asr_confidence": 0.91,
            "transcript_stability": 0.88,
            "visual_speech_evidence": 0.82,
        }),
    ));
    lattice.add(make_boundary(
        "speech_start",
        1000,
        0.72,
        HypothesisSource::EndpointDetector,
        vec!["energy.onset".to_string()],
    ));
    lattice.add(SpanHypothesis::new(
        SpanHypothesisKind::PhoneClassCandidate,
        "fricative",
        1020,
        1060,
        0.68,
        0.66,
        HypothesisSource::PhoneClassifier,
        vec!["spectral_flux".to_string()],
        json!(null),
    ));

    let engine = SpeechHypothesisEngine::with_default_sources();
    let output = engine.fuse(&lattice).expect("fused");

    let unique_sources: std::collections::HashSet<&str> = output
        .evidence_trace
        .iter()
        .map(|entry| entry.source.as_str())
        .collect();
    assert!(unique_sources.len() >= 3);
}

#[test]
fn speech_hypothesis_engine_applies_stability_and_rescoring() {
    let mut lattice = HypothesisLattice::new();
    let low_acoustic_high_stability = make_word_candidate_with_provenance(
        "texting",
        1000,
        1300,
        0.25,
        json!({
            "asr_confidence": 0.94,
            "transcript_stability": 0.90,
            "stable_prefix_ratio": 0.89,
            "visual_speech_evidence": 0.87,
        }),
    );
    let high_acoustic_low_stability = make_word_candidate_with_provenance(
        "testing",
        1000,
        1300,
        0.78,
        json!({
            "asr_confidence": 0.25,
            "transcript_stability": 0.20,
        }),
    );
    let low_id = low_acoustic_high_stability.id.clone();
    let high_id = high_acoustic_low_stability.id.clone();
    lattice.add(low_acoustic_high_stability);
    lattice.add(high_acoustic_low_stability);
    lattice.connect(
        low_id.clone(),
        high_id.clone(),
        HypothesisEdgeKind::Contradicts,
        1.0,
    );

    let engine = SpeechHypothesisEngine::with_default_sources();
    let output = engine.fuse(&lattice).expect("fused");

    assert_eq!(output.fusion.resolved.label, "texting");
    assert!(output.stable_span_ids.contains(&low_id));
    assert!(output.revisable_span_ids.contains(&high_id));

    let stable = output
        .lattice
        .all_hypotheses()
        .iter()
        .find(|h| h.id == low_id)
        .expect("stable span");
    assert_eq!(stable.status, HypothesisStatus::Confirmed);
}

#[test]
fn speech_hypothesis_fusion_output_serializes_for_debugging() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate_with_provenance(
        "testing",
        1000,
        1300,
        0.55,
        json!({
            "asr_confidence": 0.85,
            "transcript_stability": 0.83,
        }),
    ));

    let engine = SpeechHypothesisEngine::with_default_sources();
    let output = engine.fuse(&lattice).expect("fused");
    let encoded = serde_json::to_string(&output).expect("serialise");
    assert!(encoded.contains("stable_span_ids"));
    assert!(encoded.contains("evidence_trace"));
    assert!(encoded.contains("fusion"));
}

#[test]
fn lattice_serializes_and_deserializes_round_trip() {
    let mut lattice = HypothesisLattice::new();
    let h1 = make_word_candidate("testing", 1000, 1300, 0.72);
    let h2 = make_word_candidate("texting", 1000, 1300, 0.19);
    let h1_id = h1.id.clone();
    let h2_id = h2.id.clone();
    lattice.add(h1);
    lattice.add(h2);
    lattice.connect(h1_id, h2_id, HypothesisEdgeKind::Contradicts, 1.0);

    let json = serde_json::to_string(&lattice).expect("serialise");
    let restored: HypothesisLattice = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(restored.hypotheses.len(), 2);
    assert_eq!(restored.edges.len(), 1);
    assert_eq!(restored.edges[0].kind, HypothesisEdgeKind::Contradicts);
}

// ── FusionProfile / FusionWeights tests ──────────────────────────────────────

/// Verify that the Default profile exactly reproduces the original heuristic
/// weights that were previously embedded as numeric literals.
#[test]
fn fusion_profile_default_matches_original_weights() {
    let w = FusionWeights::from(FusionProfile::Default);
    assert_eq!(w.asr_confidence, 3.0);
    assert_eq!(w.energy_alignment_quality, 1.5);
    assert_eq!(w.phone_segmentation_agreement, 1.0);
    assert_eq!(w.pronunciation_fit, 1.0);
    assert_eq!(w.spectral_evidence, 0.75);
    assert_eq!(w.prosody_consistency, 0.5);
    assert_eq!(w.timing_coherence, 1.25);
    assert_eq!(w.mechanical_recognizer_score, 1.0);
    assert_eq!(w.visual_speech_evidence, 0.9);
    assert_eq!(w.external_evidence_blend, 3.0);
}

/// FusionWeights::default() must produce the same values as FusionProfile::Default.
#[test]
fn fusion_weights_default_impl_matches_profile_default() {
    assert_eq!(
        FusionWeights::default(),
        FusionWeights::from(FusionProfile::Default)
    );
}

/// When a hypothesis has high ASR confidence but low acoustic signals and a
/// competing hypothesis has the inverse, the Default profile (ASR weight 3.0)
/// picks the ASR-favoured candidate while the AcousticHeavy profile (ASR
/// weight 1.0, energy weight 3.0) picks the acoustic-favoured candidate.
///
/// This test proves that profile changes can alter the winning hypothesis.
#[test]
fn fusion_profile_acoustic_heavy_overrides_default_winner() {
    // h_asr  : moderate base confidence, strong ASR signal, weak energy signal
    // h_acoustic: moderate base confidence, weak ASR signal, strong energy signal
    let h_asr = make_word_candidate("whisper-word", 1000, 1300, 0.5);
    let h_acoustic = make_word_candidate("acoustic-word", 1000, 1300, 0.5);
    let asr_id = h_asr.id.clone();
    let acoustic_id = h_acoustic.id.clone();

    let mut lattice = HypothesisLattice::new();
    lattice.add(h_asr);
    lattice.add(h_acoustic);

    let evidence = vec![
        (
            asr_id,
            FusionInput {
                asr_confidence: Some(0.9),
                energy_alignment_quality: Some(0.2),
                ..Default::default()
            },
        ),
        (
            acoustic_id,
            FusionInput {
                asr_confidence: Some(0.2),
                energy_alignment_quality: Some(0.9),
                ..Default::default()
            },
        ),
    ];

    // Default profile: ASR weight dominates → "whisper-word" wins.
    let default_result = fuse_hypotheses(
        &lattice,
        &evidence,
        &FusionWeights::from(FusionProfile::Default),
    )
    .unwrap();
    assert_eq!(
        default_result.resolved.label, "whisper-word",
        "Default profile should favour the high-ASR hypothesis"
    );

    // AcousticHeavy profile: energy/acoustic weight dominates → "acoustic-word" wins.
    let acoustic_result = fuse_hypotheses(
        &lattice,
        &evidence,
        &FusionWeights::from(FusionProfile::AcousticHeavy),
    )
    .unwrap();
    assert_eq!(
        acoustic_result.resolved.label, "acoustic-word",
        "AcousticHeavy profile should favour the high-energy hypothesis"
    );
}

/// The Realtime profile must also produce a valid fusion result and must differ
/// from the Default profile when ASR and energy signals conflict.
#[test]
fn fusion_profile_realtime_favours_energy_over_asr() {
    let h_asr = make_word_candidate("asr-word", 1000, 1300, 0.5);
    let h_energy = make_word_candidate("energy-word", 1000, 1300, 0.5);
    let asr_id = h_asr.id.clone();
    let energy_id = h_energy.id.clone();

    let mut lattice = HypothesisLattice::new();
    lattice.add(h_asr);
    lattice.add(h_energy);

    let evidence = vec![
        (
            asr_id,
            FusionInput {
                asr_confidence: Some(0.9),
                energy_alignment_quality: Some(0.1),
                timing_coherence: Some(0.1),
                ..Default::default()
            },
        ),
        (
            energy_id,
            FusionInput {
                asr_confidence: Some(0.1),
                energy_alignment_quality: Some(0.9),
                timing_coherence: Some(0.9),
                ..Default::default()
            },
        ),
    ];

    // Default profile: ASR weight (3.0) beats energy (1.5) + timing (1.25) → asr-word wins.
    let default_result = fuse_hypotheses(
        &lattice,
        &evidence,
        &FusionWeights::from(FusionProfile::Default),
    )
    .unwrap();
    assert_eq!(default_result.resolved.label, "asr-word");

    // Realtime profile: energy (2.5) + timing (2.0) beats ASR (1.5) → energy-word wins.
    let realtime_result = fuse_hypotheses(
        &lattice,
        &evidence,
        &FusionWeights::from(FusionProfile::Realtime),
    )
    .unwrap();
    assert_eq!(realtime_result.resolved.label, "energy-word");
}

/// Verify that SpeechHypothesisEngine::with_profile() stores the expected
/// weights and that with_profile(Default) == with_default_sources().
#[test]
fn speech_hypothesis_engine_with_profile_stores_weights() {
    let default_engine = SpeechHypothesisEngine::with_default_sources();
    let profile_engine = SpeechHypothesisEngine::with_profile(FusionProfile::Default);
    assert_eq!(
        default_engine.weights(),
        profile_engine.weights(),
        "with_profile(Default) must produce the same weights as with_default_sources()"
    );

    let acoustic_engine = SpeechHypothesisEngine::with_profile(FusionProfile::AcousticHeavy);
    assert_ne!(
        acoustic_engine.weights(),
        default_engine.weights(),
        "AcousticHeavy weights must differ from Default weights"
    );
}

/// Verify that set_weights() is reflected in subsequent fusions.
#[test]
fn speech_hypothesis_engine_set_weights_changes_profile() {
    let mut engine = SpeechHypothesisEngine::with_default_sources();
    engine.set_weights(FusionWeights::from(FusionProfile::VisualAssist));
    assert_eq!(
        engine.weights(),
        &FusionWeights::from(FusionProfile::VisualAssist)
    );
}

/// All five named profiles must each produce a valid (non-None) fusion result
/// from the same lattice.
#[test]
fn all_fusion_profiles_produce_valid_results() {
    let mut lattice = HypothesisLattice::new();
    lattice.add(make_word_candidate("alpha", 1000, 1300, 0.6));
    lattice.add(make_word_candidate("beta", 1000, 1300, 0.4));

    let profiles = [
        FusionProfile::Default,
        FusionProfile::Conservative,
        FusionProfile::Realtime,
        FusionProfile::AcousticHeavy,
        FusionProfile::VisualAssist,
    ];

    for profile in profiles {
        let weights = FusionWeights::from(profile);
        let result = fuse_hypotheses(&lattice, &[], &weights);
        assert!(
            result.is_some(),
            "Profile {:?} should produce a fusion result",
            profile
        );
    }
}
