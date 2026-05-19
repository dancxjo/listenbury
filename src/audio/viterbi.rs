//! Known-pronunciation Viterbi alignment scaffold.
//!
//! Given a known phone sequence and acoustic feature frames, runs a simple
//! left-to-right HMM Viterbi alignment and emits one [`SpanHypothesis`] per
//! phone.
//!
//! ## How it works
//!
//! 1. Feature frames that fall inside the word interval are selected.
//! 2. Each frame is scored against each phone state via a heuristic emission
//!    probability computed by [`crate::audio::phone_class::classify_frame`].
//! 3. Viterbi dynamic programming finds the best monotonic assignment of
//!    frames to phones.
//! 4. One [`SpanHypothesis`] is emitted per phone with timing derived from the
//!    assigned frames.
//!
//! When no feature frames fall in the word interval the aligner falls back to
//! proportional segmentation.

use serde_json::json;

use crate::audio::features::AcousticFeatureStream;
use crate::audio::hypothesis::{
    HypothesisSource, HypothesisStatus, SpanHypothesis, SpanHypothesisId, SpanHypothesisKind,
};
use crate::audio::phone_class::{classify_frame, CoarsePhoneClass};

// ---------------------------------------------------------------------------
// Phone state
// ---------------------------------------------------------------------------

/// A single phone state used as input to the Viterbi aligner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneState {
    /// ARPAbet or IPA symbol (e.g. `"TH"` or `"θ"`).
    pub symbol: String,
    /// Broad phone class expected at this position (e.g. `"fricative"`).
    pub phone_class: String,
}

impl PhoneState {
    pub fn new(symbol: impl Into<String>, phone_class: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            phone_class: phone_class.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run Viterbi forced alignment for `phones` over `features` within
/// `[word_start_ms, word_end_ms]`.
///
/// Returns one `SpanHypothesis` per phone covering the full word span.
pub fn viterbi_align_pronunciation(
    phones: &[PhoneState],
    word_start_ms: u64,
    word_end_ms: u64,
    features: &AcousticFeatureStream,
) -> Vec<SpanHypothesis> {
    if phones.is_empty() || word_end_ms <= word_start_ms {
        return Vec::new();
    }

    // Collect frames inside the word interval.
    let word_frames: Vec<&crate::audio::features::AcousticFeatureFrame> = features
        .frames
        .iter()
        .filter(|f| f.frame_start_ms >= word_start_ms && f.frame_end_ms <= word_end_ms)
        .collect();

    if word_frames.is_empty() {
        return proportional_fallback(phones, word_start_ms, word_end_ms);
    }

    let n_frames = word_frames.len();
    let n_phones = phones.len();

    // ---- Viterbi DP --------------------------------------------------------
    // dp[f][p] = best log-probability of reaching phone p at frame f.
    // bt[f][p] = phone index at frame f-1 on the best path to (f, p).
    let neg_inf: f32 = f32::NEG_INFINITY;
    let mut dp = vec![vec![neg_inf; n_phones]; n_frames];
    let mut bt = vec![vec![0usize; n_phones]; n_frames];

    let emit = |fi: usize, pi: usize| -> f32 {
        let frame = word_frames[fi];
        let (class, _) = classify_frame(frame);
        phone_class_match_score(class, &phones[pi].phone_class)
    };

    // Initialise frame 0.
    dp[0][0] = emit(0, 0).ln().max(-20.0);
    // Phones 1..n_phones cannot be reached at frame 0 (enforce left-to-right).
    // dp[0][p>0] stays at neg_inf.

    // Fill.
    for f in 1..n_frames {
        for p in 0..n_phones {
            // Stay in the same phone state.
            let stay = dp[f - 1][p];
            // Advance from the previous phone state.
            let advance = if p > 0 { dp[f - 1][p - 1] } else { neg_inf };
            let (best, from_p) = if advance >= stay {
                (advance, p.saturating_sub(1))
            } else {
                (stay, p)
            };
            dp[f][p] = if best > neg_inf {
                best + emit(f, p).ln().max(-20.0)
            } else {
                neg_inf
            };
            bt[f][p] = from_p;
        }
    }

    // Backtrace from the last frame, last phone.
    let mut phone_assignment = vec![0usize; n_frames];
    let mut p = n_phones - 1;
    for f in (0..n_frames).rev() {
        phone_assignment[f] = p;
        if f > 0 {
            p = bt[f][p];
        }
    }

    // Compute overall path score (geometric mean of per-frame emission scores).
    let path_score: f32 = {
        let total_log: f32 = (0..n_frames)
            .map(|f| emit(f, phone_assignment[f]).ln().max(-20.0))
            .sum();
        // Exponentiate the per-frame average to get a 0..1 score.
        (total_log / n_frames as f32).exp().clamp(0.0, 1.0)
    };

    // Pre-compute proportional alternative for each phone (used in provenance).
    let proportional_alts = proportional_fallback(phones, word_start_ms, word_end_ms);

    // ---- Build hypotheses --------------------------------------------------
    let mut hypotheses = Vec::with_capacity(n_phones);
    for phone_idx in 0..n_phones {
        let assigned_frames: Vec<usize> = phone_assignment
            .iter()
            .enumerate()
            .filter(|&(_, &ap)| ap == phone_idx)
            .map(|(f, _)| f)
            .collect();

        let (start_ms, end_ms) = if assigned_frames.is_empty() {
            let boundary = if phone_idx == 0 {
                word_start_ms
            } else {
                word_end_ms
            };
            (boundary, boundary)
        } else {
            let first = assigned_frames[0];
            let last = *assigned_frames.last().unwrap();
            (
                word_frames[first].frame_start_ms,
                word_frames[last].frame_end_ms,
            )
        };

        let score = if assigned_frames.is_empty() {
            0.30
        } else {
            assigned_frames
                .iter()
                .map(|&fi| emit(fi, phone_idx))
                .sum::<f32>()
                / assigned_frames.len() as f32
        };

        // Collect per-frame emission evidence strings.
        let emission_evidence: Vec<serde_json::Value> = assigned_frames
            .iter()
            .map(|&fi| {
                let frame = word_frames[fi];
                let (class, features) = classify_frame(frame);
                let match_score = phone_class_match_score(class, &phones[phone_idx].phone_class);
                json!({
                    "frame_start_ms": frame.frame_start_ms,
                    "detected_class": class.as_str(),
                    "features": features,
                    "match_score": match_score,
                })
            })
            .collect();

        // Collect conflict frames (detected class does not match expected).
        let conflicts: Vec<serde_json::Value> = assigned_frames
            .iter()
            .filter(|&&fi| {
                let frame = word_frames[fi];
                let (class, _) = classify_frame(frame);
                phone_class_match_score(class, &phones[phone_idx].phone_class) < 0.5
            })
            .map(|&fi| {
                let frame = word_frames[fi];
                let (class, _) = classify_frame(frame);
                json!({
                    "frame_start_ms": frame.frame_start_ms,
                    "detected_class": class.as_str(),
                    "expected_class": phones[phone_idx].phone_class,
                })
            })
            .collect();

        // Collect boundary evidence from the first and last assigned frames.
        let start_boundary_evidence = assigned_frames.first().map(|&fi| {
            let frame = word_frames[fi];
            json!({
                "frame_start_ms": frame.frame_start_ms,
                "rms_energy": frame.rms_energy,
                "spectral_flux": frame.spectral_flux,
                "zcr": frame.zero_crossing_rate,
            })
        });
        let end_boundary_evidence = assigned_frames.last().map(|&fi| {
            let frame = word_frames[fi];
            json!({
                "frame_end_ms": frame.frame_end_ms,
                "rms_energy": frame.rms_energy,
                "spectral_flux": frame.spectral_flux,
                "zcr": frame.zero_crossing_rate,
            })
        });

        // Proportional alternative boundary for this phone.
        let alternative_boundary = proportional_alts.get(phone_idx).map(|h| {
            json!({
                "start_ms": h.start_ms,
                "end_ms": h.end_ms,
                "method": "viterbi.proportional_fallback",
                "score": h.score,
            })
        });

        // Collect the features used for emission scoring from all assigned frames.
        let mut features_used: Vec<String> = assigned_frames
            .iter()
            .flat_map(|&fi| {
                let frame = word_frames[fi];
                let (_, features) = classify_frame(frame);
                features
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        features_used.insert(0, "viterbi.forced_alignment".to_string());
        features_used.push(format!("phone_class.{}", phones[phone_idx].phone_class));

        hypotheses.push(SpanHypothesis {
            id: SpanHypothesisId::new(),
            kind: SpanHypothesisKind::PronunciationAlignment,
            label: phones[phone_idx].symbol.clone(),
            start_ms,
            end_ms,
            score,
            confidence: score.clamp(0.0, 1.0),
            source: HypothesisSource::ViterbiAlignment,
            features_used,
            status: HypothesisStatus::Provisional,
            provenance: json!({
                "phone": phones[phone_idx].symbol,
                "phone_class": phones[phone_idx].phone_class,
                "method": "viterbi.fused_heuristic",
                "assigned_frames": assigned_frames.len(),
                "word_start_ms": word_start_ms,
                "word_end_ms": word_end_ms,
                "path_score": path_score,
                "emission_evidence": emission_evidence,
                "boundary_evidence": {
                    "start": start_boundary_evidence,
                    "end": end_boundary_evidence,
                },
                "conflicts": conflicts,
                "alternative_boundaries": alternative_boundary,
            }),
        });
    }
    hypotheses
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Heuristic emission score: how well does the detected coarse class match
/// the expected phone class string (0.0–1.0).
fn phone_class_match_score(detected: CoarsePhoneClass, expected: &str) -> f32 {
    if detected == CoarsePhoneClass::SilenceNoise {
        return match expected {
            "stop_closure" | "silence" => 0.70,
            _ => 0.15,
        };
    }
    let is_match = match expected {
        "vowel" | "diphthong" => detected == CoarsePhoneClass::VowelOrSonorant,
        "fricative" => detected == CoarsePhoneClass::Fricative,
        "stop" => {
            detected == CoarsePhoneClass::StopClosure || detected == CoarsePhoneClass::StopBurst
        }
        "nasal" => {
            detected == CoarsePhoneClass::Nasal
                || detected == CoarsePhoneClass::VowelOrSonorant
        }
        "approximant_liquid" => {
            detected == CoarsePhoneClass::ApproximantLiquid
                || detected == CoarsePhoneClass::VowelOrSonorant
        }
        "affricate" => {
            detected == CoarsePhoneClass::Fricative || detected == CoarsePhoneClass::StopBurst
        }
        _ => false,
    };
    if is_match { 0.72 } else { 0.20 }
}

/// Proportional fallback: divide the word interval evenly among phones.
fn proportional_fallback(
    phones: &[PhoneState],
    word_start_ms: u64,
    word_end_ms: u64,
) -> Vec<SpanHypothesis> {
    let duration = word_end_ms.saturating_sub(word_start_ms) as f32;
    let n = phones.len() as f32;
    phones
        .iter()
        .enumerate()
        .map(|(index, phone)| {
            let start_ms = word_start_ms + (index as f32 / n * duration).round() as u64;
            let end_ms = if index + 1 == phones.len() {
                word_end_ms
            } else {
                word_start_ms + ((index + 1) as f32 / n * duration).round() as u64
            };
            SpanHypothesis {
                id: SpanHypothesisId::new(),
                kind: SpanHypothesisKind::PronunciationAlignment,
                label: phone.symbol.clone(),
                start_ms,
                end_ms,
                score: 0.30,
                confidence: 0.30,
                source: HypothesisSource::ViterbiAlignment,
                features_used: vec!["viterbi.proportional_fallback".to_string()],
                status: HypothesisStatus::Provisional,
                provenance: json!({
                    "phone": phone.symbol,
                    "phone_class": phone.phone_class,
                    "method": "viterbi.proportional_fallback",
                    "fallback": true,
                    "emission_evidence": [],
                    "boundary_evidence": { "start": null, "end": null },
                    "conflicts": [],
                    "alternative_boundaries": null,
                    "path_score": 0.30,
                }),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::features::{AcousticFeatureFrame, AcousticFeatureStream};

    fn voiced_frame(start_ms: u64) -> AcousticFeatureFrame {
        AcousticFeatureFrame {
            frame_start_ms: start_ms,
            frame_end_ms: start_ms + 10,
            rms_energy: 0.06,
            peak_amplitude: 0.09,
            zero_crossing_rate: 0.05,
            spectral_flux: 0.04,
            low_band_energy_db: -12.0,
            high_band_energy_db: -28.0,
        }
    }

    fn fricative_frame(start_ms: u64) -> AcousticFeatureFrame {
        AcousticFeatureFrame {
            frame_start_ms: start_ms,
            frame_end_ms: start_ms + 10,
            rms_energy: 0.04,
            peak_amplitude: 0.06,
            zero_crossing_rate: 0.22,
            spectral_flux: 0.05,
            low_band_energy_db: -22.0,
            high_band_energy_db: -14.0,
        }
    }

    fn silence_frame(start_ms: u64) -> AcousticFeatureFrame {
        AcousticFeatureFrame {
            frame_start_ms: start_ms,
            frame_end_ms: start_ms + 10,
            rms_energy: 0.001,
            peak_amplitude: 0.002,
            zero_crossing_rate: 0.03,
            spectral_flux: 0.01,
            low_band_energy_db: -55.0,
            high_band_energy_db: -58.0,
        }
    }

    #[test]
    fn empty_phones_produces_no_hypotheses() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0)],
        };
        let hyps = viterbi_align_pronunciation(&[], 0, 100, &stream);
        assert!(hyps.is_empty());
    }

    #[test]
    fn zero_duration_word_produces_no_hypotheses() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0)],
        };
        let phones = vec![PhoneState::new("A", "vowel")];
        let hyps = viterbi_align_pronunciation(&phones, 100, 100, &stream);
        assert!(hyps.is_empty());
    }

    #[test]
    fn alignment_produces_one_hypothesis_per_phone() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                fricative_frame(5985),
                fricative_frame(5995),
                voiced_frame(6005),
                voiced_frame(6015),
                voiced_frame(6025),
                voiced_frame(6035),
            ],
        };
        let phones = vec![
            PhoneState::new("TH", "fricative"),
            PhoneState::new("R", "approximant_liquid"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 5985, 6045, &stream);
        assert_eq!(hyps.len(), 3);
    }

    #[test]
    fn alignment_is_monotonic() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                fricative_frame(1000),
                silence_frame(1010),
                voiced_frame(1020),
                voiced_frame(1030),
            ],
        };
        let phones = vec![
            PhoneState::new("S", "fricative"),
            PhoneState::new("AH", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 1000, 1040, &stream);
        assert_eq!(hyps.len(), 2);
        assert!(hyps[0].start_ms <= hyps[1].start_ms);
        assert!(hyps[0].end_ms <= hyps[1].end_ms);
    }

    #[test]
    fn full_word_span_is_covered() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0), voiced_frame(10), voiced_frame(20)],
        };
        let phones = vec![
            PhoneState::new("A", "vowel"),
            PhoneState::new("B", "stop"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 0, 30, &stream);
        assert_eq!(hyps.len(), 2);
        assert_eq!(hyps[0].start_ms, 0);
        assert_eq!(hyps.last().unwrap().end_ms, 30);
    }

    #[test]
    fn fallback_used_when_no_frames_in_word_interval() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(500)], // far outside the word interval
        };
        let phones = vec![
            PhoneState::new("TH", "fricative"),
            PhoneState::new("R", "approximant_liquid"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 0, 300, &stream);
        assert_eq!(hyps.len(), 3);
        for h in &hyps {
            assert!(h.provenance["fallback"] == true);
        }
    }

    #[test]
    fn hypotheses_carry_provenance() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0), voiced_frame(10)],
        };
        let phones = vec![PhoneState::new("AH", "vowel")];
        let hyps = viterbi_align_pronunciation(&phones, 0, 20, &stream);
        assert_eq!(hyps.len(), 1);
        let prov = &hyps[0].provenance;
        assert_eq!(prov["phone"], "AH");
        assert_eq!(prov["phone_class"], "vowel");
        assert_eq!(prov["word_start_ms"], 0);
        assert_eq!(prov["word_end_ms"], 20);
    }

    // ---- New acceptance-criteria tests ------------------------------------

    #[test]
    fn provenance_includes_method_field() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0), voiced_frame(10)],
        };
        let phones = vec![PhoneState::new("AH", "vowel")];
        let hyps = viterbi_align_pronunciation(&phones, 0, 20, &stream);
        let prov = &hyps[0].provenance;
        assert_eq!(prov["method"], "viterbi.fused_heuristic");
    }

    #[test]
    fn provenance_includes_path_score() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0), voiced_frame(10)],
        };
        let phones = vec![PhoneState::new("AH", "vowel")];
        let hyps = viterbi_align_pronunciation(&phones, 0, 20, &stream);
        let prov = &hyps[0].provenance;
        let path_score = prov["path_score"].as_f64().expect("path_score is a number");
        assert!(path_score >= 0.0 && path_score <= 1.0);
    }

    #[test]
    fn provenance_includes_emission_evidence() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0), voiced_frame(10)],
        };
        let phones = vec![PhoneState::new("AH", "vowel")];
        let hyps = viterbi_align_pronunciation(&phones, 0, 20, &stream);
        let prov = &hyps[0].provenance;
        let evidence = prov["emission_evidence"].as_array().expect("array");
        assert!(!evidence.is_empty(), "emission_evidence should be non-empty");
        // Each entry should have detected_class and match_score.
        let first = &evidence[0];
        assert!(first["detected_class"].is_string());
        assert!(first["match_score"].is_number());
    }

    #[test]
    fn provenance_includes_boundary_evidence() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(0), voiced_frame(10)],
        };
        let phones = vec![PhoneState::new("AH", "vowel")];
        let hyps = viterbi_align_pronunciation(&phones, 0, 20, &stream);
        let prov = &hyps[0].provenance;
        assert!(
            prov["boundary_evidence"].is_object(),
            "boundary_evidence should be an object"
        );
        let be = &prov["boundary_evidence"];
        assert!(!be["start"].is_null() || !be["end"].is_null());
    }

    #[test]
    fn provenance_includes_alternative_boundaries() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![fricative_frame(0), voiced_frame(10), voiced_frame(20)],
        };
        let phones = vec![
            PhoneState::new("S", "fricative"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 0, 30, &stream);
        assert_eq!(hyps.len(), 2);
        let prov = &hyps[0].provenance;
        // alternative_boundaries comes from proportional fallback.
        assert!(!prov["alternative_boundaries"].is_null());
    }

    #[test]
    fn fallback_method_is_proportional() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![voiced_frame(500)], // outside word interval
        };
        let phones = vec![
            PhoneState::new("TH", "fricative"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 0, 200, &stream);
        for h in &hyps {
            assert_eq!(h.provenance["method"], "viterbi.proportional_fallback");
        }
    }

    // ---- Vowel-heavy word --------------------------------------------------
    // Word "audio": [AO, D, IY, OW] — three vowels, one stop.

    #[test]
    fn vowel_heavy_word_aligns_correctly() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                voiced_frame(0),   // AO
                voiced_frame(10),  // AO
                voiced_frame(20),  // AO
                silence_frame(30), // D (stop closure)
                voiced_frame(40),  // IY
                voiced_frame(50),  // IY
                voiced_frame(60),  // OW
                voiced_frame(70),  // OW
            ],
        };
        let phones = vec![
            PhoneState::new("AO", "vowel"),
            PhoneState::new("D", "stop"),
            PhoneState::new("IY", "vowel"),
            PhoneState::new("OW", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 0, 80, &stream);
        assert_eq!(hyps.len(), 4);
        // All phones must be in order.
        for i in 1..hyps.len() {
            assert!(hyps[i].start_ms >= hyps[i - 1].start_ms, "phones out of order");
        }
        // First phone starts at word start.
        assert_eq!(hyps[0].start_ms, 0);
        // Last phone ends at word end.
        assert_eq!(hyps.last().unwrap().end_ms, 80);
    }

    // ---- Fricative + vowel (e.g. "see" = S IY) ----------------------------

    #[test]
    fn fricative_plus_vowel_word_aligns_correctly() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                fricative_frame(1000),
                fricative_frame(1010),
                fricative_frame(1020),
                voiced_frame(1030),
                voiced_frame(1040),
                voiced_frame(1050),
            ],
        };
        let phones = vec![
            PhoneState::new("S", "fricative"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 1000, 1060, &stream);
        assert_eq!(hyps.len(), 2);
        // Fricative should be assigned first, vowel second.
        assert!(hyps[0].start_ms < hyps[1].start_ms);
        // S should cover the first part; end of S <= start of IY.
        assert!(hyps[0].end_ms <= hyps[1].start_ms);
    }

    // ---- Stop + vowel (e.g. "key" = K IY) ---------------------------------

    fn stop_burst_frame(start_ms: u64) -> AcousticFeatureFrame {
        AcousticFeatureFrame {
            frame_start_ms: start_ms,
            frame_end_ms: start_ms + 10,
            rms_energy: 0.07,
            peak_amplitude: 0.10,
            zero_crossing_rate: 0.10,
            spectral_flux: 0.18, // high flux → stop_burst
            low_band_energy_db: -18.0,
            high_band_energy_db: -30.0,
        }
    }

    #[test]
    fn stop_plus_vowel_word_aligns_correctly() {
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                silence_frame(2000),    // K stop closure
                stop_burst_frame(2010), // K burst
                voiced_frame(2020),     // IY
                voiced_frame(2030),     // IY
                voiced_frame(2040),     // IY
            ],
        };
        let phones = vec![
            PhoneState::new("K", "stop"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 2000, 2050, &stream);
        assert_eq!(hyps.len(), 2);
        assert!(hyps[0].start_ms < hyps[1].start_ms);
        assert!(hyps[0].end_ms <= hyps[1].end_ms);
        assert_eq!(hyps.last().unwrap().end_ms, 2050);
    }

    // ---- Noisy / contradictory frames -------------------------------------
    // All frames look like silence even though we expect fricative + vowel.
    // The aligner records conflicts in provenance.

    #[test]
    fn contradictory_frames_record_conflicts_in_provenance() {
        // Use silence frames for a fricative + vowel sequence → all mismatch.
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                silence_frame(0),
                silence_frame(10),
                silence_frame(20),
                silence_frame(30),
            ],
        };
        let phones = vec![
            PhoneState::new("S", "fricative"),
            PhoneState::new("IY", "vowel"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 0, 40, &stream);
        assert_eq!(hyps.len(), 2);
        // Conflicts should be recorded because silence doesn't match fricative/vowel.
        let conflicts_0 = hyps[0].provenance["conflicts"]
            .as_array()
            .expect("conflicts array");
        let conflicts_1 = hyps[1].provenance["conflicts"]
            .as_array()
            .expect("conflicts array");
        // At least one of the phones should have conflicts.
        assert!(
            !conflicts_0.is_empty() || !conflicts_1.is_empty(),
            "expected at least one conflict entry for mismatched frames"
        );
    }

    // ---- Monotonicity under any frame sequence ----------------------------

    #[test]
    fn monotonicity_enforced_for_three_phone_sequence() {
        // Intentionally scrambled frame types to stress-test the DP.
        let stream = AcousticFeatureStream {
            hop_ms: 10.0,
            frames: vec![
                voiced_frame(500),
                fricative_frame(510),
                silence_frame(520),
                voiced_frame(530),
                fricative_frame(540),
                silence_frame(550),
            ],
        };
        let phones = vec![
            PhoneState::new("F", "fricative"),
            PhoneState::new("AH", "vowel"),
            PhoneState::new("N", "nasal"),
        ];
        let hyps = viterbi_align_pronunciation(&phones, 500, 560, &stream);
        assert_eq!(hyps.len(), 3);
        for i in 1..hyps.len() {
            assert!(
                hyps[i].start_ms >= hyps[i - 1].start_ms,
                "monotonicity violated: hyp[{}].start_ms={} < hyp[{}].start_ms={}",
                i,
                hyps[i].start_ms,
                i - 1,
                hyps[i - 1].start_ms
            );
        }
    }
}
