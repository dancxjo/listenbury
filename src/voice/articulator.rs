//! Articulator pass: sung phrase → vocal gesture plan.
//!
//! This module is the bridge from a semantic [`SungPhrase`] (produced by the
//! prosody layer) to a renderer-neutral [`ArticulatorPlan`] that downstream
//! voice passes — such as the coarticulation pass — can consume.
//!
//! # What this is not
//!
//! - It is **not** final waveform synthesis.  No audio data is produced here.
//! - It is not a neural vocoder, Piper/eSpeak integration, or ONNX graph pass.
//! - It does not perform high-quality natural singing coarticulation; that is
//!   the job of the [`coarticulation`](super::coarticulation) module.
//!
//! # What this does
//!
//! 1. Walks every syllable in the phrase, assigning each phone to its
//!    structural role: onset consonant attack, pitch-bearing nucleus, or coda
//!    consonant release.
//! 2. Marks consonants as unvoiced when their IPA label is in the well-known
//!    set of unvoiced phones; all vowels and voiced consonants remain voiced.
//! 3. Derives a phrase-level [`PitchCurve`] from the note targets already
//!    embedded in the syllables, using linear interpolation between control
//!    points.
//! 4. Constructs an [`EnergyCurve`] from per-syllable note velocities,
//!    representing intensity separately from pitch.
//!
//! # Output
//!
//! The primary output type is [`ArticulatorPlan`], which bundles:
//! - the ordered [`VocalGesturePlan`] (phone gestures ready for the
//!   coarticulation pass),
//! - an optional [`PitchCurve`] (present when any syllable carries a note
//!   target), and
//! - an [`EnergyCurve`] representing phrase energy/intensity over time.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::prosody::pitch_curve::{Interpolation, PitchCurve};
use crate::prosody::singing::SungPhrase;
use crate::voice::coarticulation::{PhoneGesture, PhoneRole, VocalGesturePlan};

// ─── EnergyCurve ─────────────────────────────────────────────────────────────

/// A single energy control point in an [`EnergyCurve`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnergyPoint {
    /// Timestamp from phrase start.
    pub t: Duration,
    /// Normalised energy level in `[0.0, 1.0]` where `1.0` is maximum.
    pub level: f32,
}

impl EnergyPoint {
    /// Construct an energy point.
    #[inline]
    pub fn new(t: Duration, level: f32) -> Self {
        Self { t, level }
    }
}

/// Phrase energy/intensity trajectory represented as time-ordered control
/// points.
///
/// Energy is modelled separately from pitch so that renderers can shape
/// loudness independently (e.g. a soft high note or a loud low note).  The
/// initial implementation uses a step function: energy jumps to the new level
/// at each control point and holds until the next one.
///
/// An empty curve is valid and represents a phrase with no dynamic information
/// (renderers may substitute a default level).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyCurve {
    /// Energy control points in time order.
    pub points: Vec<EnergyPoint>,
}

impl EnergyCurve {
    /// Construct a curve from a list of control points.
    ///
    /// Points are sorted by timestamp on construction.
    pub fn new(mut points: Vec<EnergyPoint>) -> Self {
        points.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        Self { points }
    }

    /// Construct a flat curve at the given level over a single span.
    pub fn flat(onset: Duration, level: f32) -> Self {
        Self {
            points: vec![EnergyPoint::new(onset, level)],
        }
    }

    /// An empty curve carrying no dynamic information.
    pub fn empty() -> Self {
        Self { points: Vec::new() }
    }

    /// Sample the energy level at time `t` using a step function (hold the
    /// most recent control point's level).
    ///
    /// Returns `0.5` when the curve is empty or when `t` precedes all control
    /// points.
    pub fn sample(&self, t: Duration) -> f32 {
        if self.points.is_empty() {
            return 0.5;
        }
        // Find the last point whose timestamp ≤ t.
        let level = self
            .points
            .iter()
            .rev()
            .find(|p| p.t <= t)
            .map(|p| p.level)
            .unwrap_or(self.points[0].level);
        level
    }
}

// ─── ArticulatorPlan ─────────────────────────────────────────────────────────

/// The output of the articulator pass: a renderer-neutral plan derived from a
/// [`SungPhrase`].
///
/// `ArticulatorPlan` is designed to feed directly into the coarticulation pass
/// ([`crate::voice::coarticulate`]) or into a future voice renderer.  It
/// intentionally carries no audio synthesis state.
///
/// ## Representation
///
/// - **`gestures`** – the ordered sequence of phone gestures, each tagged with
///   its syllable-structural role (onset / nucleus / coda) and voicing status.
///   Onset and coda consonants represent attack and release material; nucleus
///   phones are the pitch-bearing core of the syllable.
/// - **`pitch_curve`** – a phrase-level pitch trajectory derived from the note
///   targets embedded in the syllables.  `None` when no syllable carries a note
///   target (e.g. spoken phrases or phrases without musical intent).
/// - **`energy_curve`** – phrase energy/intensity as a step curve over time,
///   constructed from per-syllable velocity annotations.
#[derive(Debug, Clone, PartialEq)]
pub struct ArticulatorPlan {
    /// Ordered phone gestures produced by the articulator.
    ///
    /// These are the raw inputs for the coarticulation pass and are **not** yet
    /// final: boundaries, timing micro-adjustments, and pitch-activity flags
    /// may be refined by [`crate::voice::coarticulate`].
    pub gestures: VocalGesturePlan,
    /// Phrase-level pitch trajectory.
    ///
    /// `None` when no syllable in the phrase carries a [`NoteTarget`].
    ///
    /// [`NoteTarget`]: crate::prosody::note_target::NoteTarget
    pub pitch_curve: Option<PitchCurve>,
    /// Phrase energy/intensity curve, separate from pitch.
    pub energy_curve: EnergyCurve,
}

// ─── Voicing heuristic ───────────────────────────────────────────────────────

/// Determine whether an IPA phone label represents a voiced sound.
///
/// This is a best-effort static heuristic based on the IPA chart.  It returns
/// `false` for the canonical unvoiced consonants and `true` for everything
/// else (vowels, voiced consonants, syllabic sonorants).
///
/// Phones not in the unvoiced set are assumed voiced so that novel or
/// composite phones degrade gracefully.
pub fn is_phone_voiced(ipa: &str) -> bool {
    // Well-known unvoiced consonant IPA symbols (including common digraphs).
    const UNVOICED: &[&str] = &[
        "p", "t", "k", "f", "s", "ʃ", "θ", "h", "x", "ç", "ʔ",
        // affricates
        "tʃ", "ts", "pf", "t͡ʃ", "t͡s",
        // aspirated stops (X-SAMPA / borrowed representations)
        "pʰ", "tʰ", "kʰ",
    ];
    !UNVOICED.contains(&ipa)
}

// ─── Articulator pass ────────────────────────────────────────────────────────

/// Convert a [`SungPhrase`] into an [`ArticulatorPlan`].
///
/// This is the primary entry point of the articulator module.  The conversion
/// is deterministic and allocation-only: no randomness, no audio I/O, no ML
/// inference.
///
/// # Phone gestures
///
/// Each phone in every syllable is emitted as a [`PhoneGesture`]:
///
/// - Phones in the **onset span** become [`PhoneRole::Onset`] gestures;
///   their voicing is inferred from the IPA label.
/// - Phones in the **nucleus span** become [`PhoneRole::Nucleus`] gestures
///   and are always marked voiced (nuclei are inherently pitch-bearing).
/// - Phones in the **coda span** become [`PhoneRole::Coda`] gestures;
///   their voicing is again inferred from the IPA label.
///
/// All gestures are emitted in syllable order, onset → nucleus → coda within
/// each syllable, syllables in phrase order.
///
/// # Pitch curve
///
/// Derived via [`SungPhrase::phrase_pitch_curve`] with linear interpolation.
/// `None` when no syllable carries a note target.
///
/// # Energy curve
///
/// One [`EnergyPoint`] is emitted per syllable that carries a note with a
/// velocity annotation.  The energy level is the normalised velocity
/// (`0.0..=1.0`).  Syllables without note targets do not contribute points;
/// when no syllable has velocity data the curve is empty.
pub fn articulate(phrase: &SungPhrase) -> ArticulatorPlan {
    let mut phone_gestures: Vec<PhoneGesture> = Vec::new();
    let mut energy_points: Vec<EnergyPoint> = Vec::new();

    for syllable in &phrase.syllables {
        // ── Collect phone gestures ────────────────────────────────────────
        let phones = &syllable.phones;

        // Onset phones (attack consonants).
        for idx in syllable.onset.start..syllable.onset.end {
            if let Some(tpr) = phones.get(idx) {
                phone_gestures.push(PhoneGesture {
                    phone: tpr.phone.ipa.clone(),
                    onset_ms: tpr.start.millis,
                    duration_ms: tpr.end.millis.saturating_sub(tpr.start.millis),
                    is_voiced: is_phone_voiced(&tpr.phone.ipa),
                    role: PhoneRole::Onset,
                    is_legato_context: false,
                });
            }
        }

        // Nucleus phones (pitch-bearing vowel material).
        for idx in syllable.nucleus.start..syllable.nucleus.end {
            if let Some(tpr) = phones.get(idx) {
                phone_gestures.push(PhoneGesture {
                    phone: tpr.phone.ipa.clone(),
                    onset_ms: tpr.start.millis,
                    duration_ms: tpr.end.millis.saturating_sub(tpr.start.millis),
                    // Nuclei are always voiced: they are the pitch-bearing core.
                    is_voiced: true,
                    role: PhoneRole::Nucleus,
                    is_legato_context: false,
                });
            }
        }

        // Coda phones (release consonants).
        for idx in syllable.coda.start..syllable.coda.end {
            if let Some(tpr) = phones.get(idx) {
                phone_gestures.push(PhoneGesture {
                    phone: tpr.phone.ipa.clone(),
                    onset_ms: tpr.start.millis,
                    duration_ms: tpr.end.millis.saturating_sub(tpr.start.millis),
                    is_voiced: is_phone_voiced(&tpr.phone.ipa),
                    role: PhoneRole::Coda,
                    is_legato_context: false,
                });
            }
        }

        // ── Energy point from note velocity ───────────────────────────────
        if let Some(note) = &syllable.note {
            let t = Duration::from_millis(note.onset.millis);
            let level = note.velocity.as_f32();
            energy_points.push(EnergyPoint::new(t, level));
        }
    }

    let gestures = VocalGesturePlan {
        gestures: phone_gestures,
    };
    let pitch_curve = phrase.phrase_pitch_curve(Interpolation::Linear);
    let energy_curve = EnergyCurve::new(energy_points);

    ArticulatorPlan {
        gestures,
        pitch_curve,
        energy_curve,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;
    use crate::prosody::note_target::{
        MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
    };
    use crate::prosody::syllable::{PhoneSpan, SungSyllable, TimedPhoneRef};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn timed(ipa: &str, start_ms: u64, end_ms: u64) -> TimedPhoneRef {
        TimedPhoneRef::new(
            Phone::new_ipa(ipa),
            TimePoint::from_millis(start_ms),
            TimePoint::from_millis(end_ms),
        )
        .expect("valid timed phone")
    }

    fn note(midi: u8, onset_ms: u64, duration_ms: u64) -> NoteTarget {
        NoteTarget {
            pitch: PitchTarget::new(MidiNote::new(midi).unwrap()),
            onset: TimePoint::from_millis(onset_ms),
            duration: NoteDuration::from_millis(duration_ms),
            velocity: Velocity::mezzo_forte(),
            articulation: NoteArticulation::Neutral,
        }
    }

    /// Build the "hel" syllable (h=onset, ɛ=nucleus, l=coda) from 0..240 ms.
    fn hel_syllable(with_note: bool) -> SungSyllable {
        SungSyllable::new(
            "hel",
            vec![timed("h", 0, 30), timed("ɛ", 30, 180), timed("l", 180, 240)],
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 3).unwrap(),
            None,
            if with_note { Some(note(60, 0, 240)) } else { None },
        )
        .unwrap()
    }

    /// Build the "lo" syllable (l=onset, oʊ=nucleus, empty coda) from 240..490 ms.
    fn lo_syllable(with_note: bool) -> SungSyllable {
        SungSyllable::new(
            "lo",
            vec![timed("l", 240, 270), timed("oʊ", 270, 490)],
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 2).unwrap(),
            None,
            if with_note { Some(note(67, 240, 250)) } else { None },
        )
        .unwrap()
    }

    fn hello_phrase(with_notes: bool) -> SungPhrase {
        let mut phrase = SungPhrase::new();
        phrase.push(hel_syllable(with_notes)).unwrap();
        phrase.push(lo_syllable(with_notes)).unwrap();
        phrase
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// "hel" + "lo" produces ordered phone gestures.
    #[test]
    fn hello_phrase_produces_ordered_phone_gestures() {
        let plan = articulate(&hello_phrase(false));
        let gestures = &plan.gestures.gestures;

        // Expect: h (onset), ɛ (nucleus), l (coda), l (onset), oʊ (nucleus)
        assert_eq!(gestures.len(), 5, "should produce 5 phone gestures");

        // Gestures must be in strictly non-decreasing onset order.
        for window in gestures.windows(2) {
            assert!(
                window[0].onset_ms <= window[1].onset_ms,
                "gesture '{}' at {} ms must precede '{}' at {} ms",
                window[0].phone,
                window[0].onset_ms,
                window[1].phone,
                window[1].onset_ms,
            );
        }

        // Phones are in the expected order.
        let labels: Vec<&str> = gestures.iter().map(|g| g.phone.as_str()).collect();
        assert_eq!(labels, vec!["h", "ɛ", "l", "l", "oʊ"]);
    }

    /// Vowel nuclei receive `PhoneRole::Nucleus` and `is_voiced = true`.
    #[test]
    fn vowel_nuclei_receive_pitch_bearing_role() {
        let plan = articulate(&hello_phrase(false));
        let gestures = &plan.gestures.gestures;

        // ɛ is the nucleus of "hel" (index 1).
        let epsilon = &gestures[1];
        assert_eq!(epsilon.phone, "ɛ");
        assert_eq!(epsilon.role, PhoneRole::Nucleus);
        assert!(epsilon.is_voiced, "ɛ nucleus must be voiced");

        // oʊ is the nucleus of "lo" (index 4).
        let ou = &gestures[4];
        assert_eq!(ou.phone, "oʊ");
        assert_eq!(ou.role, PhoneRole::Nucleus);
        assert!(ou.is_voiced, "oʊ nucleus must be voiced");
    }

    /// Unvoiced consonants do not become sustained pitched regions.
    #[test]
    fn unvoiced_consonants_are_not_pitched() {
        let plan = articulate(&hello_phrase(false));
        let gestures = &plan.gestures.gestures;

        // "h" is an unvoiced onset (index 0).
        let h = &gestures[0];
        assert_eq!(h.phone, "h");
        assert_eq!(h.role, PhoneRole::Onset);
        assert!(
            !h.is_voiced,
            "/h/ is an unvoiced consonant and must not be marked voiced"
        );
    }

    /// Output duration (last gesture end) matches total phrase duration.
    #[test]
    fn output_duration_matches_phrase_duration() {
        let phrase = hello_phrase(false);
        let expected_duration = phrase.total_duration_millis().unwrap();

        let plan = articulate(&phrase);
        let gestures = &plan.gestures.gestures;
        let last = gestures.last().expect("plan must not be empty");
        let plan_end = last.onset_ms + last.duration_ms;

        assert_eq!(
            plan_end, expected_duration,
            "articulator plan end ({plan_end} ms) must equal phrase duration ({expected_duration} ms)"
        );
    }

    /// A phrase with note targets produces a `Some(PitchCurve)`.
    #[test]
    fn phrase_with_notes_produces_pitch_curve() {
        let plan = articulate(&hello_phrase(true));
        assert!(
            plan.pitch_curve.is_some(),
            "phrase with note targets must yield a pitch curve"
        );
    }

    /// A phrase without note targets yields `pitch_curve = None`.
    #[test]
    fn phrase_without_notes_yields_no_pitch_curve() {
        let plan = articulate(&hello_phrase(false));
        assert!(
            plan.pitch_curve.is_none(),
            "phrase without notes must not produce a pitch curve"
        );
    }

    /// Energy curve contains one point per note-bearing syllable.
    #[test]
    fn energy_curve_has_one_point_per_noted_syllable() {
        let plan = articulate(&hello_phrase(true));
        // "hel" and "lo" each have a note → 2 energy points.
        assert_eq!(
            plan.energy_curve.points.len(),
            2,
            "two noted syllables must yield two energy points"
        );
    }

    /// Energy is separate from pitch: a phrase with notes has both non-empty
    /// energy and a pitch curve; neither is a copy of the other.
    #[test]
    fn energy_and_pitch_are_independent() {
        let plan = articulate(&hello_phrase(true));
        assert!(plan.pitch_curve.is_some());
        assert!(!plan.energy_curve.points.is_empty());

        // Energy levels are derived from velocity, not from frequency.
        let expected_level = Velocity::mezzo_forte().as_f32();
        for ep in &plan.energy_curve.points {
            assert!(
                (ep.level - expected_level).abs() < 1e-4,
                "energy level {:.4} != expected {:.4}",
                ep.level,
                expected_level
            );
        }
    }

    /// `is_phone_voiced` correctly identifies voiced and unvoiced phones.
    #[test]
    fn voicing_heuristic_classifies_phones() {
        // Unvoiced consonants
        for ipa in &["p", "t", "k", "f", "s", "ʃ", "θ", "h"] {
            assert!(
                !is_phone_voiced(ipa),
                "/{ipa}/ should be unvoiced"
            );
        }
        // Voiced sounds (consonants + vowels)
        for ipa in &["b", "d", "g", "v", "z", "m", "n", "l", "ɹ", "a", "ɛ", "oʊ", "eɪ"] {
            assert!(
                is_phone_voiced(ipa),
                "/{ipa}/ should be voiced"
            );
        }
    }

    /// The `l` coda consonant in "hel" is voiced.
    #[test]
    fn voiced_coda_consonant_is_marked_voiced() {
        let plan = articulate(&hello_phrase(false));
        // /l/ coda in "hel" is index 2.
        let l_coda = &plan.gestures.gestures[2];
        assert_eq!(l_coda.phone, "l");
        assert_eq!(l_coda.role, PhoneRole::Coda);
        assert!(l_coda.is_voiced, "/l/ is a voiced consonant");
    }

    /// An empty phrase yields an empty gesture plan.
    #[test]
    fn empty_phrase_yields_empty_plan() {
        let plan = articulate(&SungPhrase::new());
        assert!(plan.gestures.gestures.is_empty());
        assert!(plan.pitch_curve.is_none());
        assert!(plan.energy_curve.points.is_empty());
    }
}
