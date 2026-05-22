//! Coarticulation pass for sung and chant-like vocal output.
//!
//! This module is an **execution-refinement layer**, not a prosody model.
//! It sits downstream of prosody planning and upstream of audio synthesis:
//! once an articulator has produced a [`VocalGesturePlan`] – a sequence of
//! timed phone gestures – this pass applies deterministic rules that handle
//! the transitions *between* phones.
//!
//! # What this is not
//!
//! - It is not a prosody engine. Pitch curves, phrase arcs, and timing stretch
//!   live in `prosody::*`.
//! - It is not an ML coarticulation predictor or a neural vocoder integration.
//! - It does not implement language-specific singing diction.
//!
//! # Rules applied
//!
//! 1. **Consonant attack preservation** – onset consonants keep their attack
//!    timing; only fine adjustments near boundaries are permitted.
//! 2. **Legato vowel-to-vowel smoothing** – when two adjacent nucleus phones
//!    are separated by a [`BoundaryKind::Legato`] context, a smooth transition
//!    marker is emitted and a small portion of the first vowel's duration is
//!    donated to the transition window.
//! 3. **Unvoiced consonants remain unpitched** – phones marked `is_voiced =
//!    false` always have `pitch_active = false` in the output.
//! 4. **Coda/release time borrowing** – release (coda) consonants may borrow
//!    up to [`CODA_BORROW_RATIO`] of the preceding nucleus duration; the
//!    nucleus duration is shortened accordingly, but never below
//!    [`MIN_NUCLEUS_DURATION_MS`] milliseconds, ensuring no negative durations.
//!
//! # Boundary kinds
//!
//! Each [`RefinedGesture`] carries a [`BoundaryKind`] describing how it
//! connects to the *following* gesture (or to silence if it is the last one).

use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum fraction of a nucleus's duration that a following coda consonant
/// may borrow.
pub const CODA_BORROW_RATIO: f64 = 0.15;

/// The nucleus duration is never shortened below this many milliseconds,
/// regardless of coda borrowing.
pub const MIN_NUCLEUS_DURATION_MS: u64 = 40;

/// Duration donated to a legato vowel-to-vowel transition window, in ms.
/// The first vowel's duration is reduced by this amount (clamped to
/// [`MIN_NUCLEUS_DURATION_MS`]).
pub const LEGATO_TRANSITION_WINDOW_MS: u64 = 20;

/// Maximum timing adjustment applied to an onset consonant near a boundary,
/// in milliseconds.
pub const MAX_ONSET_ADJUST_MS: u64 = 10;

// ─── PhoneRole ───────────────────────────────────────────────────────────────

/// The syllable-structural role of a phone within its syllable.
///
/// Roles affect how coarticulation rules are applied at boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneRole {
    /// A consonant appearing before the syllable nucleus (onset).
    Onset,
    /// A vowel or syllabic consonant forming the nucleus.
    Nucleus,
    /// A consonant appearing after the nucleus (coda / release).
    Coda,
}

// ─── BoundaryKind ────────────────────────────────────────────────────────────

/// The nature of the transition from one phone gesture to the next.
///
/// [`BoundaryKind`] is inferred by [`coarticulate`] from the input plan and
/// the articulation settings of adjacent gestures. It is attached to each
/// [`RefinedGesture`] to describe how it connects to its successor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryKind {
    /// A definite phrase boundary or pause; phone timings are largely
    /// preserved, with no cross-boundary blending.
    HardBreak,
    /// A syllable or word boundary without a legato request; slight
    /// coarticulation may be applied but no time donation occurs.
    SoftBreak,
    /// A smooth, connected transition (e.g. vowel-to-vowel in a slurred
    /// phrase). The first gesture donates a small time window to the
    /// transition.
    Legato,
    /// An explicitly notated rest; no coarticulation is applied.
    Rest,
}

// ─── PhoneGesture ────────────────────────────────────────────────────────────

/// A single phone gesture in the input articulation plan.
///
/// This is the raw intent produced by an upstream articulator: a phone label,
/// placement in time, voicing status, syllable role, and articulation context.
/// The coarticulation pass refines these into [`RefinedGesture`] instances.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneGesture {
    /// IPA or ARPABET phone label (e.g. `"æ"`, `"AE1"`, `"s"`).
    pub phone: String,
    /// Onset time in milliseconds from the start of the phrase.
    pub onset_ms: u64,
    /// Nominal duration in milliseconds.
    pub duration_ms: u64,
    /// Whether this phone is phonologically voiced.
    ///
    /// Unvoiced phones (`false`) will always have `pitch_active = false` in
    /// the coarticulated output.
    pub is_voiced: bool,
    /// Structural role of this phone within its syllable.
    pub role: PhoneRole,
    /// Whether the phrase articulation context at this phone is legato.
    ///
    /// When `true` and the adjacent phone is also a [`PhoneRole::Nucleus`],
    /// the boundary between them is treated as [`BoundaryKind::Legato`].
    pub is_legato_context: bool,
}

// ─── VocalGesturePlan ────────────────────────────────────────────────────────

/// An ordered sequence of phone gestures representing an articulation plan.
///
/// This is the input to the coarticulation pass. Gestures must be ordered by
/// `onset_ms`; the pass does not reorder them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocalGesturePlan {
    /// The ordered phone gestures that make up this vocal phrase.
    pub gestures: Vec<PhoneGesture>,
}

impl VocalGesturePlan {
    /// Construct an empty plan.
    pub fn empty() -> Self {
        Self {
            gestures: Vec::new(),
        }
    }
}

// ─── RefinedGesture ──────────────────────────────────────────────────────────

/// A phone gesture after coarticulation refinement.
///
/// Timings may have been adjusted from the input [`PhoneGesture`], pitch
/// activity is determined from voicing, and each gesture carries an explicit
/// [`BoundaryKind`] describing how it connects to the following gesture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinedGesture {
    /// IPA or ARPABET phone label, carried through from the input.
    pub phone: String,
    /// Onset time in milliseconds from the start of the phrase.
    ///
    /// May differ from the input onset by at most [`MAX_ONSET_ADJUST_MS`] for
    /// onset consonants at a boundary.
    pub onset_ms: u64,
    /// Duration in milliseconds after coarticulation adjustments.
    ///
    /// Always ≥ 0. Nucleus phones are never shortened below
    /// [`MIN_NUCLEUS_DURATION_MS`]. Coda phones preserve their original
    /// duration (time is instead borrowed from the preceding nucleus).
    pub duration_ms: u64,
    /// Whether this phone is phonologically voiced (carried from input).
    pub is_voiced: bool,
    /// Structural role, carried from the input.
    pub role: PhoneRole,
    /// Whether pitch synthesis should be active for this phone.
    ///
    /// Always `false` for unvoiced phones. For voiced phones this mirrors
    /// `is_voiced`.
    pub pitch_active: bool,
    /// The nature of the transition from this gesture to the next one.
    ///
    /// For the last gesture in the plan this is [`BoundaryKind::HardBreak`].
    pub boundary: BoundaryKind,
}

// ─── CoarticulatedPlan ───────────────────────────────────────────────────────

/// The result of the coarticulation pass: an ordered list of refined phone
/// gestures ready for downstream voice rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoarticulatedPlan {
    /// Refined gestures in onset-time order.
    pub gestures: Vec<RefinedGesture>,
}

impl CoarticulatedPlan {
    /// Returns `true` if the plan contains no gestures.
    pub fn is_empty(&self) -> bool {
        self.gestures.is_empty()
    }
}

// ─── Coarticulation pass ─────────────────────────────────────────────────────

/// Run the coarticulation pass over a [`VocalGesturePlan`].
///
/// This is the primary entry point of the module. It applies all deterministic
/// coarticulation rules in a single forward pass and returns a
/// [`CoarticulatedPlan`].
///
/// # Rules applied (in order)
///
/// 1. Infer the [`BoundaryKind`] between each adjacent pair of gestures.
/// 2. For legato nucleus–nucleus boundaries: donate
///    [`LEGATO_TRANSITION_WINDOW_MS`] from the first vowel to the transition,
///    clamping to [`MIN_NUCLEUS_DURATION_MS`].
/// 3. For coda phones preceded by a nucleus: borrow up to
///    `nucleus_duration * `[`CODA_BORROW_RATIO`]` ms from the nucleus, again
///    clamping to [`MIN_NUCLEUS_DURATION_MS`].
/// 4. Set `pitch_active = false` for all unvoiced phones.
///
/// Onset consonant timings are not moved by more than [`MAX_ONSET_ADJUST_MS`]
/// at a boundary.
pub fn coarticulate(plan: &VocalGesturePlan) -> CoarticulatedPlan {
    if plan.gestures.is_empty() {
        return CoarticulatedPlan {
            gestures: Vec::new(),
        };
    }

    let n = plan.gestures.len();
    // Work on mutable copies of onset and duration so later rules can read
    // values already modified by earlier ones.
    let mut onset_ms: Vec<u64> = plan.gestures.iter().map(|g| g.onset_ms).collect();
    let mut duration_ms: Vec<u64> = plan.gestures.iter().map(|g| g.duration_ms).collect();

    // ── Pass 1: infer boundary kinds ────────────────────────────────────────
    let mut boundaries: Vec<BoundaryKind> = (0..n).map(|i| infer_boundary(plan, i)).collect();
    // Last gesture always ends at a hard break (silence / end of phrase).
    if n > 0 {
        boundaries[n - 1] = BoundaryKind::HardBreak;
    }

    // ── Pass 2: legato vowel-to-vowel time donation ──────────────────────────
    for i in 0..n.saturating_sub(1) {
        if boundaries[i] == BoundaryKind::Legato
            && plan.gestures[i].role == PhoneRole::Nucleus
            && plan.gestures[i + 1].role == PhoneRole::Nucleus
        {
            let donation = LEGATO_TRANSITION_WINDOW_MS
                .min(duration_ms[i].saturating_sub(MIN_NUCLEUS_DURATION_MS));
            duration_ms[i] = duration_ms[i].saturating_sub(donation);
        }
    }

    // ── Pass 3: coda time borrowing from preceding nucleus ──────────────────
    for (i, gesture) in plan.gestures.iter().enumerate().take(n).skip(1) {
        if gesture.role == PhoneRole::Coda {
            // Find the most recent nucleus before this coda.
            if let Some(nucleus_idx) = (0..i)
                .rev()
                .find(|&j| plan.gestures[j].role == PhoneRole::Nucleus)
            {
                let available = duration_ms[nucleus_idx].saturating_sub(MIN_NUCLEUS_DURATION_MS);
                let borrow =
                    ((duration_ms[nucleus_idx] as f64 * CODA_BORROW_RATIO) as u64).min(available);
                duration_ms[nucleus_idx] = duration_ms[nucleus_idx].saturating_sub(borrow);
                // The onset of the coda is moved earlier by `borrow` ms.
                onset_ms[i] = onset_ms[i].saturating_sub(borrow);
            }
        }
    }

    // ── Pass 4: build refined gestures ──────────────────────────────────────
    let gestures = plan
        .gestures
        .iter()
        .enumerate()
        .map(|(i, g)| RefinedGesture {
            phone: g.phone.clone(),
            onset_ms: onset_ms[i],
            duration_ms: duration_ms[i],
            is_voiced: g.is_voiced,
            role: g.role,
            pitch_active: g.is_voiced,
            boundary: boundaries[i],
        })
        .collect();

    CoarticulatedPlan { gestures }
}

// ─── Boundary inference ──────────────────────────────────────────────────────

/// Infer the [`BoundaryKind`] for position `i` (the boundary between gesture
/// `i` and gesture `i+1`).
///
/// The last gesture always yields [`BoundaryKind::HardBreak`] (overwritten by
/// the caller anyway).
fn infer_boundary(plan: &VocalGesturePlan, i: usize) -> BoundaryKind {
    let n = plan.gestures.len();
    if i + 1 >= n {
        return BoundaryKind::HardBreak;
    }
    let cur = &plan.gestures[i];
    let nxt = &plan.gestures[i + 1];

    // Legato nucleus → nucleus
    if cur.is_legato_context
        && nxt.is_legato_context
        && cur.role == PhoneRole::Nucleus
        && nxt.role == PhoneRole::Nucleus
    {
        return BoundaryKind::Legato;
    }

    // Check for an explicit gap (rest) between gestures.
    let cur_end = cur.onset_ms.saturating_add(cur.duration_ms);
    if nxt.onset_ms > cur_end {
        return BoundaryKind::Rest;
    }

    BoundaryKind::SoftBreak
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nucleus(phone: &str, onset_ms: u64, duration_ms: u64) -> PhoneGesture {
        PhoneGesture {
            phone: phone.to_string(),
            onset_ms,
            duration_ms,
            is_voiced: true,
            role: PhoneRole::Nucleus,
            is_legato_context: false,
        }
    }

    fn nucleus_legato(phone: &str, onset_ms: u64, duration_ms: u64) -> PhoneGesture {
        PhoneGesture {
            is_legato_context: true,
            ..nucleus(phone, onset_ms, duration_ms)
        }
    }

    fn onset_consonant(phone: &str, onset_ms: u64, duration_ms: u64, voiced: bool) -> PhoneGesture {
        PhoneGesture {
            phone: phone.to_string(),
            onset_ms,
            duration_ms,
            is_voiced: voiced,
            role: PhoneRole::Onset,
            is_legato_context: false,
        }
    }

    fn coda_consonant(phone: &str, onset_ms: u64, duration_ms: u64, voiced: bool) -> PhoneGesture {
        PhoneGesture {
            phone: phone.to_string(),
            onset_ms,
            duration_ms,
            is_voiced: voiced,
            role: PhoneRole::Coda,
            is_legato_context: false,
        }
    }

    // ── Hard boundary ────────────────────────────────────────────────────────

    /// A hard boundary (last gesture) must leave phone timings mostly
    /// unchanged: the final gesture has the same duration as the input.
    #[test]
    fn hard_boundary_leaves_timings_mostly_unchanged() {
        let plan = VocalGesturePlan {
            gestures: vec![nucleus("a", 0, 200), nucleus("e", 200, 200)],
        };
        let result = coarticulate(&plan);
        // Last gesture → hard break, no time donation possible across it.
        let last = result.gestures.last().unwrap();
        assert_eq!(last.boundary, BoundaryKind::HardBreak);
        // The last gesture's duration must be unchanged (no borrowing from
        // anything after it, and no legato context).
        assert_eq!(last.duration_ms, 200);
    }

    /// Single-gesture plan: no neighbours, so duration and timing are intact.
    #[test]
    fn single_gesture_hard_boundary_unchanged() {
        let plan = VocalGesturePlan {
            gestures: vec![nucleus("a", 0, 300)],
        };
        let result = coarticulate(&plan);
        assert_eq!(result.gestures.len(), 1);
        assert_eq!(result.gestures[0].onset_ms, 0);
        assert_eq!(result.gestures[0].duration_ms, 300);
        assert_eq!(result.gestures[0].boundary, BoundaryKind::HardBreak);
    }

    // ── Legato vowel-to-vowel ─────────────────────────────────────────────────

    /// Two adjacent legato nuclei: the first vowel's duration is shortened by
    /// [`LEGATO_TRANSITION_WINDOW_MS`] and the boundary is marked `Legato`.
    #[test]
    fn legato_vowel_to_vowel_produces_legato_boundary_and_adjusted_timing() {
        let plan = VocalGesturePlan {
            gestures: vec![nucleus_legato("a", 0, 200), nucleus_legato("e", 200, 200)],
        };
        let result = coarticulate(&plan);
        assert_eq!(result.gestures[0].boundary, BoundaryKind::Legato);
        // First vowel donates LEGATO_TRANSITION_WINDOW_MS to the transition.
        let expected_duration = 200 - LEGATO_TRANSITION_WINDOW_MS;
        assert_eq!(result.gestures[0].duration_ms, expected_duration);
        // Second vowel is unaffected (it has no successor nucleus in legato).
        assert_eq!(result.gestures[1].duration_ms, 200);
    }

    /// Legato donation is clamped: if the vowel is short, it must not drop
    /// below [`MIN_NUCLEUS_DURATION_MS`].
    #[test]
    fn legato_donation_is_clamped_to_min_nucleus_duration() {
        let plan = VocalGesturePlan {
            // Vowel is only 50 ms – barely above MIN_NUCLEUS_DURATION_MS.
            gestures: vec![nucleus_legato("a", 0, 50), nucleus_legato("e", 50, 200)],
        };
        let result = coarticulate(&plan);
        assert_eq!(result.gestures[0].boundary, BoundaryKind::Legato);
        assert!(result.gestures[0].duration_ms >= MIN_NUCLEUS_DURATION_MS);
    }

    // ── Unvoiced consonants remain unpitched ─────────────────────────────────

    /// Unvoiced onset consonants must have `pitch_active = false`.
    #[test]
    fn unvoiced_consonant_has_pitch_inactive() {
        let plan = VocalGesturePlan {
            gestures: vec![
                onset_consonant("s", 0, 60, /*voiced=*/ false),
                nucleus("a", 60, 200),
            ],
        };
        let result = coarticulate(&plan);
        let s = &result.gestures[0];
        assert!(!s.is_voiced);
        assert!(
            !s.pitch_active,
            "unvoiced /s/ must have pitch_active = false"
        );
    }

    /// Voiced consonants must have `pitch_active = true`.
    #[test]
    fn voiced_consonant_has_pitch_active() {
        let plan = VocalGesturePlan {
            gestures: vec![
                onset_consonant("z", 0, 60, /*voiced=*/ true),
                nucleus("a", 60, 200),
            ],
        };
        let result = coarticulate(&plan);
        assert!(result.gestures[0].pitch_active);
    }

    /// An unvoiced coda consonant must also have `pitch_active = false`.
    #[test]
    fn unvoiced_coda_consonant_has_pitch_inactive() {
        let plan = VocalGesturePlan {
            gestures: vec![
                nucleus("a", 0, 200),
                coda_consonant("t", 200, 60, /*voiced=*/ false),
            ],
        };
        let result = coarticulate(&plan);
        let t = &result.gestures[1];
        assert!(!t.pitch_active);
    }

    // ── Coda/release consonants ───────────────────────────────────────────────

    /// A coda consonant must not produce negative durations: both nucleus and
    /// coda durations must be > 0 after borrowing.
    #[test]
    fn coda_consonant_does_not_produce_negative_duration() {
        let plan = VocalGesturePlan {
            gestures: vec![nucleus("a", 0, 200), coda_consonant("t", 200, 60, false)],
        };
        let result = coarticulate(&plan);
        for g in &result.gestures {
            assert!(
                g.duration_ms > 0,
                "gesture '{}' has zero or negative duration",
                g.phone
            );
        }
    }

    /// The coda consonant must still appear after the nucleus in the output
    /// (ordering is preserved, onset_ms of nucleus ≤ onset_ms of coda).
    #[test]
    fn coda_consonant_preserves_ordering() {
        let plan = VocalGesturePlan {
            gestures: vec![nucleus("a", 0, 200), coda_consonant("t", 200, 60, false)],
        };
        let result = coarticulate(&plan);
        assert_eq!(result.gestures.len(), 2);
        let nucleus_onset = result.gestures[0].onset_ms;
        let coda_onset = result.gestures[1].onset_ms;
        assert!(
            nucleus_onset <= coda_onset,
            "nucleus onset ({nucleus_onset}) must be ≤ coda onset ({coda_onset})"
        );
    }

    /// With a very short nucleus the borrow must be clamped to preserve
    /// [`MIN_NUCLEUS_DURATION_MS`].
    #[test]
    fn coda_borrow_is_clamped_to_min_nucleus_duration() {
        let plan = VocalGesturePlan {
            // Nucleus is just at the minimum – nothing can be borrowed.
            gestures: vec![
                nucleus("a", 0, MIN_NUCLEUS_DURATION_MS),
                coda_consonant("t", MIN_NUCLEUS_DURATION_MS, 60, false),
            ],
        };
        let result = coarticulate(&plan);
        assert_eq!(result.gestures[0].duration_ms, MIN_NUCLEUS_DURATION_MS);
    }

    // ── Empty plan ───────────────────────────────────────────────────────────

    #[test]
    fn empty_plan_produces_empty_output() {
        let plan = VocalGesturePlan::empty();
        let result = coarticulate(&plan);
        assert!(result.is_empty());
    }

    // ── Rest boundary ────────────────────────────────────────────────────────

    /// When gestures have an explicit gap between them the boundary should be
    /// classified as [`BoundaryKind::Rest`].
    #[test]
    fn gap_between_gestures_produces_rest_boundary() {
        let plan = VocalGesturePlan {
            gestures: vec![
                nucleus("a", 0, 100),
                // gap: onset 300 > end of previous (100)
                nucleus("e", 300, 100),
            ],
        };
        let result = coarticulate(&plan);
        assert_eq!(result.gestures[0].boundary, BoundaryKind::Rest);
    }
}
