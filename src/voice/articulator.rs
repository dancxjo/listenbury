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
//! 2. Marks phones as voiced or unvoiced from the active phonemic inventory's
//!    feature data.
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

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::linguistic::phonology::{Phone, PhoneDecompositionPolicy, PhonemicInventory};
use crate::linguistic::variety::EnglishVariety;
use crate::prosody::pitch_curve::{Interpolation, PitchCurve};
use crate::prosody::singing::SungPhrase;
use crate::prosody::syllable::{FollowingBoundary, NucleusSpanError, PhoneSpan};
use crate::voice::coarticulation::{PhoneGesture, PhoneRole, VocalGesturePlan};
use crate::voice::tract::{PhoneAcousticTarget, PhoneRenderTarget};

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
        self.points
            .iter()
            .rev()
            .find(|p| p.t <= t)
            .map(|p| p.level)
            .unwrap_or(self.points[0].level)
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
    /// Per-syllable span metadata preserved for backend adapters.
    pub syllables: Vec<SyllableRenderSpan>,
}

/// A half-open gesture index span in an [`ArticulatorPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureSpan {
    pub start: usize,
    pub end: usize,
}

/// Preserved per-syllable gesture spans in the shared sung plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyllableRenderSpan {
    pub text: String,
    pub following_boundary: FollowingBoundary,
    pub gesture_span: GestureSpan,
    pub onset: GestureSpan,
    pub nucleus: GestureSpan,
    pub coda: GestureSpan,
    pub pitch_bearing_spans: Vec<GestureSpan>,
}

/// Backends expected to consume/degrade the shared sung plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SungBackendKind {
    Klatt,
    Mbrola,
    RiperKlattFallback,
    /// Reserved landing zone for the future direct Riper/ONNX sung path.
    RiperOnnxDirect,
    Piper,
}

/// Expected level of sung-detail fidelity by backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SungBackendDetail {
    /// Full phone timings + pitch-bearing durations/F0 where supported.
    PhoneTimed,
    /// Phone-timed contract rendered through the Klatt fallback adapter.
    PhoneTimedViaKlattFallback,
    /// Partial phone/prosody fidelity based on backend control surface.
    PartialPhoneProsody,
    /// Explicitly degraded to coarse text/phoneme hints.
    CoarseHintsOnly,
}

/// Errors from strict articulator-plan construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArticulationError {
    DecompositionFailed {
        syllable_text: String,
        policy: PhoneDecompositionPolicy,
        source: NucleusSpanError,
    },
}

impl fmt::Display for ArticulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecompositionFailed {
                syllable_text,
                policy,
                source,
            } => write!(
                f,
                "phone decomposition failed for syllable `{syllable_text}` with {policy:?}: {source:?}"
            ),
        }
    }
}

impl Error for ArticulationError {}

/// Backend-neutral phone-timed render target carrying shared timing, pitch,
/// and amplitude intent — without any renderer-specific source/filter details.
///
/// This is the shared substrate consumed by all phone-timed backends (Klatt,
/// MBROLA, …).  Backends that need additional acoustic parameters (e.g. Klatt
/// formant/source tables) must enrich this through their own adapter step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneTimedRenderTarget {
    /// The phone to render.
    pub phone: crate::linguistic::phonology::Phone,
    /// Requested duration in milliseconds.
    pub duration_ms: u64,
    /// Fundamental frequency in Hz.  `None` for unvoiced phones.
    pub f0_hz: Option<f32>,
    /// Overall amplitude (linear 0.0–1.0).
    pub amplitude: f32,
}

/// Backend-specific render contract derived from the shared sung plan.
///
/// The variants intentionally encode the amount of detail a backend may
/// consume.  Adapters should accept this type rather than a full
/// [`ArticulatorPlan`] when degradation is expected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum RenderPlan {
    /// Backend-neutral phone-timed targets carrying shared timing/pitch intent.
    ///
    /// Backends that require additional renderer-specific parameters (e.g.
    /// Klatt source/filter tables) must adapt these through a dedicated adapter
    /// step, such as [`klatt_render_targets_from_phone_timed`].
    PhoneTimed(Vec<PhoneTimedRenderTarget>),
    /// Text plus phone/prosody hints for backends with a partial control
    /// surface, such as the future direct Riper/ONNX path.
    PartialProsody {
        text: String,
        phones: Vec<PartialProsodyPhone>,
        pitch_hints: Vec<PitchHint>,
    },
    /// Coarse text-only rendering for TTS backends that cannot honor phone
    /// timing.  `ssml_hint` is advisory when a process backend supports it.
    CoarseText {
        text: String,
        ssml_hint: Option<String>,
    },
}

/// Per-phone hint preserved for partially controllable backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PartialProsodyPhone {
    pub phone: String,
    pub onset_ms: u64,
    pub duration_ms: u64,
    pub is_voiced: bool,
    pub role: PhoneRole,
}

/// Pitch hint sampled from the phrase pitch curve for a pitch-bearing phone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchHint {
    pub phone_index: usize,
    pub onset_ms: u64,
    pub duration_ms: u64,
    pub f0_hz: f32,
}

/// Describe how a backend is expected to consume this shared plan.
pub fn backend_detail_expectation(kind: SungBackendKind) -> SungBackendDetail {
    match kind {
        SungBackendKind::Klatt => SungBackendDetail::PhoneTimed,
        SungBackendKind::Mbrola => SungBackendDetail::PhoneTimed,
        SungBackendKind::RiperKlattFallback => SungBackendDetail::PhoneTimedViaKlattFallback,
        SungBackendKind::RiperOnnxDirect => SungBackendDetail::PartialPhoneProsody,
        SungBackendKind::Piper => SungBackendDetail::CoarseHintsOnly,
    }
}

/// Build the explicit degraded render contract for a backend.
pub fn render_plan_for_backend(
    kind: SungBackendKind,
    plan: &ArticulatorPlan,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> RenderPlan {
    match kind {
        SungBackendKind::Klatt | SungBackendKind::Mbrola | SungBackendKind::RiperKlattFallback => {
            RenderPlan::PhoneTimed(phone_timed_targets_from_articulator_plan(
                plan, amplitude, targets,
            ))
        }
        SungBackendKind::RiperOnnxDirect => partial_prosody_render_plan(plan),
        SungBackendKind::Piper => coarse_text_render_plan(plan),
    }
}

/// Build a partially degraded phone/prosody plan for the Riper adapter.
pub fn partial_prosody_render_plan(plan: &ArticulatorPlan) -> RenderPlan {
    let phones = plan
        .gestures
        .gestures
        .iter()
        .map(|gesture| PartialProsodyPhone {
            phone: gesture.phone.clone(),
            onset_ms: gesture.onset_ms,
            duration_ms: gesture.duration_ms,
            is_voiced: gesture.is_voiced,
            role: gesture.role,
        })
        .collect();
    RenderPlan::PartialProsody {
        text: render_text_from_plan(plan),
        phones,
        pitch_hints: pitch_hints_from_plan(plan),
    }
}

/// Build a coarse text-only plan for the process Piper adapter.
pub fn coarse_text_render_plan(plan: &ArticulatorPlan) -> RenderPlan {
    RenderPlan::CoarseText {
        text: render_text_from_plan(plan),
        ssml_hint: None,
    }
}

fn render_text_from_plan(plan: &ArticulatorPlan) -> String {
    let mut text = String::new();
    for syllable in &plan.syllables {
        text.push_str(&syllable.text);
        append_following_boundary(&mut text, syllable.following_boundary);
    }
    text.trim_end().to_string()
}

fn append_following_boundary(text: &mut String, boundary: FollowingBoundary) {
    match boundary {
        FollowingBoundary::None => {}
        FollowingBoundary::Word => text.push(' '),
        FollowingBoundary::Phrase => {
            if !text
                .chars()
                .last()
                .is_some_and(|c| matches!(c, '.' | '!' | '?' | ',' | ';' | ':'))
            {
                text.push('.');
            }
            text.push(' ');
        }
        FollowingBoundary::Rest => text.push_str(" ... "),
    }
}

fn pitch_hints_from_plan(plan: &ArticulatorPlan) -> Vec<PitchHint> {
    let Some(curve) = &plan.pitch_curve else {
        return Vec::new();
    };
    plan.gestures
        .gestures
        .iter()
        .enumerate()
        .filter_map(|(phone_index, gesture)| {
            if gesture.role != PhoneRole::Nucleus || !gesture.is_voiced {
                return None;
            }
            let midpoint = gesture.onset_ms + (gesture.duration_ms / 2);
            Some(PitchHint {
                phone_index,
                onset_ms: gesture.onset_ms,
                duration_ms: gesture.duration_ms,
                f0_hz: curve.sample_hz(Duration::from_millis(midpoint)),
            })
        })
        .collect()
}

// ─── Voicing features ────────────────────────────────────────────────────────

/// Determine whether a phone is voiced according to the active inventory.
pub fn is_phone_voiced(inventory: &PhonemicInventory, phone: &Phone) -> bool {
    inventory.features_for_phone(phone).is_voiced()
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
///   their voicing comes from the active inventory's feature data.
/// - Phones in the **nucleus span** become [`PhoneRole::Nucleus`] gestures
///   and preserve phone voicing while remaining the pitch-bearing span.
/// - Phones in the **coda span** become [`PhoneRole::Coda`] gestures;
///   their voicing comes from the same inventory feature data.
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
    let inventory = EnglishVariety::GeneralAmerican.phonemic_inventory();
    articulate_with_inventory_and_decomposition_policy(
        phrase,
        &inventory,
        PhoneDecompositionPolicy::KeepPhonemic,
    )
}

/// Convert a [`SungPhrase`] into an [`ArticulatorPlan`] using an explicit
/// phonemic inventory for phone feature lookup.
pub fn articulate_with_inventory(
    phrase: &SungPhrase,
    inventory: &PhonemicInventory,
) -> ArticulatorPlan {
    articulate_with_inventory_and_decomposition_policy(
        phrase,
        inventory,
        PhoneDecompositionPolicy::KeepPhonemic,
    )
}

/// Convert a [`SungPhrase`] into an [`ArticulatorPlan`] after applying an
/// explicit phone decomposition policy.
pub fn articulate_with_decomposition_policy(
    phrase: &SungPhrase,
    policy: PhoneDecompositionPolicy,
) -> ArticulatorPlan {
    let inventory = EnglishVariety::GeneralAmerican.phonemic_inventory();
    articulate_with_inventory_and_decomposition_policy(phrase, &inventory, policy)
}

/// Convert a [`SungPhrase`] with explicit inventory and decomposition policy.
pub fn articulate_with_inventory_and_decomposition_policy(
    phrase: &SungPhrase,
    inventory: &PhonemicInventory,
    policy: PhoneDecompositionPolicy,
) -> ArticulatorPlan {
    build_articulator_plan(phrase, inventory, policy, true)
        .expect("fallback mode should recover from decomposition errors")
}

/// Try to convert a [`SungPhrase`] with explicit inventory and decomposition
/// policy, returning structural decomposition errors instead of silently
/// falling back to broad phonemic phones.
pub fn try_articulate_with_inventory_and_decomposition_policy(
    phrase: &SungPhrase,
    inventory: &PhonemicInventory,
    policy: PhoneDecompositionPolicy,
) -> Result<ArticulatorPlan, ArticulationError> {
    build_articulator_plan(phrase, inventory, policy, false)
}

fn build_articulator_plan(
    phrase: &SungPhrase,
    inventory: &PhonemicInventory,
    policy: PhoneDecompositionPolicy,
    fallback_on_decomposition_error: bool,
) -> Result<ArticulatorPlan, ArticulationError> {
    let mut phone_gestures: Vec<PhoneGesture> = Vec::new();
    let mut energy_points: Vec<EnergyPoint> = Vec::new();
    let mut syllable_spans: Vec<SyllableRenderSpan> = Vec::new();

    for syllable in &phrase.syllables {
        let decomposed;
        let syllable = if policy == PhoneDecompositionPolicy::KeepPhonemic {
            syllable
        } else {
            match syllable.with_decomposition_policy(policy) {
                Ok(next) => {
                    decomposed = next;
                    &decomposed
                }
                Err(source) if fallback_on_decomposition_error => {
                    tracing::warn!(
                        syllable_text = %syllable.text,
                        ?policy,
                        error = ?source,
                        "phone decomposition failed; falling back to broad phonemic syllable"
                    );
                    syllable
                }
                Err(source) => {
                    return Err(ArticulationError::DecompositionFailed {
                        syllable_text: syllable.text.clone(),
                        policy,
                        source,
                    });
                }
            }
        };
        // ── Collect phone gestures ────────────────────────────────────────
        let phones = &syllable.phones;
        if phones.is_empty() {
            continue;
        }
        let base_start_ms = phones[0].start.millis;
        let explicit_duration_ms = phones
            .last()
            .map(|p| p.end.millis.saturating_sub(base_start_ms))
            .unwrap_or(0);
        let mut durations_ms: Vec<u64> = phones
            .iter()
            .map(|tpr| tpr.end.millis.saturating_sub(tpr.start.millis))
            .collect();
        let note_duration_ms = syllable
            .note
            .as_ref()
            .map(|n| n.duration.millis)
            .unwrap_or(0);
        if note_duration_ms > explicit_duration_ms {
            let extra = note_duration_ms - explicit_duration_ms;
            let pitch_spans = syllable.nucleus_subspans();
            stretch_pitch_bearing_durations(&mut durations_ms, extra, &pitch_spans);
        }

        let gesture_start = phone_gestures.len();
        let mut onset_ms = base_start_ms;
        for (idx, tpr) in phones.iter().enumerate() {
            let role = role_for_index(idx, syllable.onset, syllable.nucleus, syllable.coda);
            let duration_ms = durations_ms
                .get(idx)
                .copied()
                .unwrap_or_else(|| tpr.end.millis.saturating_sub(tpr.start.millis).max(1));
            phone_gestures.push(PhoneGesture {
                phone: tpr.phone.ipa.clone(),
                onset_ms,
                duration_ms,
                is_voiced: inventory.features_for_phone(&tpr.phone).is_voiced(),
                role,
                is_legato_context: false,
            });
            onset_ms = onset_ms.saturating_add(duration_ms);
        }
        let gesture_end = phone_gestures.len();
        let pitch_bearing_spans = syllable
            .nucleus_subspans()
            .iter()
            .map(|span| gesture_span_from_phone_span(gesture_start, *span))
            .collect();
        syllable_spans.push(SyllableRenderSpan {
            text: syllable.text.clone(),
            following_boundary: syllable.following_boundary,
            gesture_span: GestureSpan {
                start: gesture_start,
                end: gesture_end,
            },
            onset: gesture_span_from_phone_span(gesture_start, syllable.onset),
            nucleus: gesture_span_from_phone_span(gesture_start, syllable.nucleus),
            coda: gesture_span_from_phone_span(gesture_start, syllable.coda),
            pitch_bearing_spans,
        });

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

    Ok(ArticulatorPlan {
        gestures,
        pitch_curve,
        energy_curve,
        syllables: syllable_spans,
    })
}

/// Build a backend-neutral phone-timed sung plan from the shared articulator plan.
///
/// This adapter preserves per-phone durations from the shared plan and samples
/// F0 only for voiced nucleus phones.  The resulting targets carry no
/// Klatt-specific source/filter fields; backends that need them must pass the
/// neutral targets through [`klatt_render_targets_from_phone_timed`].
pub fn phone_timed_targets_from_articulator_plan(
    plan: &ArticulatorPlan,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> Vec<PhoneTimedRenderTarget> {
    plan.gestures
        .gestures
        .iter()
        .map(|gesture| {
            let table_entry = targets.get(gesture.phone.as_str());
            let f0_hz = if gesture.role == PhoneRole::Nucleus && gesture.is_voiced {
                plan.pitch_curve.as_ref().map(|curve| {
                    let midpoint = gesture.onset_ms + (gesture.duration_ms / 2);
                    curve.sample_hz(Duration::from_millis(midpoint))
                })
            } else {
                None
            };
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa(&gesture.phone),
                duration_ms: gesture.duration_ms,
                f0_hz: match table_entry {
                    Some(t) if !t.voiced => None,
                    _ => f0_hz,
                },
                amplitude,
            }
        })
        .collect()
}

/// Enrich backend-neutral [`PhoneTimedRenderTarget`]s with Klatt-specific
/// source and filter parameters from the acoustic target table.
///
/// This is the Klatt adapter step: it takes a slice of shared phone-timed
/// targets and produces [`PhoneRenderTarget`]s ready for
/// [`crate::voice::tract::render_phone_string`].
pub fn klatt_render_targets_from_phone_timed(
    neutral: &[PhoneTimedRenderTarget],
    acoustic_table: &HashMap<String, PhoneAcousticTarget>,
) -> Vec<PhoneRenderTarget> {
    neutral
        .iter()
        .map(|target| {
            let table_entry = acoustic_table.get(target.phone.ipa.as_str());
            PhoneRenderTarget {
                phone: target.phone.clone(),
                duration_ms: target.duration_ms,
                f0_hz: target.f0_hz,
                amplitude: target.amplitude,
                source: table_entry.map(|t| t.source.clone()),
                filter: table_entry.and_then(|t| t.filter.clone()),
            }
        })
        .collect()
}

/// Build Klatt phone render targets from the shared sung plan.
///
/// # Deprecation note
///
/// Prefer [`phone_timed_targets_from_articulator_plan`] to build the
/// backend-neutral plan, then pass the result through
/// [`klatt_render_targets_from_phone_timed`] for the Klatt adapter step.
/// This combined helper is retained for call-sites that still want the old
/// single-call convenience.
#[deprecated(
    since = "0.1.0",
    note = "use phone_timed_targets_from_articulator_plan + klatt_render_targets_from_phone_timed"
)]
pub fn klatt_targets_from_articulator_plan(
    plan: &ArticulatorPlan,
    amplitude: f32,
    targets: &HashMap<String, PhoneAcousticTarget>,
) -> Vec<PhoneRenderTarget> {
    klatt_render_targets_from_phone_timed(
        &phone_timed_targets_from_articulator_plan(plan, amplitude, targets),
        targets,
    )
}

fn role_for_index(idx: usize, onset: PhoneSpan, nucleus: PhoneSpan, _coda: PhoneSpan) -> PhoneRole {
    if idx < onset.end {
        PhoneRole::Onset
    } else if idx < nucleus.end {
        PhoneRole::Nucleus
    } else {
        PhoneRole::Coda
    }
}

fn gesture_span_from_phone_span(base: usize, span: PhoneSpan) -> GestureSpan {
    GestureSpan {
        start: base + span.start,
        end: base + span.end,
    }
}

fn stretch_pitch_bearing_durations(
    durations: &mut [u64],
    extra_ms: u64,
    pitch_spans: &[PhoneSpan],
) {
    if extra_ms == 0 {
        return;
    }
    let mut pitch_indices = Vec::new();
    for span in pitch_spans {
        for idx in span.start..span.end {
            if idx < durations.len() {
                pitch_indices.push(idx);
            }
        }
    }
    if pitch_indices.is_empty() {
        return;
    }

    let base_total: u128 = pitch_indices
        .iter()
        .map(|idx| durations[*idx] as u128)
        .sum();
    if base_total == 0 {
        let last = *pitch_indices.last().expect("checked non-empty");
        durations[last] = durations[last].saturating_add(extra_ms);
        return;
    }

    let mut remaining = extra_ms;
    for idx in pitch_indices
        .iter()
        .take(pitch_indices.len().saturating_sub(1))
    {
        let add = ((extra_ms as u128) * (durations[*idx] as u128) / base_total) as u64;
        durations[*idx] = durations[*idx].saturating_add(add);
        remaining = remaining.saturating_sub(add);
    }
    let last = *pitch_indices.last().expect("checked non-empty");
    durations[last] = durations[last].saturating_add(remaining);
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
            if with_note {
                Some(note(60, 0, 240))
            } else {
                None
            },
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
            if with_note {
                Some(note(67, 240, 250))
            } else {
                None
            },
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

    /// Inventory features correctly identify voiced and unvoiced phones.
    #[test]
    fn inventory_features_classify_phone_voicing() {
        let inventory = EnglishVariety::GeneralAmerican.phonemic_inventory();

        // Unvoiced consonants
        for ipa in &["p", "t", "k", "f", "s", "ʃ", "θ", "h"] {
            let phone = Phone::mapped(*ipa);
            assert!(
                !inventory.features_for_phone(&phone).is_voiced(),
                "/{ipa}/ should be unvoiced"
            );
        }
        // Voiced sounds (consonants + vowels)
        for ipa in &[
            "b", "d", "ɡ", "v", "z", "m", "n", "l", "ɹ", "a", "ɛ", "oʊ", "eɪ",
        ] {
            let phone = Phone::mapped(*ipa);
            assert!(
                inventory.features_for_phone(&phone).is_voiced(),
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
        assert!(plan.syllables.is_empty());
    }

    #[test]
    fn long_gal_note_stretches_nucleus_not_consonants() {
        let gal = SungSyllable::new(
            "gal",
            vec![timed("ɡ", 0, 40), timed("æ", 40, 120), timed("l", 120, 160)],
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 3).unwrap(),
            None,
            Some(note(60, 0, 1000)),
        )
        .unwrap();
        let mut phrase = SungPhrase::new();
        phrase.push(gal).unwrap();
        let plan = articulate(&phrase);
        let gestures = &plan.gestures.gestures;
        assert_eq!(gestures.len(), 3);
        assert_eq!(gestures[0].duration_ms, 40, "onset should stay short");
        assert_eq!(gestures[2].duration_ms, 40, "coda should stay short");
        assert!(
            gestures[1].duration_ms > 800,
            "nucleus should absorb most of note sustain"
        );
        let end_ms = gestures
            .last()
            .map(|g| g.onset_ms + g.duration_ms)
            .unwrap_or_default();
        assert_eq!(end_ms, 1000, "note duration should shape total plan");
        assert_eq!(plan.syllables[0].nucleus, GestureSpan { start: 1, end: 2 });
    }

    #[test]
    fn split_for_singing_allocates_diphthong_sustain_to_vowel_then_glide() {
        let lo = SungSyllable::new(
            "lo",
            vec![timed("l", 0, 40), timed("oʊ", 40, 240)],
            PhoneSpan::new(0, 1).unwrap(),
            PhoneSpan::new(1, 2).unwrap(),
            PhoneSpan::new(2, 2).unwrap(),
            None,
            Some(note(67, 0, 1000)),
        )
        .unwrap();
        let mut phrase = SungPhrase::new();
        phrase.push(lo).unwrap();

        let broad = articulate(&phrase);
        assert_eq!(
            broad
                .gestures
                .gestures
                .iter()
                .map(|gesture| gesture.phone.as_str())
                .collect::<Vec<_>>(),
            vec!["l", "oʊ"]
        );

        let split = articulate_with_decomposition_policy(
            &phrase,
            PhoneDecompositionPolicy::SplitForSinging,
        );
        let gestures = &split.gestures.gestures;
        assert_eq!(
            gestures
                .iter()
                .map(|gesture| gesture.phone.as_str())
                .collect::<Vec<_>>(),
            vec!["l", "o", "ʊ"]
        );
        assert_eq!(split.syllables[0].nucleus, GestureSpan { start: 1, end: 3 });
        assert_eq!(
            split.syllables[0].pitch_bearing_spans,
            vec![
                GestureSpan { start: 1, end: 2 },
                GestureSpan { start: 2, end: 3 }
            ]
        );
        assert!(
            gestures[1].duration_ms > gestures[2].duration_ms * 3,
            "stable vowel should receive most sung diphthong sustain"
        );
        assert_eq!(gestures[0].duration_ms, 40);
    }

    #[test]
    fn try_articulate_reports_decomposition_failure() {
        let malformed = SungSyllable {
            text: "bad".to_string(),
            phones: vec![timed("aɪ", 0, 100)],
            onset: PhoneSpan { start: 1, end: 0 },
            nucleus: PhoneSpan { start: 0, end: 1 },
            coda: PhoneSpan { start: 1, end: 1 },
            following_boundary: crate::prosody::syllable::FollowingBoundary::None,
            stress: None,
            note: None,
            pitch_curve: None,
            vibrato: None,
        };
        let phrase = SungPhrase {
            syllables: vec![malformed],
        };
        let inventory = EnglishVariety::GeneralAmerican.phonemic_inventory();

        let err = try_articulate_with_inventory_and_decomposition_policy(
            &phrase,
            &inventory,
            PhoneDecompositionPolicy::SplitForSinging,
        )
        .expect_err("strict articulation should surface decomposition errors");

        assert_eq!(
            err,
            ArticulationError::DecompositionFailed {
                syllable_text: "bad".to_string(),
                policy: PhoneDecompositionPolicy::SplitForSinging,
                source: NucleusSpanError::Inverted,
            }
        );
    }

    #[test]
    fn convenience_articulate_logs_and_falls_back_on_decomposition_failure() {
        let malformed = SungSyllable {
            text: "bad".to_string(),
            phones: vec![timed("aɪ", 0, 100)],
            onset: PhoneSpan { start: 1, end: 0 },
            nucleus: PhoneSpan { start: 0, end: 1 },
            coda: PhoneSpan { start: 1, end: 1 },
            following_boundary: crate::prosody::syllable::FollowingBoundary::None,
            stress: None,
            note: None,
            pitch_curve: None,
            vibrato: None,
        };
        let phrase = SungPhrase {
            syllables: vec![malformed],
        };

        let plan = articulate_with_decomposition_policy(
            &phrase,
            PhoneDecompositionPolicy::SplitForSinging,
        );

        assert_eq!(plan.gestures.gestures.len(), 1);
        assert_eq!(plan.gestures.gestures[0].phone, "aɪ");
    }

    #[test]
    fn phone_timed_adapter_keeps_unvoiced_phone_unpitched() {
        let plan = articulate(&hello_phrase(true));
        let table = crate::voice::tract::default_english_phone_targets();
        let targets = phone_timed_targets_from_articulator_plan(&plan, 0.7, &table);
        assert_eq!(targets[0].phone.ipa, "h");
        assert!(targets[0].f0_hz.is_none(), "unvoiced /h/ must not get F0");
        assert_eq!(targets[1].phone.ipa, "ɛ");
        assert!(targets[1].f0_hz.is_some(), "nucleus vowel should carry F0");
    }

    #[test]
    fn klatt_adapter_enriches_neutral_targets_with_source_filter() {
        let plan = articulate(&hello_phrase(true));
        let table = crate::voice::tract::default_english_phone_targets();
        let neutral = phone_timed_targets_from_articulator_plan(&plan, 0.7, &table);
        let klatt = klatt_render_targets_from_phone_timed(&neutral, &table);
        // Neutral and Klatt targets have the same phone/duration/f0/amplitude
        assert_eq!(neutral.len(), klatt.len());
        for (n, k) in neutral.iter().zip(klatt.iter()) {
            assert_eq!(n.phone.ipa, k.phone.ipa);
            assert_eq!(n.duration_ms, k.duration_ms);
            assert_eq!(n.f0_hz, k.f0_hz);
            assert_eq!(n.amplitude, k.amplitude);
        }
        // Klatt targets carry source/filter; neutral ones do not
        assert!(
            klatt.iter().any(|t| t.source.is_some() || t.filter.is_some()),
            "Klatt adapter should enrich at least some phones with source/filter"
        );
    }

    #[test]
    fn backend_degradation_expectations_are_explicit() {
        assert_eq!(
            backend_detail_expectation(SungBackendKind::Klatt),
            SungBackendDetail::PhoneTimed
        );
        assert_eq!(
            backend_detail_expectation(SungBackendKind::Mbrola),
            SungBackendDetail::PhoneTimed
        );
        assert_eq!(
            backend_detail_expectation(SungBackendKind::RiperKlattFallback),
            SungBackendDetail::PhoneTimedViaKlattFallback
        );
        assert_eq!(
            backend_detail_expectation(SungBackendKind::RiperOnnxDirect),
            SungBackendDetail::PartialPhoneProsody
        );
        assert_eq!(
            backend_detail_expectation(SungBackendKind::Piper),
            SungBackendDetail::CoarseHintsOnly
        );
    }

    #[test]
    fn backend_render_plans_encode_degradation_in_data() {
        let plan = articulate(&hello_phrase(true));
        let table = crate::voice::tract::default_english_phone_targets();

        let klatt = render_plan_for_backend(SungBackendKind::Klatt, &plan, 0.7, &table);
        let RenderPlan::PhoneTimed(klatt_targets) = klatt else {
            panic!("Klatt should receive a phone-timed render plan");
        };
        assert_eq!(
            klatt_targets
                .iter()
                .map(|target| target.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["h", "ɛ", "l", "l", "oʊ"]
        );

        let mbrola = render_plan_for_backend(SungBackendKind::Mbrola, &plan, 0.7, &table);
        let RenderPlan::PhoneTimed(targets) = mbrola else {
            panic!("MBROLA should receive a phone-timed render plan");
        };
        assert_eq!(
            targets
                .iter()
                .map(|target| target.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["h", "ɛ", "l", "l", "oʊ"]
        );

        let riper_fallback =
            render_plan_for_backend(SungBackendKind::RiperKlattFallback, &plan, 0.7, &table);
        let RenderPlan::PhoneTimed(targets) = riper_fallback else {
            panic!("Riper Klatt fallback should receive a phone-timed render plan");
        };
        assert_eq!(
            targets
                .iter()
                .map(|target| target.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["h", "ɛ", "l", "l", "oʊ"]
        );

        let riper_direct =
            render_plan_for_backend(SungBackendKind::RiperOnnxDirect, &plan, 0.7, &table);
        let RenderPlan::PartialProsody {
            text,
            phones,
            pitch_hints,
        } = riper_direct
        else {
            panic!("future direct Riper ONNX path should receive a partial prosody render plan");
        };
        assert_eq!(text, "hello");
        assert_eq!(
            phones
                .iter()
                .map(|phone| phone.phone.as_str())
                .collect::<Vec<_>>(),
            vec!["h", "ɛ", "l", "l", "oʊ"]
        );
        assert!(
            pitch_hints
                .iter()
                .any(|hint| phones[hint.phone_index].phone == "ɛ"),
            "Riper should get advisory nucleus pitch hints"
        );

        let piper = render_plan_for_backend(SungBackendKind::Piper, &plan, 0.7, &table);
        assert_eq!(
            piper,
            RenderPlan::CoarseText {
                text: "hello".to_string(),
                ssml_hint: None,
            }
        );
    }

    #[test]
    fn degraded_text_uses_explicit_syllable_boundaries() {
        let mut phrase = SungPhrase::new();
        phrase
            .push(hel_syllable(false).with_following_boundary(FollowingBoundary::None))
            .unwrap();
        phrase
            .push(lo_syllable(false).with_following_boundary(FollowingBoundary::Word))
            .unwrap();
        phrase
            .push(
                SungSyllable::new(
                    "world",
                    vec![timed("w", 500, 540), timed("ɝ", 540, 720)],
                    PhoneSpan::new(0, 1).unwrap(),
                    PhoneSpan::new(1, 2).unwrap(),
                    PhoneSpan::new(2, 2).unwrap(),
                    None,
                    None,
                )
                .unwrap()
                .with_following_boundary(FollowingBoundary::Phrase),
            )
            .unwrap();
        phrase
            .push(
                SungSyllable::new(
                    "again",
                    vec![timed("ə", 780, 880), timed("ɡ", 880, 930)],
                    PhoneSpan::new(0, 0).unwrap(),
                    PhoneSpan::new(0, 1).unwrap(),
                    PhoneSpan::new(1, 2).unwrap(),
                    None,
                    None,
                )
                .unwrap()
                .with_following_boundary(FollowingBoundary::Rest),
            )
            .unwrap();

        let RenderPlan::CoarseText { text, .. } = coarse_text_render_plan(&articulate(&phrase))
        else {
            panic!("expected coarse text render plan");
        };
        assert_eq!(text, "hello world. again ...");
    }
}
