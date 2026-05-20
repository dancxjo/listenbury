use serde::{Deserialize, Serialize};

/// Status of a requested prosody control after synthesis.
///
/// Every control that is passed into a controlled synthesis call will produce
/// one of these statuses so callers can distinguish between controls that
/// actually changed the audio and controls that were silently noted or dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProsodyControlStatus {
    /// The control was captured as an explicit request before backend handling.
    Requested,
    /// The control was applied and took effect as a direct ONNX tensor input.
    Realized,
    /// The control was approximated by post-processing (e.g., silence was
    /// appended to the PCM to satisfy a pause request).
    Approximated,
    /// The control intent was recorded but no Piper runtime knob is available;
    /// the audio is unchanged by this control.
    AdvisoryOnly,
    /// Alias for advisory controls in diagnostics output that uses shorter wording.
    Advisory,
    /// The control is not supported by the current model path and was ignored.
    IgnoredUnsupported,
    /// The control was queued but not yet applied (reserved for future use).
    Deferred,
}

/// A single named prosody control status entry in synthesis diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlStatusEntry {
    /// Short name identifying the control (e.g. `"length_scale"`, `"pause_override[after sentence]"`).
    pub name: String,
    /// Whether the control was realized, approximated, advisory, etc.
    pub status: ProsodyControlStatus,
    /// Human-readable explanation of how the control was handled.
    pub detail: String,
}

/// A request to insert silence after the synthesized audio.
///
/// Piper ONNX does not expose per-token pause control, so pause requests are
/// approximated by appending silence samples to the output PCM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperPauseOverride {
    /// Duration of the requested pause in milliseconds.
    pub millis: u64,
    /// Human-readable label for diagnostics (e.g. `"after sentence"`).
    pub label: String,
}

/// A per-phoneme duration hint.
///
/// Advisory only: Piper ONNX does not expose per-phoneme timing control at
/// inference time.  These are recorded in diagnostics but do not change audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperPhonemeDurationOverride {
    /// Zero-based index into the phoneme ID sequence.
    pub phoneme_index: usize,
    /// Desired duration in milliseconds.
    pub millis: u64,
}

/// A phrase or sentence boundary hint.
///
/// Advisory only for the current ONNX model path.  Recorded in diagnostics
/// but does not change inference inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperBoundaryOverride {
    /// Index of the phoneme or word position after which the boundary occurs.
    pub after_index: usize,
    /// `true` for a strong (sentence-final) boundary; `false` for a weak
    /// (phrase-internal) boundary.
    pub strong: bool,
}

/// Low-level prosody controls for the rustified Piper synthesis backend.
///
/// These controls sit between symbolic prosody intent and the ONNX inference
/// layer.  Controls that map directly to ONNX tensor inputs (`length_scale`,
/// `noise_scale`, `noise_w`) are reported as [`ProsodyControlStatus::Realized`]
/// when synthesis completes.  Pause overrides are
/// [`ProsodyControlStatus::Approximated`] by appending silence.
/// Per-phoneme and boundary overrides are
/// [`ProsodyControlStatus::AdvisoryOnly`] — they are recorded in diagnostics
/// but do not currently alter ONNX inputs.
///
/// When all fields are `None` / empty (the default), synthesis behaves
/// identically to the uncontrolled path.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PiperProsodyControls {
    /// Override the global speaking-rate scale.  Values > 1.0 slow down;
    /// values < 1.0 speed up.  `Realized` when the model exposes a `scales`,
    /// `length_scale`, or combined tensor input.
    pub length_scale: Option<f32>,
    /// Override the acoustic variance/noise scale.  Affects how natural vs
    /// monotone the voice sounds.  `Realized` when the model exposes
    /// `noise_scale` or `scales`.
    pub noise_scale: Option<f32>,
    /// Override the phoneme-duration noise (`noise_w`).  Affects timing
    /// variation between phonemes.  `Realized` when the model exposes
    /// `noise_w` or `scales`.
    pub noise_w: Option<f32>,
    /// Silence segments to append after synthesis.  Each entry is
    /// `Approximated` by inserting the requested number of zero samples at the
    /// end of the output PCM.
    pub pause_overrides: Vec<PiperPauseOverride>,
    /// Per-phoneme duration hints.  `AdvisoryOnly`; no per-phoneme control is
    /// available in the current ONNX path.
    pub phoneme_duration_overrides: Vec<PiperPhonemeDurationOverride>,
    /// Phrase/sentence boundary hints.  `AdvisoryOnly` for the current ONNX
    /// path.
    pub boundary_overrides: Vec<PiperBoundaryOverride>,
}

/// Diagnostics produced by a controlled Piper synthesis pass.
///
/// Callers can inspect these to understand which controls took effect, which
/// were only advisory, and what the resulting PCM characteristics were.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperSynthesisDiagnostics {
    /// The phoneme IDs that were submitted to the ONNX model.
    pub input_phoneme_ids: Vec<i64>,
    /// The effective length scale used (after applying any override).
    pub applied_length_scale: f32,
    /// The effective noise scale used (after applying any override).
    pub applied_noise_scale: f32,
    /// The effective noise_w used (after applying any override).
    pub applied_noise_w: f32,
    /// Total silence appended for all pause overrides, in milliseconds.
    pub inserted_pause_ms: u64,
    /// Total PCM duration (model output + inserted pauses) in milliseconds.
    pub pcm_duration_ms: u64,
    /// Per-control status entries explaining what was applied, approximated,
    /// or ignored.
    pub control_statuses: Vec<ControlStatusEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prosody_controls_default_is_all_none_and_empty() {
        let controls = PiperProsodyControls::default();
        assert!(controls.length_scale.is_none());
        assert!(controls.noise_scale.is_none());
        assert!(controls.noise_w.is_none());
        assert!(controls.pause_overrides.is_empty());
        assert!(controls.phoneme_duration_overrides.is_empty());
        assert!(controls.boundary_overrides.is_empty());
    }

    #[test]
    fn prosody_control_status_variants_are_distinct() {
        let statuses = [
            ProsodyControlStatus::Requested,
            ProsodyControlStatus::Realized,
            ProsodyControlStatus::Approximated,
            ProsodyControlStatus::AdvisoryOnly,
            ProsodyControlStatus::Advisory,
            ProsodyControlStatus::IgnoredUnsupported,
            ProsodyControlStatus::Deferred,
        ];
        for (i, a) in statuses.iter().enumerate() {
            for (j, b) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn pause_override_stores_millis_and_label() {
        let pause = PiperPauseOverride {
            millis: 250,
            label: "after sentence".to_string(),
        };
        assert_eq!(pause.millis, 250);
        assert_eq!(pause.label, "after sentence");
    }

    #[test]
    fn phoneme_duration_override_stores_index_and_millis() {
        let ovr = PiperPhonemeDurationOverride {
            phoneme_index: 3,
            millis: 80,
        };
        assert_eq!(ovr.phoneme_index, 3);
        assert_eq!(ovr.millis, 80);
    }

    #[test]
    fn boundary_override_stores_index_and_strength() {
        let strong = PiperBoundaryOverride {
            after_index: 5,
            strong: true,
        };
        let weak = PiperBoundaryOverride {
            after_index: 2,
            strong: false,
        };
        assert!(strong.strong);
        assert!(!weak.strong);
    }

    #[test]
    fn control_status_entry_stores_name_status_and_detail() {
        let entry = ControlStatusEntry {
            name: "length_scale".to_string(),
            status: ProsodyControlStatus::Realized,
            detail: "overridden from 1.000 to 1.200".to_string(),
        };
        assert_eq!(entry.name, "length_scale");
        assert_eq!(entry.status, ProsodyControlStatus::Realized);
        assert!(entry.detail.contains("1.200"));
    }
}
