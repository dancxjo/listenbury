use crate::prosody::vibrato::Vibrato;
use crate::voice::tract::targets::{PhoneRenderTarget, VocalTractFilterTarget};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KlattFrameParams {
    pub(crate) f0_hz: Option<f32>,
    pub(crate) amplitude: f32,
    pub(crate) breathiness: f32,
    pub(crate) spectral_tilt_db_per_octave: f32,
    pub(crate) vibrato: Option<Vibrato>,
    pub(crate) filter: Option<VocalTractFilterTarget>,
}

impl KlattFrameParams {
    pub(crate) fn from_target(target: &PhoneRenderTarget) -> Self {
        Self {
            f0_hz: target.f0_hz,
            amplitude: target.amplitude,
            breathiness: target.source.as_ref().map(|s| s.breathiness).unwrap_or(0.0),
            spectral_tilt_db_per_octave: target
                .source
                .as_ref()
                .map(|s| s.spectral_tilt_db_per_octave)
                .unwrap_or(-6.0),
            vibrato: target.vibrato,
            filter: target.filter.clone(),
        }
        .clamped()
    }

    pub(crate) fn clamped(mut self) -> Self {
        self.amplitude = self.amplitude.clamp(0.0, 1.0);
        self.breathiness = self.breathiness.clamp(0.0, 1.0);
        self.spectral_tilt_db_per_octave = self.spectral_tilt_db_per_octave.clamp(-24.0, 0.0);
        self.f0_hz = self.f0_hz.map(|f0| f0.clamp(40.0, 1_200.0));
        self.filter = self.filter.map(clamp_filter);
        self
    }
}

pub(crate) fn interpolate(
    left: &KlattFrameParams,
    right: &KlattFrameParams,
    alpha: f32,
) -> KlattFrameParams {
    let alpha = alpha.clamp(0.0, 1.0);
    let interp = |a: f32, b: f32| a + (b - a) * alpha;
    KlattFrameParams {
        f0_hz: match (left.f0_hz, right.f0_hz) {
            (Some(a), Some(b)) => Some(interp(a, b)),
            (Some(a), None) => (alpha < 0.5).then_some(a),
            (None, Some(b)) => (alpha >= 0.5).then_some(b),
            (None, None) => None,
        },
        amplitude: interp(left.amplitude, right.amplitude),
        breathiness: interp(left.breathiness, right.breathiness),
        spectral_tilt_db_per_octave: interp(
            left.spectral_tilt_db_per_octave,
            right.spectral_tilt_db_per_octave,
        ),
        vibrato: if alpha < 0.5 {
            left.vibrato
        } else {
            right.vibrato
        },
        filter: match (&left.filter, &right.filter) {
            (Some(a), Some(b)) => Some(VocalTractFilterTarget {
                f1_hz: interp(a.f1_hz, b.f1_hz),
                f1_bw_hz: interp(a.f1_bw_hz, b.f1_bw_hz),
                f1_amp_db: interp(a.f1_amp_db, b.f1_amp_db),
                f2_hz: interp(a.f2_hz, b.f2_hz),
                f2_bw_hz: interp(a.f2_bw_hz, b.f2_bw_hz),
                f2_amp_db: interp(a.f2_amp_db, b.f2_amp_db),
                f3_hz: interp(a.f3_hz, b.f3_hz),
                f3_bw_hz: interp(a.f3_bw_hz, b.f3_bw_hz),
                f3_amp_db: interp(a.f3_amp_db, b.f3_amp_db),
                f4_hz: match (a.f4_hz, b.f4_hz) {
                    (Some(x), Some(y)) => Some(interp(x, y)),
                    (Some(x), None) => (alpha < 0.5).then_some(x),
                    (None, Some(y)) => (alpha >= 0.5).then_some(y),
                    (None, None) => None,
                },
                f4_bw_hz: match (a.f4_bw_hz, b.f4_bw_hz) {
                    (Some(x), Some(y)) => Some(interp(x, y)),
                    (Some(x), None) => (alpha < 0.5).then_some(x),
                    (None, Some(y)) => (alpha >= 0.5).then_some(y),
                    (None, None) => None,
                },
                f4_amp_db: match (a.f4_amp_db, b.f4_amp_db) {
                    (Some(x), Some(y)) => Some(interp(x, y)),
                    (Some(x), None) => (alpha < 0.5).then_some(x),
                    (None, Some(y)) => (alpha >= 0.5).then_some(y),
                    (None, None) => None,
                },
            }),
            (Some(a), None) => (alpha < 0.5).then_some(a.clone()),
            (None, Some(b)) => (alpha >= 0.5).then_some(b.clone()),
            (None, None) => None,
        },
    }
    .clamped()
}

fn clamp_filter(mut filter: VocalTractFilterTarget) -> VocalTractFilterTarget {
    filter.f1_hz = filter.f1_hz.clamp(150.0, 1_300.0);
    filter.f2_hz = filter.f2_hz.clamp(350.0, 4_000.0);
    filter.f3_hz = filter.f3_hz.clamp(900.0, 5_200.0);
    filter.f1_bw_hz = filter.f1_bw_hz.clamp(30.0, 500.0);
    filter.f2_bw_hz = filter.f2_bw_hz.clamp(30.0, 700.0);
    filter.f3_bw_hz = filter.f3_bw_hz.clamp(30.0, 900.0);
    filter.f4_hz = filter.f4_hz.map(|f| f.clamp(1_200.0, 6_500.0));
    filter.f4_bw_hz = filter.f4_bw_hz.map(|b| b.clamp(30.0, 1_200.0));
    filter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;
    use crate::voice::tract::targets::{
        GlottalSourceTarget, PhoneRenderTarget, VocalTractFilterTarget,
    };

    #[test]
    fn clamping_enforces_supported_ranges() {
        let target = PhoneRenderTarget {
            phone: Phone::new_ipa("a"),
            duration_ms: 100,
            f0_hz: Some(20.0),
            amplitude: 1.7,
            vibrato: None,
            source: Some(GlottalSourceTarget {
                breathiness: 2.0,
                open_quotient: 0.5,
                spectral_tilt_db_per_octave: -100.0,
            }),
            filter: Some(VocalTractFilterTarget {
                f1_hz: 10.0,
                f1_bw_hz: 1.0,
                f1_amp_db: 0.0,
                f2_hz: 10_000.0,
                f2_bw_hz: 2_000.0,
                f2_amp_db: 0.0,
                f3_hz: 100.0,
                f3_bw_hz: 2_000.0,
                f3_amp_db: 0.0,
                f4_hz: Some(9_000.0),
                f4_bw_hz: Some(2_000.0),
                f4_amp_db: Some(0.0),
            }),
        };
        let params = KlattFrameParams::from_target(&target);
        assert_eq!(params.f0_hz, Some(40.0));
        assert_eq!(params.amplitude, 1.0);
        assert_eq!(params.breathiness, 1.0);
        assert_eq!(params.spectral_tilt_db_per_octave, -24.0);
        let filter = params.filter.expect("filter should be clamped");
        assert_eq!(filter.f1_hz, 150.0);
        assert_eq!(filter.f2_hz, 4_000.0);
    }
}
