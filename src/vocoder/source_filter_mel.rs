use anyhow::{Result, ensure};

use crate::vocoder::MelFrame;
use crate::voice::{
    FormantEstimation, GlottalSourceEstimate, NoiseEstimate, PhoneAcousticTarget,
    PhoneRenderTarget, PhoneTimedRenderTarget, SourceFilterFrame, SourceFilterTrack,
    VocalTractFilterEstimate, VocalTractFilterTarget, VoicingEstimate,
    default_english_phone_targets, klatt_render_targets_from_phone_timed,
};

const SOURCE_FILTER_SAMPLE_RATE_HZ: u32 = 16_000;
const SOURCE_FILTER_FRAME_MS: u64 = 16;
const MEL_BINS: usize = 80;
const MEL_MIN_HZ: f32 = 0.0;
const SPECTRAL_REFERENCE_HZ: f32 = 80.0;
const MEL_MAX_HZ: f32 = 8_000.0;
const MEL_SPECTRAL_FLOOR: f32 = 0.006;
const MEL_TEMPORAL_SMOOTHING: f32 = 0.38;
const MEL_MIN_FRAME_ENERGY_RATIO: f32 = 0.46;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MelF0Track {
    pub(crate) mel: Vec<MelFrame>,
    pub(crate) f0_hz: Vec<f32>,
    pub(crate) voiced: Vec<bool>,
}

pub(crate) fn phone_timed_to_source_filter_track(
    targets: &[PhoneTimedRenderTarget],
) -> Result<SourceFilterTrack> {
    ensure!(
        !targets.is_empty(),
        "source-filter bridge received empty phone-timed input"
    );

    let acoustic_table = default_english_phone_targets();
    let render_targets = klatt_render_targets_from_phone_timed(targets, &acoustic_table);
    ensure!(
        !render_targets.is_empty(),
        "source-filter bridge produced no render targets"
    );

    let mut frames = Vec::new();
    let mut cursor_ms = 0_u64;
    for (index, target) in render_targets.iter().enumerate() {
        let acoustic = acoustic_table.get(target.phone.ipa.as_str());
        let next = render_targets.get(index + 1);
        append_phone_source_filter_frames(target, acoustic, next, &mut frames, &mut cursor_ms);
    }

    ensure!(
        frames.iter().any(|frame| frame.confidence > 0.0),
        "source-filter bridge produced no energetic frames"
    );

    Ok(SourceFilterTrack {
        sample_rate: SOURCE_FILTER_SAMPLE_RATE_HZ,
        hop_ms: SOURCE_FILTER_FRAME_MS as f32,
        frames,
    })
}

pub(crate) fn source_filter_track_to_mel_f0(track: &SourceFilterTrack) -> Result<MelF0Track> {
    ensure!(
        !track.frames.is_empty(),
        "mel bridge received empty source-filter track"
    );

    let mut mel = Vec::with_capacity(track.frames.len());
    let mut f0_hz = Vec::with_capacity(track.frames.len());
    let mut voiced = Vec::with_capacity(track.frames.len());

    for frame in &track.frames {
        mel.push(source_filter_frame_to_mel(frame));
        f0_hz.push(frame.voicing.f0_hz.unwrap_or(120.0).clamp(55.0, 1_200.0));
        voiced.push(frame.voicing.voicing_probability > 0.5);
    }

    smooth_mel_track(&mut mel);

    ensure!(
        mel.iter()
            .any(|frame| frame.bins.iter().any(|bin| *bin > 0.0)),
        "mel bridge produced no mel energy from source-filter track"
    );

    Ok(MelF0Track { mel, f0_hz, voiced })
}

fn append_phone_source_filter_frames(
    target: &PhoneRenderTarget,
    acoustic: Option<&PhoneAcousticTarget>,
    next: Option<&PhoneRenderTarget>,
    frames: &mut Vec<SourceFilterFrame>,
    cursor_ms: &mut u64,
) {
    let frame_durations = duration_chunks(target.duration_ms.max(1), SOURCE_FILTER_FRAME_MS);
    let frame_count = frame_durations.len();
    let base_filter = target.filter.clone();
    let next_filter = next.and_then(|target| target.filter.as_ref());

    for (frame_index, duration_ms) in frame_durations.into_iter().enumerate() {
        let local_t = if frame_count <= 1 {
            0.5
        } else {
            frame_index as f32 / (frame_count - 1) as f32
        };
        let envelope = phone_envelope(local_t, frame_count);
        let mut filter = base_filter.clone();
        if let (Some(current), Some(next)) = (filter.as_ref(), next_filter) {
            if should_blend_filter(acoustic) {
                filter = Some(lerp_filter(current, next, boundary_blend(local_t)));
            }
        }

        let frame_start_ms = *cursor_ms;
        let frame_end_ms = frame_start_ms.saturating_add(duration_ms);
        frames.push(source_filter_frame(
            target,
            acoustic,
            filter.as_ref(),
            frame_start_ms,
            frame_end_ms,
            envelope,
        ));
        *cursor_ms = frame_end_ms;
    }
}

fn source_filter_frame(
    target: &PhoneRenderTarget,
    acoustic: Option<&PhoneAcousticTarget>,
    filter: Option<&VocalTractFilterTarget>,
    frame_start_ms: u64,
    frame_end_ms: u64,
    envelope: f32,
) -> SourceFilterFrame {
    let source = target.source.clone().unwrap_or_else(default_source);
    let amplitude = target.amplitude.clamp(0.0, 1.0) * envelope;
    let is_voiced = target.f0_hz.is_some() && acoustic.map(|target| target.voiced).unwrap_or(true);
    let mut frication = acoustic.map(|target| target.frication_level).unwrap_or(0.0);
    let mut aspiration = acoustic
        .map(|target| target.aspiration_level)
        .unwrap_or(0.0);
    if target.phone.ipa.as_str() == "h" {
        frication *= 0.07;
        aspiration = aspiration.max(0.9);
    }
    let is_stop = acoustic.map(|target| target.is_stop).unwrap_or(false);
    let burst = if is_stop && envelope > 0.55 {
        acoustic.and_then(|target| target.burst_hz_hint).is_some() as u8 as f32 * 0.28
    } else {
        0.0
    };
    let noise_ratio = frication.max(aspiration).max(burst).clamp(0.0, 1.0);

    SourceFilterFrame {
        frame_start_ms,
        frame_end_ms,
        voicing: VoicingEstimate {
            f0_hz: target.f0_hz,
            f0_confidence: if is_voiced { 0.85 } else { 0.0 },
            voicing_probability: if is_voiced { 0.9 } else { 0.0 },
            hnr_db: if is_voiced {
                18.0 * (1.0 - noise_ratio)
            } else {
                -15.0
            },
        },
        source: GlottalSourceEstimate {
            spectral_tilt_db_per_octave: source.spectral_tilt_db_per_octave,
            breathiness: source.breathiness.max(aspiration).clamp(0.0, 1.0),
            open_quotient: source.open_quotient.clamp(0.0, 1.0),
        },
        filter: filter.map(filter_estimate).unwrap_or_default(),
        noise: NoiseEstimate {
            frication_energy: (frication + burst * 0.45).clamp(0.0, 1.0),
            noise_ratio,
        },
        confidence: amplitude,
    }
}

fn source_filter_frame_to_mel(frame: &SourceFilterFrame) -> MelFrame {
    let energy = frame.confidence.clamp(0.0, 1.0);
    let voiced = frame.voicing.voicing_probability.clamp(0.0, 1.0);
    let noise = frame.noise.noise_ratio.clamp(0.0, 1.0);
    let tilt = frame.source.spectral_tilt_db_per_octave;

    MelFrame {
        bins: (0..MEL_BINS)
            .map(|bin| {
                let hz = mel_bin_center_hz(bin, MEL_BINS);
                let source = source_spectrum(hz, tilt, voiced, noise);
                let filter = filter_spectrum(hz, &frame.filter);
                let frication = consonant_noise_profile(hz) * frame.noise.frication_energy * 0.22;
                let aspiration = aspiration_noise_profile(hz) * frame.source.breathiness * 0.11;
                let broad_speech_bed = speech_band_profile(hz) * MEL_SPECTRAL_FLOOR;
                (energy * (source * filter + frication + aspiration + broad_speech_bed)).max(0.0)
            })
            .collect(),
    }
}

fn smooth_mel_track(mel: &mut [MelFrame]) {
    for frame in mel.iter_mut() {
        smooth_mel_bins(&mut frame.bins);
        compress_mel_contrast(&mut frame.bins);
    }

    if mel.len() < 3 {
        return;
    }
    let original = mel.to_vec();
    for index in 1..mel.len() - 1 {
        for bin in 0..mel[index].bins.len() {
            let neighbor_mean =
                (original[index - 1].bins[bin] + original[index + 1].bins[bin]) * 0.5;
            mel[index].bins[bin] = lerp(
                original[index].bins[bin],
                neighbor_mean,
                MEL_TEMPORAL_SMOOTHING,
            );
        }
    }
    smooth_frame_energy(mel);
}

fn smooth_mel_bins(bins: &mut [f32]) {
    if bins.len() < 3 {
        return;
    }
    let original = bins.to_vec();
    for index in 1..bins.len() - 1 {
        bins[index] = original[index] * 0.58
            + (original[index - 1] + original[index + 1]) * 0.18
            + (original[index.saturating_sub(2)] + original[(index + 2).min(original.len() - 1)])
                * 0.03;
    }
}

fn compress_mel_contrast(bins: &mut [f32]) {
    let peak = bins.iter().copied().fold(0.0_f32, f32::max);
    if peak <= f32::EPSILON {
        return;
    }
    let floor = peak * 0.018;
    for bin in bins {
        *bin = (*bin + floor).powf(0.84);
    }
}

fn smooth_frame_energy(mel: &mut [MelFrame]) {
    if mel.len() < 3 {
        return;
    }
    let energies = mel
        .iter()
        .map(|frame| frame.bins.iter().sum::<f32>())
        .collect::<Vec<_>>();
    for index in 1..mel.len() - 1 {
        let local_energy =
            energies[index - 1] * 0.25 + energies[index] * 0.5 + energies[index + 1] * 0.25;
        if local_energy <= f32::EPSILON
            || energies[index] >= local_energy * MEL_MIN_FRAME_ENERGY_RATIO
        {
            continue;
        }
        let target = local_energy * MEL_MIN_FRAME_ENERGY_RATIO;
        let gain = (target / energies[index].max(f32::EPSILON)).min(4.0);
        for bin in &mut mel[index].bins {
            *bin *= gain;
        }
    }
}

fn filter_estimate(filter: &VocalTractFilterTarget) -> VocalTractFilterEstimate {
    VocalTractFilterEstimate {
        f1: Some(formant_estimate(
            filter.f1_hz,
            Some(filter.f1_bw_hz),
            filter.f1_amp_db,
        )),
        f2: Some(formant_estimate(
            filter.f2_hz,
            Some(filter.f2_bw_hz),
            filter.f2_amp_db,
        )),
        f3: Some(formant_estimate(
            filter.f3_hz,
            Some(filter.f3_bw_hz),
            filter.f3_amp_db,
        )),
        f4: filter
            .f4_hz
            .map(|hz| formant_estimate(hz, filter.f4_bw_hz, filter.f4_amp_db.unwrap_or(-9.0))),
        nasality: None,
    }
}

fn formant_estimate(
    frequency_hz: f32,
    bandwidth_hz: Option<f32>,
    amplitude_db: f32,
) -> FormantEstimation {
    FormantEstimation {
        frequency_hz,
        bandwidth_hz,
        amplitude_db,
        confidence: 0.85,
    }
}

fn filter_spectrum(hz: f32, filter: &VocalTractFilterEstimate) -> f32 {
    let mut energy = 0.12;
    for formant in [&filter.f1, &filter.f2, &filter.f3, &filter.f4]
        .into_iter()
        .flatten()
    {
        energy += gaussian_hz(
            hz,
            formant.frequency_hz,
            formant.bandwidth_hz.unwrap_or(180.0).max(110.0) * 3.4,
        ) * db_to_linear(formant.amplitude_db);
    }
    energy
}

fn source_spectrum(hz: f32, tilt_db_per_octave: f32, voiced: f32, noise: f32) -> f32 {
    let octaves = (hz.max(SPECTRAL_REFERENCE_HZ) / SPECTRAL_REFERENCE_HZ).log2();
    let tilted = db_to_linear(tilt_db_per_octave * 0.55 * octaves).clamp(0.08, 2.2);
    let broadband = high_band_noise_profile(hz).max(0.08);
    let low_periodic_tamer = (hz / (hz + 180.0)).clamp(0.18, 1.0);
    voiced * tilted * low_periodic_tamer * 0.78 + noise.max(1.0 - voiced) * broadband * 0.16
}

fn speech_band_profile(hz: f32) -> f32 {
    gaussian_hz(hz, 550.0, 650.0) * 0.45
        + gaussian_hz(hz, 1_600.0, 1_100.0) * 0.35
        + gaussian_hz(hz, 3_000.0, 1_700.0) * 0.20
}

fn aspiration_noise_profile(hz: f32) -> f32 {
    gaussian_hz(hz, 700.0, 1_050.0) * 0.58
        + gaussian_hz(hz, 1_850.0, 1_450.0) * 0.34
        + high_band_noise_profile(hz) * 0.08
}

fn phone_envelope(local_t: f32, frame_count: usize) -> f32 {
    if frame_count <= 2 {
        return 0.95;
    }
    let attack = (local_t / 0.12).clamp(0.0, 1.0);
    let release = ((1.0 - local_t) / 0.16).clamp(0.0, 1.0);
    attack.min(release).powf(0.45).max(0.55)
}

fn should_blend_filter(acoustic: Option<&PhoneAcousticTarget>) -> bool {
    acoustic
        .map(|target| target.is_vowel || target.is_nasal || target.is_approximant)
        .unwrap_or(true)
}

fn boundary_blend(local_t: f32) -> f32 {
    if local_t < 0.45 {
        0.0
    } else {
        smoothstep((local_t - 0.45) / 0.55) * 0.65
    }
}

fn lerp_filter(
    current: &VocalTractFilterTarget,
    next: &VocalTractFilterTarget,
    amount: f32,
) -> VocalTractFilterTarget {
    VocalTractFilterTarget {
        f1_hz: lerp(current.f1_hz, next.f1_hz, amount),
        f1_bw_hz: lerp(current.f1_bw_hz, next.f1_bw_hz, amount),
        f1_amp_db: lerp(current.f1_amp_db, next.f1_amp_db, amount),
        f2_hz: lerp(current.f2_hz, next.f2_hz, amount),
        f2_bw_hz: lerp(current.f2_bw_hz, next.f2_bw_hz, amount),
        f2_amp_db: lerp(current.f2_amp_db, next.f2_amp_db, amount),
        f3_hz: lerp(current.f3_hz, next.f3_hz, amount),
        f3_bw_hz: lerp(current.f3_bw_hz, next.f3_bw_hz, amount),
        f3_amp_db: lerp(current.f3_amp_db, next.f3_amp_db, amount),
        f4_hz: lerp_optional(current.f4_hz, next.f4_hz, amount),
        f4_bw_hz: lerp_optional(current.f4_bw_hz, next.f4_bw_hz, amount),
        f4_amp_db: lerp_optional(current.f4_amp_db, next.f4_amp_db, amount),
    }
}

fn lerp_optional(left: Option<f32>, right: Option<f32>, amount: f32) -> Option<f32> {
    match (left, right) {
        (Some(left), Some(right)) => Some(lerp(left, right, amount)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn duration_chunks(total_ms: u64, frame_ms: u64) -> Vec<u64> {
    let mut chunks = Vec::new();
    let mut remaining = total_ms;
    while remaining >= frame_ms {
        chunks.push(frame_ms);
        remaining -= frame_ms;
    }
    if remaining > 0 {
        chunks.push(remaining);
    }
    if chunks.is_empty() {
        chunks.push(1);
    }
    chunks
}

fn default_source() -> crate::voice::GlottalSourceTarget {
    crate::voice::GlottalSourceTarget {
        breathiness: 0.05,
        open_quotient: 0.5,
        spectral_tilt_db_per_octave: -6.0,
    }
}

fn high_band_noise_profile(hz: f32) -> f32 {
    ((hz - 2_400.0) / 4_200.0).clamp(0.0, 1.0).powf(0.9)
}

fn consonant_noise_profile(hz: f32) -> f32 {
    high_band_noise_profile(hz) * 0.55
        + gaussian_hz(hz, 2_600.0, 1_700.0) * 0.30
        + gaussian_hz(hz, 4_200.0, 2_000.0) * 0.15
}

fn gaussian_hz(hz: f32, center_hz: f32, width_hz: f32) -> f32 {
    let distance = (hz - center_hz) / width_hz.max(1.0);
    (-0.5 * distance * distance).exp()
}

fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

fn mel_bin_center_hz(index: usize, bins: usize) -> f32 {
    let t = if bins <= 1 {
        0.0
    } else {
        index as f32 / (bins - 1) as f32
    };
    let min_mel = hz_to_mel(MEL_MIN_HZ);
    let max_mel = hz_to_mel(MEL_MAX_HZ);
    mel_to_hz(lerp(min_mel, max_mel, t))
}

fn hz_to_mel(hz: f32) -> f32 {
    2_595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2_595.0) - 1.0)
}

fn smoothstep(alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    alpha * alpha * (3.0 - 2.0 * alpha)
}

fn lerp(start: f32, end: f32, t: f32) -> f32 {
    start + (end - start) * t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;

    #[test]
    fn phone_timed_targets_expand_to_source_filter_frames() {
        let targets = vec![PhoneTimedRenderTarget {
            phone: Phone::new_ipa("ɑ"),
            duration_ms: 96,
            f0_hz: Some(150.0),
            amplitude: 0.7,
            vibrato: None,
        }];

        let track = phone_timed_to_source_filter_track(&targets).expect("source-filter track");

        assert_eq!(track.sample_rate, SOURCE_FILTER_SAMPLE_RATE_HZ);
        assert_eq!(track.hop_ms, SOURCE_FILTER_FRAME_MS as f32);
        assert_eq!(track.frames.len(), 6);
        assert!(
            track
                .frames
                .iter()
                .all(|frame| frame.voicing.f0_hz == Some(150.0))
        );
        assert!(track.frames.iter().any(|frame| frame.filter.f1.is_some()));
    }

    #[test]
    fn source_filter_track_converts_to_aligned_mel_f0() {
        let targets = vec![
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa("s"),
                duration_ms: 48,
                f0_hz: None,
                amplitude: 0.7,
                vibrato: None,
            },
            PhoneTimedRenderTarget {
                phone: Phone::new_ipa("ɑ"),
                duration_ms: 64,
                f0_hz: Some(150.0),
                amplitude: 0.7,
                vibrato: None,
            },
        ];
        let track = phone_timed_to_source_filter_track(&targets).expect("source-filter track");

        let mel = source_filter_track_to_mel_f0(&track).expect("mel track");

        assert_eq!(mel.mel.len(), track.frames.len());
        assert_eq!(mel.f0_hz.len(), track.frames.len());
        assert_eq!(mel.voiced.len(), track.frames.len());
        assert!(mel.voiced.iter().any(|voiced| *voiced));
        assert!(mel.voiced.iter().any(|voiced| !*voiced));
        assert!(
            mel.mel
                .iter()
                .flat_map(|frame| frame.bins.iter())
                .any(|bin| *bin > 0.0)
        );
    }
}
