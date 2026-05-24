use anyhow::{Result, ensure};

use crate::acoustic::{
    AcousticFrameTrack, AcousticInput, AcousticModelBackend, MelFrame,
    registry::AcousticModelDescriptor,
};
use crate::voice::{
    FormantEstimation, GlottalSourceEstimate, NoiseEstimate, PhoneAcousticTarget,
    PhoneRenderTarget, PhoneTimedRenderTarget, SourceFilterFrame, SourceFilterTrack,
    VocalTractFilterEstimate, VocalTractFilterTarget, VoicingEstimate, articulate,
    default_english_phone_targets, klatt_render_targets_from_phone_timed,
    phone_timed_targets_from_articulator_plan,
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
const LOG_MEL_MIN: f32 = -8.0;
const LOG_MEL_MAX: f32 = 2.0;

pub struct SourceFilterAcousticModel;

#[derive(Debug, Clone, PartialEq)]
pub struct MelF0Track {
    pub mel: Vec<MelFrame>,
    pub f0_hz: Vec<f32>,
    pub voiced: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelTemporalDiscontinuityStats {
    pub frame_pairs: usize,
    pub mean_abs_delta: f32,
    pub p95_abs_delta: f32,
    pub max_abs_delta: f32,
}

impl SourceFilterAcousticModel {
    pub fn descriptor() -> AcousticModelDescriptor {
        AcousticModelDescriptor {
            id: "source-filter",
            notes: &[
                "Deterministic acoustic model that expands phone timing into source-filter frames, then derives mel/F0 tracks.",
                "Owns duration-controlled frame layout for HiFi-GAN and other mel/F0 vocoders.",
            ],
        }
    }
}

impl AcousticModelBackend for SourceFilterAcousticModel {
    fn id(&self) -> &'static str {
        Self::descriptor().id
    }

    fn generate(&mut self, input: AcousticInput<'_>) -> Result<AcousticFrameTrack> {
        match input {
            AcousticInput::PhoneTimed(targets) => {
                let source_filter = phone_timed_to_source_filter_track(targets)?;
                acoustic_frame_track_from_source_filter(&source_filter)
            }
            AcousticInput::Singing(plan) => {
                let acoustic_table = default_english_phone_targets();
                let articulation = articulate(plan);
                let phone_timed =
                    phone_timed_targets_from_articulator_plan(&articulation, 0.7, &acoustic_table);
                let source_filter = phone_timed_to_source_filter_track(&phone_timed)?;
                acoustic_frame_track_from_source_filter(&source_filter)
            }
            AcousticInput::SourceFilterTrack(track) => {
                acoustic_frame_track_from_source_filter(track)
            }
        }
    }
}

pub fn phone_timed_to_source_filter_track(
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

pub fn source_filter_track_to_mel_f0(track: &SourceFilterTrack) -> Result<MelF0Track> {
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
    mel = energy_mel_to_log_mel(&mel);

    ensure!(
        mel.iter()
            .any(|frame| frame.bins.iter().any(|bin| bin.is_finite())),
        "mel bridge produced no mel energy from source-filter track"
    );

    Ok(MelF0Track { mel, f0_hz, voiced })
}

pub fn mel_frame_delta_energy(mel: &[MelFrame]) -> Vec<f32> {
    if mel.len() < 2 {
        return Vec::new();
    }
    mel.windows(2)
        .map(|window| {
            let left = &window[0];
            let right = &window[1];
            let bins = left.bins.len().min(right.bins.len());
            if bins == 0 {
                return 0.0;
            }
            let delta_sum = (0..bins)
                .map(|index| (left.bins[index] - right.bins[index]).abs())
                .sum::<f32>();
            delta_sum / bins as f32
        })
        .collect()
}

pub fn summarize_mel_temporal_discontinuity(mel: &[MelFrame]) -> MelTemporalDiscontinuityStats {
    let deltas = mel_frame_delta_energy(mel);
    if deltas.is_empty() {
        return MelTemporalDiscontinuityStats {
            frame_pairs: 0,
            mean_abs_delta: 0.0,
            p95_abs_delta: 0.0,
            max_abs_delta: 0.0,
        };
    }
    let frame_pairs = deltas.len();
    let mean_abs_delta = deltas.iter().sum::<f32>() / frame_pairs as f32;
    let max_abs_delta = deltas.iter().copied().fold(0.0_f32, f32::max);
    let mut sorted = deltas;
    sorted.sort_by(|left, right| left.total_cmp(right));
    let p95_index = (((frame_pairs - 1) as f32) * 0.95).round() as usize;
    let p95_abs_delta = sorted[p95_index];
    MelTemporalDiscontinuityStats {
        frame_pairs,
        mean_abs_delta,
        p95_abs_delta,
        max_abs_delta,
    }
}

pub fn temporal_smooth_mel_frames(mel: &[MelFrame], amount: f32) -> Vec<MelFrame> {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= f32::EPSILON || mel.len() < 3 {
        return mel.to_vec();
    }
    let deltas = mel_frame_delta_energy(mel);
    let mut smoothed = mel.to_vec();
    for frame_index in 1..mel.len() - 1 {
        let bins = mel[frame_index]
            .bins
            .len()
            .min(mel[frame_index - 1].bins.len())
            .min(mel[frame_index + 1].bins.len());
        if bins == 0 {
            continue;
        }
        let local_discontinuity = deltas[frame_index - 1].max(deltas[frame_index]);
        let transition_guard = (1.0 - (local_discontinuity / 0.65)).clamp(0.0, 1.0);
        let blend = amount * transition_guard;
        if blend <= f32::EPSILON {
            continue;
        }
        for bin in 0..bins {
            let neighbor_mean =
                (mel[frame_index - 1].bins[bin] + mel[frame_index + 1].bins[bin]) * 0.5;
            smoothed[frame_index].bins[bin] =
                lerp(mel[frame_index].bins[bin], neighbor_mean, blend);
        }
    }
    smoothed
}

fn acoustic_frame_track_from_source_filter(
    track: &SourceFilterTrack,
) -> Result<AcousticFrameTrack> {
    let mel_f0 = source_filter_track_to_mel_f0(track)?;
    Ok(AcousticFrameTrack {
        mel: mel_f0.mel,
        f0_hz: mel_f0.f0_hz,
        voiced: mel_f0.voiced,
        sample_rate_hz: track.sample_rate,
        hop_samples: ((track.sample_rate as f32 * track.hop_ms) / 1_000.0).round() as usize,
    })
}

fn energy_mel_to_log_mel(mel: &[MelFrame]) -> Vec<MelFrame> {
    if mel.is_empty() {
        return Vec::new();
    }

    let mut values = Vec::with_capacity(mel.len() * MEL_BINS);
    for frame in mel {
        let frame_peak = (0..MEL_BINS)
            .map(|index| resample_bin(&frame.bins, index, MEL_BINS))
            .fold(0.0_f32, f32::max);
        let adaptive_floor = (frame_peak * 0.006).max(1.0e-5);
        for index in 0..MEL_BINS {
            let source = resample_bin(&frame.bins, index, MEL_BINS);
            values.push(
                source
                    .max(adaptive_floor)
                    .ln()
                    .clamp(LOG_MEL_MIN, LOG_MEL_MAX),
            );
        }
    }
    smooth_normalized_mel_frames(&mut values, MEL_BINS);
    values
        .chunks_exact(MEL_BINS)
        .map(|bins| MelFrame {
            bins: bins.to_vec(),
        })
        .collect()
}

fn smooth_normalized_mel_frames(values: &mut [f32], bins: usize) {
    if bins == 0 || values.len() < bins * 3 {
        return;
    }
    let original = values.to_vec();
    let frames = values.len() / bins;
    for frame_index in 1..frames - 1 {
        for bin in 0..bins {
            let index = frame_index * bins + bin;
            let neighbor_mean = (original[index - bins] + original[index + bins]) * 0.5;
            values[index] = original[index] * 0.82 + neighbor_mean * 0.18;
        }
    }
}

fn resample_bin(bins: &[f32], target_index: usize, target_bins: usize) -> f32 {
    if bins.is_empty() {
        return 0.0;
    }
    if bins.len() == 1 || target_bins <= 1 {
        return bins[0];
    }
    let source_position =
        target_index as f32 * (bins.len() - 1) as f32 / (target_bins - 1).max(1) as f32;
    let left = source_position.floor() as usize;
    let right = source_position.ceil() as usize;
    if left == right {
        bins[left]
    } else {
        lerp(bins[left], bins[right], source_position - left as f32)
    }
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
    } else if is_sibilant(target.phone.ipa.as_str()) {
        frication *= 0.48;
    } else if frication > 0.0 {
        frication *= 0.72;
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
    let mut energy = 0.16;
    for formant in [&filter.f1, &filter.f2, &filter.f3, &filter.f4]
        .into_iter()
        .flatten()
    {
        energy += gaussian_hz(
            hz,
            formant.frequency_hz,
            formant.bandwidth_hz.unwrap_or(180.0).max(110.0) * 3.8,
        ) * db_to_linear(formant.amplitude_db);
    }
    energy
}

fn source_spectrum(hz: f32, tilt_db_per_octave: f32, voiced: f32, noise: f32) -> f32 {
    let octaves = (hz.max(SPECTRAL_REFERENCE_HZ) / SPECTRAL_REFERENCE_HZ).log2();
    let tilted = db_to_linear(tilt_db_per_octave * 0.55 * octaves).clamp(0.08, 2.2);
    let broadband = high_band_noise_profile(hz).max(0.08);
    let low_periodic_tamer = (hz / (hz + 180.0)).clamp(0.18, 1.0);
    voiced * tilted * low_periodic_tamer * 0.70 + noise.max(1.0 - voiced) * broadband * 0.12
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
    high_band_noise_profile(hz) * 0.35
        + gaussian_hz(hz, 2_400.0, 1_800.0) * 0.48
        + gaussian_hz(hz, 4_000.0, 2_200.0) * 0.17
}

fn is_sibilant(ipa: &str) -> bool {
    matches!(ipa, "s" | "z" | "ʃ" | "ʒ" | "tʃ" | "dʒ")
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
                .all(|bin| bin.is_finite() && (LOG_MEL_MIN..=LOG_MEL_MAX).contains(bin))
        );
    }

    #[test]
    fn mel_discontinuity_summary_reports_frame_deltas() {
        let mel = vec![
            MelFrame {
                bins: vec![0.0, 0.0],
            },
            MelFrame {
                bins: vec![0.4, 0.2],
            },
            MelFrame {
                bins: vec![1.0, 0.8],
            },
        ];
        let deltas = mel_frame_delta_energy(&mel);
        assert_eq!(deltas.len(), 2);
        assert!(deltas[1] > deltas[0]);

        let summary = summarize_mel_temporal_discontinuity(&mel);
        assert_eq!(summary.frame_pairs, 2);
        assert!(summary.mean_abs_delta > 0.0);
        assert!(summary.p95_abs_delta >= summary.mean_abs_delta);
        assert!(summary.max_abs_delta >= summary.p95_abs_delta);
    }

    #[test]
    fn temporal_smoothing_reduces_small_framewise_jumps() {
        let mel = vec![
            MelFrame {
                bins: vec![0.0, 0.0],
            },
            MelFrame {
                bins: vec![0.6, 0.6],
            },
            MelFrame {
                bins: vec![0.0, 0.0],
            },
            MelFrame {
                bins: vec![0.1, 0.1],
            },
        ];
        let baseline = summarize_mel_temporal_discontinuity(&mel);
        let smoothed = temporal_smooth_mel_frames(&mel, 0.9);
        let after = summarize_mel_temporal_discontinuity(&smoothed);
        assert!(after.mean_abs_delta < baseline.mean_abs_delta);
        assert!(
            smoothed[2].bins[0].abs() < mel[2].bins[0].abs() + 0.1,
            "smoothing should not create larger jumps"
        );
    }
}
