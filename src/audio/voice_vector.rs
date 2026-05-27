use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::audio::AudioFrame;
use crate::soundscape::VoiceSignatureId;

pub const VOICE_VECTOR_DIMS: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct VoiceVectorObservation {
    pub signature_id: VoiceSignatureId,
    pub voice_node_id: String,
    pub vector: Vec<f32>,
    pub confidence: f32,
}

pub fn voice_vector_from_audio_frames(frames: &[AudioFrame]) -> Option<VoiceVectorObservation> {
    let (samples, sample_rate) = audio_frames_to_mono_samples(frames)?;
    if samples.len() < (sample_rate / 20).max(1) as usize {
        return None;
    }

    let mut vector = vec![0.0_f32; VOICE_VECTOR_DIMS];
    let mut energy = 0.0_f32;
    let mut zcr = 0.0_f32;
    let mut previous = samples[0];
    for (index, sample) in samples.iter().enumerate() {
        let value = sample.clamp(-1.0, 1.0);
        energy += value * value;
        if (value >= 0.0) != (previous >= 0.0) {
            zcr += 1.0;
        }
        let abs = value.abs();
        let bucket = ((abs * 8.0) as usize).min(7);
        vector[bucket] += 1.0;
        let phase = index as f32 / sample_rate.max(1) as f32;
        vector[8] += value * phase.sin();
        vector[9] += value * phase.cos();
        vector[10] += abs;
        vector[11] += if abs > 0.05 { 1.0 } else { 0.0 };
        previous = value;
    }

    let len = samples.len() as f32;
    let rms = (energy / len).sqrt();
    vector[12] = rms;
    vector[13] = zcr / len;
    vector[14] = duration_ms(samples.len(), sample_rate) as f32 / 10_000.0;
    vector[15] = sample_rate as f32 / 48_000.0;
    for value in &mut vector[0..12] {
        *value /= len;
    }
    normalize(&mut vector);

    let signature_id = VoiceSignatureId(stable_uuid_for_vector(&vector));
    Some(VoiceVectorObservation {
        signature_id,
        voice_node_id: format!("voice:{}", signature_id.0),
        vector,
        confidence: (rms * 8.0).clamp(0.05, 1.0),
    })
}

fn audio_frames_to_mono_samples(frames: &[AudioFrame]) -> Option<(Vec<f32>, u32)> {
    let first = frames
        .iter()
        .find(|frame| frame.sample_rate_hz > 0 && frame.channels > 0)?;
    let sample_rate = first.sample_rate_hz;
    let channels = usize::from(first.channels);
    let mut samples = Vec::new();
    for frame in frames {
        if frame.sample_rate_hz != sample_rate || usize::from(frame.channels) != channels {
            continue;
        }
        for chunk in frame.samples.chunks_exact(channels) {
            samples.push(chunk.iter().copied().sum::<f32>() / channels as f32);
        }
    }
    (!samples.is_empty()).then_some((samples, sample_rate))
}

fn duration_ms(sample_count: usize, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_count as f64 / f64::from(sample_rate)) * 1000.0).round() as u64
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in values {
            *value /= norm;
        }
    }
}

fn stable_uuid_for_vector(vector: &[f32]) -> Uuid {
    let mut hash = Sha256::new();
    for value in vector {
        hash.update(value.to_le_bytes());
    }
    let digest = hash.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

    #[test]
    fn voice_vector_is_stable_for_same_audio() {
        let frame = AudioFrame {
            captured_at: ExactTimestamp::now(),
            sample_rate_hz: 16_000,
            channels: 1,
            samples: (0..1600)
                .map(|index| ((index as f32 / 12.0).sin()) * 0.25)
                .collect(),
            voice_signatures: Vec::new(),
        };

        let first = voice_vector_from_audio_frames(std::slice::from_ref(&frame)).expect("voice");
        let second = voice_vector_from_audio_frames(&[frame]).expect("voice");

        assert_eq!(first.signature_id, second.signature_id);
        assert_eq!(first.vector.len(), VOICE_VECTOR_DIMS);
        assert!(first.voice_node_id.starts_with("voice:"));
    }
}
