use sha2::{Digest, Sha256};

use crate::vision::VisionFrame;

pub const IMAGE_VECTOR_DIMS: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ImageVectorObservation {
    pub image_id: String,
    pub vector: Vec<f32>,
}

pub fn vectorize_rgba_frame(frame: &VisionFrame) -> Option<ImageVectorObservation> {
    let width = usize::try_from(frame.width).ok()?;
    let height = usize::try_from(frame.height).ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    let expected = width.checked_mul(height)?.checked_mul(4)?;
    if frame.bytes.len() < expected {
        return None;
    }

    let mut hash = Sha256::new();
    hash.update(frame.width.to_le_bytes());
    hash.update(frame.height.to_le_bytes());
    hash.update(&frame.bytes[..expected]);
    let image_id = format!("image:{}", hex_prefix(&hash.finalize(), 16));

    let mut bins = vec![0.0_f32; IMAGE_VECTOR_DIMS];
    let mut count = 0.0_f32;
    for pixel in frame.bytes[..expected].chunks_exact(4) {
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        let a = pixel[3] as f32 / 255.0;
        let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
        let saturation = (r.max(g).max(b) - r.min(g).min(b)).clamp(0.0, 1.0);
        let x = (count as usize % width) as f32 / width.max(1) as f32;
        let y = (count as usize / width) as f32 / height.max(1) as f32;

        accumulate(&mut bins[0..8], luma, 1.0);
        accumulate(&mut bins[8..16], saturation, 1.0);
        accumulate(&mut bins[16..24], r, 0.34);
        accumulate(&mut bins[16..24], g, 0.33);
        accumulate(&mut bins[16..24], b, 0.33);
        bins[24] += luma;
        bins[25] += saturation;
        bins[26] += a;
        bins[27] += x * luma;
        bins[28] += y * luma;
        bins[29] += (x - 0.5).abs() * luma;
        bins[30] += (y - 0.5).abs() * luma;
        bins[31] += (r - b).abs();
        count += 1.0;
    }

    if count <= 0.0 {
        return None;
    }
    for value in &mut bins[24..] {
        *value /= count;
    }
    normalize(&mut bins);

    Some(ImageVectorObservation {
        image_id,
        vector: bins,
    })
}

fn accumulate(bins: &mut [f32], value: f32, weight: f32) {
    if bins.is_empty() {
        return;
    }
    let index = ((value.clamp(0.0, 1.0) * bins.len() as f32) as usize).min(bins.len() - 1);
    bins[index] += weight;
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in values {
            *value /= norm;
        }
    }
}

fn hex_prefix(bytes: &[u8], len: usize) -> String {
    bytes
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 0x0f])
        .take(len)
        .map(|nibble| char::from_digit(u32::from(nibble), 16).expect("hex digit"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ExactTimestamp;

    #[test]
    fn rgba_vectorization_is_stable_and_normalized() {
        let frame = VisionFrame {
            captured_at: ExactTimestamp::now(),
            width: 2,
            height: 2,
            bytes: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
        };

        let first = vectorize_rgba_frame(&frame).expect("vector");
        let second = vectorize_rgba_frame(&frame).expect("vector");

        assert_eq!(first.image_id, second.image_id);
        assert_eq!(first.vector.len(), IMAGE_VECTOR_DIMS);
        let norm = first
            .vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
