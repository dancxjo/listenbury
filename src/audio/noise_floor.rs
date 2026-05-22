//! Adaptive room-noise floor tracking for framewise acoustic features.

const EPSILON: f32 = 1e-8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloorConfig {
    pub initial_rms: f32,
    pub min_rms: f32,
    pub silence_alpha: f32,
    pub noise_alpha: f32,
    pub transient_ratio: f32,
    pub speech_hold_frames: u8,
}

impl Default for NoiseFloorConfig {
    fn default() -> Self {
        Self {
            initial_rms: 0.003,
            min_rms: 0.0005,
            silence_alpha: 0.05,
            noise_alpha: 0.0125,
            transient_ratio: 3.0,
            speech_hold_frames: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloorObservation {
    pub noise_floor_rms: f32,
    pub energy_over_noise: f32,
    pub snr_db: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveNoiseFloor {
    config: NoiseFloorConfig,
    noise_floor_rms: f32,
    speech_hold_remaining: u8,
}

impl Default for AdaptiveNoiseFloor {
    fn default() -> Self {
        Self::new(NoiseFloorConfig::default())
    }
}

impl AdaptiveNoiseFloor {
    pub fn new(config: NoiseFloorConfig) -> Self {
        Self {
            noise_floor_rms: config.initial_rms.max(config.min_rms),
            config,
            speech_hold_remaining: 0,
        }
    }

    pub fn current_rms(&self) -> f32 {
        self.noise_floor_rms
    }

    pub fn observe(&mut self, rms: f32, speech_like: bool) -> NoiseFloorObservation {
        let safe_rms = rms.max(0.0);
        if speech_like {
            self.speech_hold_remaining = self.config.speech_hold_frames;
        }

        if self.speech_hold_remaining > 0 {
            self.speech_hold_remaining -= 1;
        } else {
            let alpha = if safe_rms <= self.noise_floor_rms {
                self.config.silence_alpha
            } else {
                self.config.noise_alpha
            };
            let transient_cap = self.noise_floor_rms * self.config.transient_ratio;
            let bounded_rms = safe_rms.min(transient_cap);
            self.noise_floor_rms = (self.noise_floor_rms * (1.0 - alpha) + bounded_rms * alpha)
                .max(self.config.min_rms);
        }

        let floor = self.noise_floor_rms.max(self.config.min_rms);
        NoiseFloorObservation {
            noise_floor_rms: self.noise_floor_rms,
            energy_over_noise: ((safe_rms - floor) / floor).clamp(-1.0, 32.0),
            snr_db: 20.0 * ((safe_rms + EPSILON) / (floor + EPSILON)).log10(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_speech_burst_does_not_rebase_floor() {
        let mut tracker = AdaptiveNoiseFloor::new(NoiseFloorConfig {
            initial_rms: 0.004,
            min_rms: 0.001,
            silence_alpha: 0.08,
            noise_alpha: 0.02,
            transient_ratio: 2.5,
            speech_hold_frames: 8,
        });

        for _ in 0..40 {
            tracker.observe(0.0045, false);
        }
        let before_burst = tracker.current_rms();
        for _ in 0..4 {
            tracker.observe(0.08, true);
        }
        for _ in 0..8 {
            tracker.observe(0.0048, false);
        }
        let after_burst = tracker.current_rms();
        assert!(
            (after_burst - before_burst) < 0.0015,
            "burst should not significantly lift floor: before={before_burst} after={after_burst}"
        );
    }

    #[test]
    fn sustained_room_noise_slowly_raises_floor() {
        let mut tracker = AdaptiveNoiseFloor::default();
        let initial = tracker.current_rms();
        for _ in 0..240 {
            tracker.observe(0.012, false);
        }
        let raised = tracker.current_rms();
        assert!(raised > initial);
        assert!(raised < 0.012);
    }
}
