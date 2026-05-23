use std::time::Duration;

use crate::voice::tract::targets::{
    GlottalSourceTarget, PhoneRenderTarget, VocalTractFilterTarget,
};

use super::params::{KlattFrameParams, interpolate};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhoneClass {
    Vowel,
    GlideLiquid,
    Nasal,
    Fricative,
    Stop,
    Affricate,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Stable,
    Transition,
    Closure,
    Burst,
    Frication,
    Aspiration,
    Murmur,
}

#[derive(Debug, Clone)]
struct EventTarget {
    target: PhoneRenderTarget,
    class: PhoneClass,
    kind: EventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrajectoryConfig {
    pub frame_ms: u64,
    pub blend_ms: u64,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            frame_ms: 5,
            blend_ms: 15,
        }
    }
}

pub(crate) fn trajectory_targets_from_phones(
    targets: &[PhoneRenderTarget],
    config: TrajectoryConfig,
) -> Vec<PhoneRenderTarget> {
    if targets.is_empty() {
        return Vec::new();
    }

    let event_targets = expand_phone_events(targets);
    let mut out = Vec::new();
    let frame_ms = config.frame_ms.max(1);
    for (idx, event) in event_targets.iter().enumerate() {
        let chunks = duration_chunks(event.target.duration_ms.max(1), frame_ms);
        let blend_chunks = ((config.blend_ms + frame_ms - 1) / frame_ms) as usize;
        let left = KlattFrameParams::from_target(&event.target);
        let right = event_targets
            .get(idx + 1)
            .map(|next| KlattFrameParams::from_target(&next.target));
        let blend_allowed = right
            .as_ref()
            .is_some_and(|_| should_blend_phone_boundary(event, &event_targets[idx + 1]));
        let mut t_ms = 0u64;
        for (chunk_idx, duration_ms) in chunks.iter().copied().enumerate() {
            let mut params = left.clone();
            if let Some(next) = right.as_ref() {
                if blend_allowed {
                    let blend_start = chunks.len().saturating_sub(blend_chunks.max(1));
                    if chunk_idx >= blend_start {
                        let rel = (chunk_idx - blend_start + 1) as f32;
                        let den = (chunks.len() - blend_start + 1) as f32;
                        let alpha =
                            blend_alpha(event.class, event_targets[idx + 1].class, rel / den);
                        params = interpolate(&left, next, alpha);
                    }
                }
            }
            if let (Some(base_f0), Some(vibrato)) = (params.f0_hz, params.vibrato) {
                params.f0_hz = Some(vibrato.apply_to_hz(base_f0, Duration::from_millis(t_ms)));
            }
            out.push(to_target(&event.target, params, duration_ms));
            t_ms = t_ms.saturating_add(duration_ms);
        }
    }
    out
}

fn to_target(
    base: &PhoneRenderTarget,
    params: KlattFrameParams,
    duration_ms: u64,
) -> PhoneRenderTarget {
    PhoneRenderTarget {
        phone: base.phone.clone(),
        duration_ms,
        f0_hz: params.f0_hz,
        amplitude: params.amplitude,
        vibrato: params.vibrato,
        source: Some(GlottalSourceTarget {
            breathiness: params.breathiness,
            open_quotient: base.source.as_ref().map(|s| s.open_quotient).unwrap_or(0.5),
            spectral_tilt_db_per_octave: params.spectral_tilt_db_per_octave,
        }),
        filter: params.filter,
    }
}

fn should_blend_phone_boundary(left: &EventTarget, right: &EventTarget) -> bool {
    // Keep deliberate articulatory edges for obstruent events. Stops, bursts,
    // frication onsets, and aspiration are intentionally discontinuous so
    // consonant landmarks survive smoothing.
    if matches!(
        left.kind,
        EventKind::Closure | EventKind::Burst | EventKind::Frication | EventKind::Aspiration
    ) || matches!(
        right.kind,
        EventKind::Closure | EventKind::Burst | EventKind::Frication | EventKind::Aspiration
    ) {
        return false;
    }

    matches!(
        (left.class, right.class),
        (PhoneClass::Vowel, PhoneClass::Vowel)
            | (PhoneClass::Vowel, PhoneClass::GlideLiquid)
            | (PhoneClass::GlideLiquid, PhoneClass::Vowel)
            | (PhoneClass::GlideLiquid, PhoneClass::GlideLiquid)
            | (PhoneClass::Nasal, PhoneClass::Vowel)
            | (PhoneClass::Vowel, PhoneClass::Nasal)
            | (PhoneClass::Nasal, PhoneClass::GlideLiquid)
            | (PhoneClass::GlideLiquid, PhoneClass::Nasal)
            | (PhoneClass::Nasal, PhoneClass::Nasal)
    )
}

fn blend_alpha(left: PhoneClass, right: PhoneClass, alpha: f32) -> f32 {
    let curved = smoothstep(alpha.clamp(0.0, 1.0));
    match (left, right) {
        (PhoneClass::GlideLiquid, PhoneClass::Vowel)
        | (PhoneClass::Vowel, PhoneClass::GlideLiquid)
        | (PhoneClass::GlideLiquid, PhoneClass::GlideLiquid) => curved.powf(0.7),
        _ => curved,
    }
}

fn expand_phone_events(targets: &[PhoneRenderTarget]) -> Vec<EventTarget> {
    let mut out = Vec::new();
    for target in targets {
        out.extend(explicit_phone_events(target));
    }
    out
}

fn explicit_phone_events(target: &PhoneRenderTarget) -> Vec<EventTarget> {
    match classify_phone(target.phone.ipa.as_str()) {
        PhoneClass::Stop => stop_events(target),
        PhoneClass::Affricate => affricate_events(target),
        PhoneClass::Fricative => fricative_events(target),
        PhoneClass::Nasal => nasal_events(target),
        class => vec![EventTarget {
            target: target.clone(),
            class,
            kind: EventKind::Stable,
        }],
    }
}

fn classify_phone(ipa: &str) -> PhoneClass {
    if matches!(
        ipa,
        "i" | "ɪ"
            | "e"
            | "ɛ"
            | "æ"
            | "ə"
            | "ʌ"
            | "ɑ"
            | "ɔ"
            | "o"
            | "ʊ"
            | "u"
            | "aɪ"
            | "ɑɪ"
            | "aʊ"
            | "eɪ"
            | "oʊ"
            | "ɔɪ"
    ) {
        return PhoneClass::Vowel;
    }
    if matches!(ipa, "j" | "w" | "l" | "ɹ") {
        return PhoneClass::GlideLiquid;
    }
    if matches!(ipa, "m" | "n" | "ŋ") {
        return PhoneClass::Nasal;
    }
    if matches!(ipa, "tʃ" | "dʒ") {
        return PhoneClass::Affricate;
    }
    if matches!(ipa, "p" | "b" | "t" | "d" | "k" | "ɡ") {
        return PhoneClass::Stop;
    }
    if matches!(ipa, "f" | "v" | "θ" | "ð" | "s" | "z" | "ʃ" | "ʒ" | "h") {
        return PhoneClass::Fricative;
    }
    PhoneClass::Other
}

fn stop_events(target: &PhoneRenderTarget) -> Vec<EventTarget> {
    let ipa = target.phone.ipa.as_str();
    let voiced = target.f0_hz.is_some();
    let mut out = Vec::new();
    let d = partition_duration(target.duration_ms.max(4), &[20, 40, 12, 28]);

    let mut approach = target.clone();
    approach.duration_ms = d[0];
    approach.amplitude = target.amplitude * 0.6;
    approach.filter = Some(transition_filter_for_place(ipa, target.filter.as_ref()));
    out.push(EventTarget {
        target: approach,
        class: PhoneClass::Stop,
        kind: EventKind::Transition,
    });

    let mut closure = target.clone();
    closure.duration_ms = d[1];
    closure.amplitude = 0.02;
    closure.filter = None;
    closure.f0_hz = if voiced { target.f0_hz } else { None };
    closure.source = Some(update_source(
        &closure,
        if voiced { 0.2 } else { 0.0 },
        -12.0,
    ));
    out.push(EventTarget {
        target: closure,
        class: PhoneClass::Stop,
        kind: EventKind::Closure,
    });

    let mut burst = target.clone();
    burst.duration_ms = d[2];
    burst.amplitude = target.amplitude.max(0.85);
    burst.f0_hz = None;
    burst.filter = Some(burst_filter_for_place(ipa, target.filter.as_ref()));
    burst.source = Some(update_source(&burst, 1.0, -1.0));
    out.push(EventTarget {
        target: burst,
        class: PhoneClass::Stop,
        kind: EventKind::Burst,
    });

    let mut tail = target.clone();
    tail.duration_ms = d[3];
    tail.amplitude = target.amplitude * if voiced { 0.5 } else { 0.65 };
    tail.f0_hz = if voiced { target.f0_hz } else { None };
    tail.filter = Some(release_filter_for_place(ipa, target.filter.as_ref()));
    tail.source = Some(update_source(
        &tail,
        if voiced { 0.45 } else { 0.95 },
        if voiced { -6.0 } else { -2.0 },
    ));
    out.push(EventTarget {
        target: tail,
        class: PhoneClass::Stop,
        kind: EventKind::Aspiration,
    });

    out
}

fn affricate_events(target: &PhoneRenderTarget) -> Vec<EventTarget> {
    let voiced = target.f0_hz.is_some();
    let d = partition_duration(target.duration_ms.max(3), &[35, 12, 53]);
    let ipa = target.phone.ipa.as_str();

    let mut closure = target.clone();
    closure.duration_ms = d[0];
    closure.amplitude = 0.02;
    closure.filter = None;
    closure.f0_hz = if voiced { target.f0_hz } else { None };
    closure.source = Some(update_source(
        &closure,
        if voiced { 0.2 } else { 0.0 },
        -12.0,
    ));

    let mut burst = target.clone();
    burst.duration_ms = d[1];
    burst.amplitude = target.amplitude.max(0.85);
    burst.f0_hz = None;
    burst.filter = Some(burst_filter_for_place(ipa, target.filter.as_ref()));
    burst.source = Some(update_source(&burst, 1.0, -1.0));

    let mut frication = target.clone();
    frication.duration_ms = d[2];
    frication.amplitude = target.amplitude * 0.85;
    frication.filter = Some(frication_filter_for_phone(ipa, target.filter.as_ref()));
    frication.f0_hz = if voiced { target.f0_hz } else { None };
    frication.source = Some(update_source(
        &frication,
        if voiced { 0.6 } else { 0.98 },
        if voiced { -4.0 } else { -2.0 },
    ));

    vec![
        EventTarget {
            target: closure,
            class: PhoneClass::Affricate,
            kind: EventKind::Closure,
        },
        EventTarget {
            target: burst,
            class: PhoneClass::Affricate,
            kind: EventKind::Burst,
        },
        EventTarget {
            target: frication,
            class: PhoneClass::Affricate,
            kind: EventKind::Frication,
        },
    ]
}

fn fricative_events(target: &PhoneRenderTarget) -> Vec<EventTarget> {
    let voiced = target.f0_hz.is_some();
    let d = partition_duration(target.duration_ms.max(3), &[20, 60, 20]);
    let ipa = target.phone.ipa.as_str();

    let mut onset = target.clone();
    onset.duration_ms = d[0];
    onset.amplitude = target.amplitude * 0.75;
    onset.filter = Some(frication_filter_for_phone(ipa, target.filter.as_ref()));
    onset.source = Some(update_source(
        &onset,
        if voiced { 0.55 } else { 0.98 },
        if voiced { -4.0 } else { -2.0 },
    ));

    let mut body = target.clone();
    body.duration_ms = d[1];
    body.filter = Some(frication_filter_for_phone(ipa, target.filter.as_ref()));
    body.source = Some(update_source(
        &body,
        if voiced { 0.45 } else { 0.98 },
        if voiced { -4.0 } else { -2.0 },
    ));

    let mut offset = target.clone();
    offset.duration_ms = d[2];
    offset.amplitude = target.amplitude * 0.6;
    offset.filter = Some(frication_filter_for_phone(ipa, target.filter.as_ref()));
    offset.source = Some(update_source(
        &offset,
        if voiced { 0.5 } else { 0.95 },
        if voiced { -5.0 } else { -3.0 },
    ));

    vec![
        EventTarget {
            target: onset,
            class: PhoneClass::Fricative,
            kind: EventKind::Transition,
        },
        EventTarget {
            target: body,
            class: PhoneClass::Fricative,
            kind: EventKind::Frication,
        },
        EventTarget {
            target: offset,
            class: PhoneClass::Fricative,
            kind: EventKind::Transition,
        },
    ]
}

fn nasal_events(target: &PhoneRenderTarget) -> Vec<EventTarget> {
    let d = partition_duration(target.duration_ms.max(3), &[20, 60, 20]);

    let mut attenuation = target.clone();
    attenuation.duration_ms = d[0];
    attenuation.amplitude = target.amplitude * 0.6;
    attenuation.filter = attenuation.filter.as_ref().map(nasalize_filter);
    attenuation.source = Some(update_source(&attenuation, 0.15, -10.0));

    let mut murmur = target.clone();
    murmur.duration_ms = d[1];
    murmur.amplitude = target.amplitude * 0.5;
    murmur.filter = murmur.filter.as_ref().map(nasalize_filter);
    murmur.source = Some(update_source(&murmur, 0.1, -12.0));

    let mut transition = target.clone();
    transition.duration_ms = d[2];
    transition.amplitude = target.amplitude * 0.75;
    transition.filter = transition.filter.as_ref().map(nasalize_filter);
    transition.source = Some(update_source(&transition, 0.12, -9.0));

    vec![
        EventTarget {
            target: attenuation,
            class: PhoneClass::Nasal,
            kind: EventKind::Transition,
        },
        EventTarget {
            target: murmur,
            class: PhoneClass::Nasal,
            kind: EventKind::Murmur,
        },
        EventTarget {
            target: transition,
            class: PhoneClass::Nasal,
            kind: EventKind::Transition,
        },
    ]
}

fn partition_duration(total_ms: u64, weights: &[u64]) -> Vec<u64> {
    let mut total = total_ms.max(weights.len() as u64);
    let sum_weights: u64 = weights.iter().sum();
    let mut out: Vec<u64> = weights
        .iter()
        .map(|weight| ((total * *weight) / sum_weights).max(1))
        .collect();
    let mut assigned: u64 = out.iter().sum();

    while assigned < total {
        if let Some(last) = out.last_mut() {
            *last += 1;
            assigned += 1;
        }
    }
    while assigned > total {
        if let Some((idx, _)) = out.iter().enumerate().rfind(|(_, d)| **d > 1) {
            out[idx] -= 1;
            assigned -= 1;
        } else {
            total += 1;
        }
    }
    out
}

fn update_source(
    target: &PhoneRenderTarget,
    breathiness: f32,
    tilt_db_per_octave: f32,
) -> GlottalSourceTarget {
    let base = target.source.clone().unwrap_or(GlottalSourceTarget {
        breathiness: 0.05,
        open_quotient: 0.5,
        spectral_tilt_db_per_octave: -6.0,
    });
    GlottalSourceTarget {
        breathiness,
        open_quotient: base.open_quotient,
        spectral_tilt_db_per_octave: tilt_db_per_octave,
    }
}

fn fallback_filter() -> VocalTractFilterTarget {
    VocalTractFilterTarget {
        f1_hz: 400.0,
        f1_bw_hz: 150.0,
        f1_amp_db: -10.0,
        f2_hz: 2200.0,
        f2_bw_hz: 350.0,
        f2_amp_db: -5.0,
        f3_hz: 3200.0,
        f3_bw_hz: 450.0,
        f3_amp_db: -6.0,
        f4_hz: None,
        f4_bw_hz: None,
        f4_amp_db: None,
    }
}

fn transition_filter_for_place(
    ipa: &str,
    fallback: Option<&VocalTractFilterTarget>,
) -> VocalTractFilterTarget {
    let mut filter = fallback.cloned().unwrap_or_else(fallback_filter);
    match ipa {
        "p" | "b" => {
            filter.f2_hz = 900.0;
            filter.f3_hz = 2200.0;
        }
        "t" | "d" => {
            filter.f2_hz = 2200.0;
            filter.f3_hz = 3300.0;
        }
        "k" | "ɡ" => {
            filter.f2_hz = 1800.0;
            filter.f3_hz = 2500.0;
        }
        _ => {}
    }
    filter
}

fn burst_filter_for_place(
    ipa: &str,
    fallback: Option<&VocalTractFilterTarget>,
) -> VocalTractFilterTarget {
    let mut filter = fallback.cloned().unwrap_or_else(fallback_filter);
    match ipa {
        "p" | "b" => {
            filter.f2_hz = 1100.0;
            filter.f2_bw_hz = 500.0;
            filter.f3_hz = 2300.0;
            filter.f3_bw_hz = 700.0;
        }
        "t" | "d" => {
            filter.f2_hz = 3200.0;
            filter.f2_bw_hz = 280.0;
            filter.f3_hz = 4000.0;
            filter.f3_bw_hz = 380.0;
        }
        "k" | "ɡ" => {
            filter.f2_hz = 1900.0;
            filter.f2_bw_hz = 420.0;
            filter.f3_hz = 2600.0;
            filter.f3_bw_hz = 500.0;
        }
        "tʃ" | "dʒ" => {
            filter.f2_hz = 2500.0;
            filter.f2_bw_hz = 520.0;
            filter.f3_hz = 3400.0;
            filter.f3_bw_hz = 620.0;
        }
        _ => {}
    }
    filter
}

fn release_filter_for_place(
    ipa: &str,
    fallback: Option<&VocalTractFilterTarget>,
) -> VocalTractFilterTarget {
    let mut filter = burst_filter_for_place(ipa, fallback);
    filter.f2_bw_hz += 80.0;
    filter.f3_bw_hz += 80.0;
    filter
}

fn frication_filter_for_phone(
    ipa: &str,
    fallback: Option<&VocalTractFilterTarget>,
) -> VocalTractFilterTarget {
    let mut filter = fallback.cloned().unwrap_or_else(fallback_filter);
    match ipa {
        "s" | "z" => {
            filter.f2_hz = 3600.0;
            filter.f2_bw_hz = 260.0;
            filter.f3_hz = 4300.0;
            filter.f3_bw_hz = 320.0;
        }
        "ʃ" | "ʒ" | "tʃ" | "dʒ" => {
            filter.f2_hz = 2500.0;
            filter.f2_bw_hz = 520.0;
            filter.f3_hz = 3400.0;
            filter.f3_bw_hz = 620.0;
        }
        "k" | "ɡ" => {
            filter.f2_hz = 1900.0;
            filter.f3_hz = 2600.0;
        }
        "p" | "b" | "m" | "w" | "f" | "v" => {
            filter.f2_hz = 1200.0;
            filter.f3_hz = 2400.0;
        }
        "h" => {
            filter.f2_hz = 1500.0;
            filter.f2_bw_hz = 650.0;
            filter.f3_hz = 2500.0;
            filter.f3_bw_hz = 760.0;
        }
        _ => {}
    }
    filter
}

fn nasalize_filter(filter: &VocalTractFilterTarget) -> VocalTractFilterTarget {
    let mut f = filter.clone();
    f.f1_amp_db -= 4.0;
    f.f2_amp_db -= 5.0;
    f.f3_amp_db -= 6.0;
    f.f2_bw_hz *= 1.2;
    f.f3_bw_hz *= 1.2;
    f
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

fn smoothstep(alpha: f32) -> f32 {
    alpha * alpha * (3.0 - 2.0 * alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::{Phone, PhoneString};
    use crate::prosody::vibrato::Vibrato;
    use crate::voice::tract::targets::{
        default_english_phone_targets, phone_render_targets_from_string,
    };

    #[test]
    fn interpolation_creates_intermediate_f0_near_boundary() {
        let table = default_english_phone_targets();
        let left = PhoneString {
            phones: vec![Phone::new_ipa("i"), Phone::new_ipa("e")],
        };
        let mut targets = phone_render_targets_from_string(&left, Some(220.0), 0.8, &table);
        targets[1].f0_hz = Some(330.0);
        targets[0].duration_ms = 100;
        targets[1].duration_ms = 100;
        let traj = trajectory_targets_from_phones(
            &targets,
            TrajectoryConfig {
                frame_ms: 10,
                blend_ms: 20,
            },
        );
        assert!(
            traj.iter()
                .any(|t| t.f0_hz.is_some_and(|f0| f0 > 220.0 && f0 < 330.0)),
            "expected interpolated F0 values in blended trajectory"
        );
    }

    #[test]
    fn vibrato_modulates_sustained_phone_f0() {
        let table = default_english_phone_targets();
        let mut targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("æ")],
            },
            Some(220.0),
            0.8,
            &table,
        );
        targets[0].duration_ms = 400;
        targets[0].vibrato = Some(Vibrato::new(5.0, 30.0, Duration::ZERO, Duration::ZERO, 0.0));
        let traj = trajectory_targets_from_phones(&targets, TrajectoryConfig::default());
        let min = traj
            .iter()
            .filter_map(|target| target.f0_hz)
            .fold(f32::INFINITY, f32::min);
        let max = traj
            .iter()
            .filter_map(|target| target.f0_hz)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max > min,
            "vibrato should modulate F0 over trajectory frames"
        );
    }

    #[test]
    fn voiceless_stop_contains_closure_and_burst_events() {
        let table = default_english_phone_targets();
        let mut targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("t")],
            },
            Some(220.0),
            0.8,
            &table,
        );
        targets[0].duration_ms = 80;
        let events = explicit_phone_events(&targets[0]);
        assert!(
            events
                .iter()
                .any(|event| event.kind == EventKind::Closure && event.target.amplitude <= 0.03),
            "stop should include low-energy closure"
        );
        assert!(
            events
                .iter()
                .any(|event| event.kind == EventKind::Burst && event.target.amplitude >= 0.8),
            "stop should include burst event"
        );
    }

    #[test]
    fn affricate_contains_closure_and_frication_tail() {
        let table = default_english_phone_targets();
        let mut targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("tʃ")],
            },
            Some(220.0),
            0.8,
            &table,
        );
        targets[0].duration_ms = 90;
        let events = explicit_phone_events(&targets[0]);
        assert!(
            events.iter().any(|event| event.kind == EventKind::Closure),
            "affricate should include closure"
        );
        assert!(
            events
                .iter()
                .any(|event| event.kind == EventKind::Frication),
            "affricate should include frication tail"
        );
    }

    #[test]
    fn fricative_keeps_sustained_noise_interval() {
        let table = default_english_phone_targets();
        let mut targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("s")],
            },
            None,
            0.8,
            &table,
        );
        targets[0].duration_ms = 90;
        let events = explicit_phone_events(&targets[0]);
        let frication = events
            .iter()
            .find(|event| event.kind == EventKind::Frication)
            .expect("fricative should include sustained frication");
        assert!(
            frication.target.duration_ms >= 40,
            "frication body should be sustained"
        );
    }

    #[test]
    fn smoothing_preserves_stop_burst_discontinuity() {
        let table = default_english_phone_targets();
        let mut targets = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![
                    Phone::new_ipa("ɑ"),
                    Phone::new_ipa("t"),
                    Phone::new_ipa("ɑ"),
                ],
            },
            Some(200.0),
            0.7,
            &table,
        );
        for target in &mut targets {
            target.duration_ms = 80;
        }
        let traj = trajectory_targets_from_phones(
            &targets,
            TrajectoryConfig {
                frame_ms: 5,
                blend_ms: 20,
            },
        );
        assert!(
            traj.iter().any(|target| target.amplitude <= 0.03),
            "trajectory should preserve stop closure dip"
        );
        assert!(
            traj.iter().any(|target| target.amplitude >= 0.7),
            "trajectory should preserve release burst energy"
        );
    }

    #[test]
    fn voiced_and_voiceless_fricatives_differ_in_source_behavior() {
        let table = default_english_phone_targets();
        let voiced = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("z")],
            },
            Some(190.0),
            0.7,
            &table,
        );
        let voiceless = phone_render_targets_from_string(
            &PhoneString {
                phones: vec![Phone::new_ipa("s")],
            },
            Some(190.0),
            0.7,
            &table,
        );
        let voiced_events = explicit_phone_events(&voiced[0]);
        let voiceless_events = explicit_phone_events(&voiceless[0]);
        let voiced_body = voiced_events
            .iter()
            .find(|event| event.kind == EventKind::Frication)
            .expect("voiced fricative body");
        let voiceless_body = voiceless_events
            .iter()
            .find(|event| event.kind == EventKind::Frication)
            .expect("voiceless fricative body");
        assert!(voiced_body.target.f0_hz.is_some());
        assert!(voiceless_body.target.f0_hz.is_none());
        assert!(
            voiced_body
                .target
                .source
                .as_ref()
                .expect("voiced source")
                .breathiness
                < voiceless_body
                    .target
                    .source
                    .as_ref()
                    .expect("voiceless source")
                    .breathiness
        );
    }
}
