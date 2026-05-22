use std::collections::HashMap;

use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use thiserror::Error;

use crate::prosody::note_target::{
    MidiNote, NoteArticulation, NoteDuration, NoteTarget, PitchTarget, TimePoint, Velocity,
};

const DEFAULT_TEMPO_US_PER_QUARTER: u32 = 500_000;
const MICROS_PER_MILLISECOND: u128 = 1_000;
const ROUND_TO_NEAREST_MILLISECOND_OFFSET: u128 = 500;

#[derive(Debug, Clone, Copy, Default)]
pub struct MidiImportOptions {
    pub track: Option<usize>,
    pub channel: Option<u8>,
}

#[derive(Debug, Error)]
pub enum MidiImportError {
    #[error("failed to parse MIDI data: {0}")]
    Parse(#[from] midly::Error),
    #[error("unsupported MIDI timing mode; only metrical (ticks per quarter note) is supported")]
    UnsupportedTiming,
    #[error("invalid track index {track}; file contains {track_count} tracks")]
    InvalidTrack { track: usize, track_count: usize },
    #[error("invalid MIDI channel {0}; expected 0..=15")]
    InvalidChannel(u8),
    #[error("malformed MIDI: {0} note(s) were started but never ended")]
    UnclosedNotes(usize),
}

#[derive(Debug, Clone, Copy)]
struct NoteSpan {
    key: u8,
    velocity: u8,
    start_tick: u64,
    end_tick: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveNote {
    velocity: u8,
    start_tick: u64,
}

pub fn note_targets_from_midi_bytes(
    bytes: &[u8],
    options: MidiImportOptions,
) -> Result<Vec<NoteTarget>, MidiImportError> {
    let smf = Smf::parse(bytes)?;
    let ppq = match smf.header.timing {
        Timing::Metrical(ppq) => ppq.as_int() as u32,
        Timing::Timecode(_, _) => return Err(MidiImportError::UnsupportedTiming),
    };

    let selected_channel = options.channel.map(|channel| {
        if channel <= 15 {
            Ok(channel)
        } else {
            Err(MidiImportError::InvalidChannel(channel))
        }
    });
    let selected_channel = selected_channel.transpose()?;

    let track_indices: Vec<usize> = if let Some(track) = options.track {
        if track >= smf.tracks.len() {
            return Err(MidiImportError::InvalidTrack {
                track,
                track_count: smf.tracks.len(),
            });
        }
        vec![track]
    } else {
        (0..smf.tracks.len()).collect()
    };

    let tempo_events = collect_tempo_events(&smf);
    let note_spans = collect_note_spans(&smf, &track_indices, selected_channel)?;
    let mut tick_to_millis = HashMap::new();
    for span in &note_spans {
        tick_to_millis
            .entry(span.start_tick)
            .or_insert_with(|| ticks_to_millis(span.start_tick, ppq, &tempo_events));
        tick_to_millis
            .entry(span.end_tick)
            .or_insert_with(|| ticks_to_millis(span.end_tick, ppq, &tempo_events));
    }

    let mut note_targets: Vec<NoteTarget> = note_spans
        .iter()
        .map(|span| {
            let onset_ms = *tick_to_millis
                .get(&span.start_tick)
                .unwrap_or_else(|| panic!("start tick {} should be cached", span.start_tick));
            let end_ms = *tick_to_millis
                .get(&span.end_tick)
                .unwrap_or_else(|| panic!("end tick {} should be cached", span.end_tick));
            NoteTarget {
                pitch: PitchTarget::new(
                    MidiNote::new(span.key)
                        .unwrap_or_else(|| panic!("MIDI key {} must be <= 127", span.key)),
                ),
                onset: TimePoint::from_millis(onset_ms),
                duration: NoteDuration::from_millis(end_ms.saturating_sub(onset_ms)),
                velocity: Velocity::new(span.velocity).unwrap_or_else(|| {
                    panic!("MIDI velocity {} must be in 1..=127", span.velocity)
                }),
                articulation: NoteArticulation::Neutral,
            }
        })
        .collect();

    note_targets.sort_by_key(|note| {
        (
            note.onset.millis,
            note.pitch.note.as_u8(),
            note.duration.millis,
            note.velocity.as_u8(),
        )
    });

    Ok(note_targets)
}

fn collect_tempo_events(smf: &Smf<'_>) -> Vec<(u64, u32)> {
    let mut tempo_events = Vec::new();

    for track in &smf.tracks {
        let mut tick = 0_u64;
        for event in track {
            tick += event.delta.as_int() as u64;
            if let TrackEventKind::Meta(MetaMessage::Tempo(tempo)) = event.kind {
                tempo_events.push((tick, tempo.as_int()));
            }
        }
    }

    tempo_events.sort_by_key(|(tick, _)| *tick);

    let mut deduped = Vec::new();
    for (tick, tempo) in tempo_events {
        if let Some((last_tick, last_tempo)) = deduped.last_mut() {
            if *last_tick == tick {
                *last_tempo = tempo;
                continue;
            }
        }
        deduped.push((tick, tempo));
    }
    deduped
}

fn collect_note_spans(
    smf: &Smf<'_>,
    track_indices: &[usize],
    selected_channel: Option<u8>,
) -> Result<Vec<NoteSpan>, MidiImportError> {
    let mut note_spans = Vec::new();

    for &track_index in track_indices {
        let mut tick = 0_u64;
        let mut active_notes: HashMap<(u8, u8), ActiveNote> = HashMap::new();

        for event in &smf.tracks[track_index] {
            tick += event.delta.as_int() as u64;
            let TrackEventKind::Midi { channel, message } = event.kind else {
                continue;
            };

            let channel = channel.as_int();
            if selected_channel.is_some_and(|selected| selected != channel) {
                continue;
            }

            match message {
                MidiMessage::NoteOn { key, vel } => {
                    let key = key.as_int();
                    let vel = vel.as_int();
                    if vel == 0 {
                        if let Some(active) = active_notes.remove(&(channel, key)) {
                            note_spans.push(NoteSpan {
                                key,
                                velocity: active.velocity,
                                start_tick: active.start_tick,
                                end_tick: tick,
                            });
                        }
                    } else if let Some(active) = active_notes.insert(
                        (channel, key),
                        ActiveNote {
                            velocity: vel,
                            start_tick: tick,
                        },
                    ) {
                        note_spans.push(NoteSpan {
                            key,
                            velocity: active.velocity,
                            start_tick: active.start_tick,
                            end_tick: tick,
                        });
                    }
                }
                MidiMessage::NoteOff { key, .. } => {
                    let key = key.as_int();
                    if let Some(active) = active_notes.remove(&(channel, key)) {
                        note_spans.push(NoteSpan {
                            key,
                            velocity: active.velocity,
                            start_tick: active.start_tick,
                            end_tick: tick,
                        });
                    }
                }
                _ => {}
            }
        }

        if !active_notes.is_empty() {
            return Err(MidiImportError::UnclosedNotes(active_notes.len()));
        }
    }

    Ok(note_spans)
}

fn ticks_to_millis(ticks: u64, ppq: u32, tempo_events: &[(u64, u32)]) -> u64 {
    let mut previous_tick = 0_u64;
    let mut current_tempo_us_per_quarter = DEFAULT_TEMPO_US_PER_QUARTER;
    let mut total_microseconds = 0_u128;

    for &(tempo_tick, tempo_value) in tempo_events {
        if tempo_tick > ticks {
            break;
        }
        if tempo_tick > previous_tick {
            total_microseconds += ((tempo_tick - previous_tick) as u128
                * current_tempo_us_per_quarter as u128)
                / ppq as u128;
            previous_tick = tempo_tick;
        }
        current_tempo_us_per_quarter = tempo_value;
    }

    if ticks > previous_tick {
        total_microseconds +=
            ((ticks - previous_tick) as u128 * current_tempo_us_per_quarter as u128) / ppq as u128;
    }

    ((total_microseconds + ROUND_TO_NEAREST_MILLISECOND_OFFSET) / MICROS_PER_MILLISECOND) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use midly::num::{u4, u7, u15, u24, u28};
    use midly::{Format, Header, TrackEvent};

    fn smf_bytes(timing: Timing, tracks: Vec<Vec<TrackEvent<'static>>>, format: Format) -> Vec<u8> {
        let smf = Smf {
            header: Header::new(format, timing),
            tracks,
        };
        let mut out = Vec::new();
        smf.write_std(&mut out).expect("MIDI should encode");
        out
    }

    fn note_on(delta: u32, channel: u8, key: u8, vel: u8) -> TrackEvent<'static> {
        TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(channel),
                message: MidiMessage::NoteOn {
                    key: u7::new(key),
                    vel: u7::new(vel),
                },
            },
        }
    }

    fn note_off(delta: u32, channel: u8, key: u8, vel: u8) -> TrackEvent<'static> {
        TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(channel),
                message: MidiMessage::NoteOff {
                    key: u7::new(key),
                    vel: u7::new(vel),
                },
            },
        }
    }

    fn tempo(delta: u32, micros_per_quarter: u32) -> TrackEvent<'static> {
        TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(micros_per_quarter))),
        }
    }

    fn end_of_track(delta: u32) -> TrackEvent<'static> {
        TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        }
    }

    #[test]
    fn maps_default_tempo_and_velocity_zero_note_off() {
        let bytes = smf_bytes(
            Timing::Metrical(u15::new(480)),
            vec![vec![
                note_on(0, 0, 60, 100),
                note_on(480, 0, 60, 0),
                end_of_track(0),
            ]],
            Format::SingleTrack,
        );

        let notes = note_targets_from_midi_bytes(&bytes, MidiImportOptions::default()).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch.note.as_u8(), 60);
        assert_eq!(notes[0].velocity.as_u8(), 100);
        assert_eq!(notes[0].onset.millis, 0);
        assert_eq!(notes[0].duration.millis, 500);
    }

    #[test]
    fn applies_explicit_tempo_changes_when_mapping_ticks() {
        let bytes = smf_bytes(
            Timing::Metrical(u15::new(480)),
            vec![
                vec![tempo(0, 1_000_000), end_of_track(0)],
                vec![
                    note_on(480, 0, 64, 90),
                    note_off(480, 0, 64, 0),
                    end_of_track(0),
                ],
            ],
            Format::Parallel,
        );

        let notes = note_targets_from_midi_bytes(&bytes, MidiImportOptions::default()).unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch.note.as_u8(), 64);
        assert_eq!(notes[0].onset.millis, 1_000);
        assert_eq!(notes[0].duration.millis, 1_000);
    }

    #[test]
    fn filters_track_and_channel() {
        let bytes = smf_bytes(
            Timing::Metrical(u15::new(480)),
            vec![
                vec![
                    note_on(0, 1, 65, 80),
                    note_off(480, 1, 65, 0),
                    note_on(0, 0, 60, 100),
                    note_off(480, 0, 60, 0),
                    end_of_track(0),
                ],
                vec![
                    note_on(0, 1, 72, 70),
                    note_off(480, 1, 72, 0),
                    end_of_track(0),
                ],
            ],
            Format::Parallel,
        );

        let notes = note_targets_from_midi_bytes(
            &bytes,
            MidiImportOptions {
                track: Some(0),
                channel: Some(1),
            },
        )
        .unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch.note.as_u8(), 65);
        assert_eq!(notes[0].velocity.as_u8(), 80);
    }

    #[test]
    fn returns_parse_error_for_malformed_data() {
        let err = note_targets_from_midi_bytes(b"not a midi", MidiImportOptions::default())
            .expect_err("invalid bytes should fail");
        assert!(matches!(err, MidiImportError::Parse(_)));
    }

    #[test]
    fn returns_invalid_track_error_for_out_of_range_track() {
        let bytes = smf_bytes(
            Timing::Metrical(u15::new(480)),
            vec![vec![end_of_track(0)]],
            Format::SingleTrack,
        );
        let err = note_targets_from_midi_bytes(
            &bytes,
            MidiImportOptions {
                track: Some(3),
                channel: None,
            },
        )
        .expect_err("out-of-range track should fail");
        assert!(matches!(
            err,
            MidiImportError::InvalidTrack {
                track: 3,
                track_count: 1
            }
        ));
    }

    #[test]
    fn returns_unclosed_notes_error_when_note_off_is_missing() {
        let bytes = smf_bytes(
            Timing::Metrical(u15::new(480)),
            vec![vec![note_on(0, 0, 60, 100), end_of_track(0)]],
            Format::SingleTrack,
        );
        let err = note_targets_from_midi_bytes(&bytes, MidiImportOptions::default())
            .expect_err("missing note-off should fail");
        assert!(matches!(err, MidiImportError::UnclosedNotes(1)));
    }
}
