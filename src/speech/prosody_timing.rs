use std::cmp::Ordering;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForcedAlignment {
    #[serde(default)]
    pub utterance_id: Option<String>,
    #[serde(default, alias = "segments")]
    pub words: Vec<AlignedWord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedWord {
    pub word: String,
    #[serde(alias = "start", alias = "start_sec")]
    pub t0: f64,
    #[serde(alias = "end", alias = "end_sec")]
    pub t1: f64,
    #[serde(default)]
    pub phones: Vec<AlignedPhone>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignedPhone {
    #[serde(alias = "phone", alias = "symbol")]
    pub p: String,
    #[serde(alias = "start", alias = "start_sec")]
    pub t0: f64,
    #[serde(alias = "end", alias = "end_sec")]
    pub t1: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PraatProsodyAnalysis {
    #[serde(default)]
    pub silences: Vec<PraatSilence>,
    #[serde(default)]
    pub nuclei: Vec<PraatNucleus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PraatSilence {
    #[serde(alias = "start", alias = "start_sec")]
    pub t0: f64,
    #[serde(alias = "end", alias = "end_sec")]
    pub t1: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PraatNucleus {
    #[serde(alias = "time", alias = "at", alias = "center")]
    pub t: f64,
    #[serde(default)]
    pub intensity_db: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyTimingPlan {
    pub utterance_id: String,
    pub segments: Vec<ProsodySegment>,
    pub breath_groups: Vec<BreathGroup>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodySegment {
    pub word: String,
    pub t0: f64,
    pub t1: f64,
    pub phones: Vec<ProsodyPhone>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub break_hint_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub break_reason: Option<BreakReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProsodyPhone {
    pub p: String,
    pub t0: f64,
    pub t1: f64,
    #[serde(default)]
    pub nucleus: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pace_target_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BreathGroup {
    pub t0: f64,
    pub t1: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakReason {
    Silence,
    Punctuation,
    LongNucleus,
    SyllableCap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProsodyTimingConfig {
    pub min_silence_break_ms: u64,
    pub min_break_ms: u64,
    pub max_break_ms: u64,
    pub comma_break_ms: u64,
    pub sentence_break_ms: u64,
    pub long_nucleus_break_ms: u64,
    pub long_nucleus_factor: f64,
    pub max_breath_group_syllables: usize,
    pub vowel_stretch_factor: f64,
}

impl Default for ProsodyTimingConfig {
    fn default() -> Self {
        Self {
            min_silence_break_ms: 250,
            min_break_ms: 120,
            max_break_ms: 320,
            comma_break_ms: 160,
            sentence_break_ms: 260,
            long_nucleus_break_ms: 140,
            long_nucleus_factor: 1.3,
            max_breath_group_syllables: 18,
            vowel_stretch_factor: 1.08,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternalAlignmentCommand {
    pub executable: PathBuf,
    pub wav_path: PathBuf,
    pub transcript_path: PathBuf,
    pub output_json_path: PathBuf,
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PraatCommandConfig {
    pub executable: PathBuf,
    pub script_path: PathBuf,
    pub wav_path: PathBuf,
    pub output_json_path: PathBuf,
}

pub fn plan_prosody_timing(
    alignment: ForcedAlignment,
    praat: PraatProsodyAnalysis,
    config: &ProsodyTimingConfig,
) -> ProsodyTimingPlan {
    let mut segments = alignment
        .words
        .into_iter()
        .filter(|word| finite_ordered_span(word.t0, word.t1))
        .map(|word| segment_from_alignment(word, &praat, config))
        .collect::<Vec<_>>();

    apply_gap_and_punctuation_breaks(&mut segments, &praat, config);
    apply_long_nucleus_breaks(&mut segments, config);
    apply_syllable_cap_breaks(&mut segments, config);

    let breath_groups = breath_groups_from_segments(&segments);
    ProsodyTimingPlan {
        utterance_id: alignment
            .utterance_id
            .unwrap_or_else(|| "utterance".to_string()),
        segments,
        breath_groups,
    }
}

pub fn forced_alignment_from_json(json: &str) -> Result<ForcedAlignment> {
    serde_json::from_str(json).context("parse forced-alignment JSON")
}

pub fn praat_analysis_from_json(json: &str) -> Result<PraatProsodyAnalysis> {
    serde_json::from_str(json).context("parse Praat prosody JSON")
}

pub fn run_external_alignment(command: &ExternalAlignmentCommand) -> Result<ForcedAlignment> {
    let status = Command::new(&command.executable)
        .arg(&command.wav_path)
        .arg(&command.transcript_path)
        .arg(&command.output_json_path)
        .args(&command.extra_args)
        .status()
        .with_context(|| {
            format!(
                "run forced aligner at {}",
                command.executable.to_string_lossy()
            )
        })?;
    if !status.success() {
        bail!("forced aligner exited with status {status}");
    }
    let json = std::fs::read_to_string(&command.output_json_path).with_context(|| {
        format!(
            "read forced-alignment output {}",
            command.output_json_path.display()
        )
    })?;
    forced_alignment_from_json(&json)
}

pub fn run_praat_analysis(command: &PraatCommandConfig) -> Result<PraatProsodyAnalysis> {
    let status = Command::new(&command.executable)
        .arg("--run")
        .arg(&command.script_path)
        .arg(&command.wav_path)
        .arg(&command.output_json_path)
        .status()
        .with_context(|| format!("run Praat at {}", command.executable.to_string_lossy()))?;
    if !status.success() {
        bail!("Praat analysis exited with status {status}");
    }
    let json = std::fs::read_to_string(&command.output_json_path).with_context(|| {
        format!(
            "read Praat prosody output {}",
            command.output_json_path.display()
        )
    })?;
    praat_analysis_from_json(&json)
}

pub fn prosody_plan_to_ssml(plan: &ProsodyTimingPlan) -> String {
    let mut ssml = String::from("<speak>");
    for (index, segment) in plan.segments.iter().enumerate() {
        if index > 0 {
            ssml.push(' ');
        }
        ssml.push_str("<mark name=\"");
        ssml.push_str(&escape_xml_attr(&format!("w{index}")));
        ssml.push_str("\"/>");
        ssml.push_str(&escape_xml_text(&segment.word));
        if let Some(ms) = segment.break_hint_ms {
            ssml.push_str("<break time=\"");
            ssml.push_str(&ms.to_string());
            ssml.push_str("ms\"/>");
        }
    }
    ssml.push_str("</speak>");
    ssml
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperTimingPlan {
    pub phonemes: Vec<PiperTimingPhone>,
    pub breaks: Vec<PiperTimingBreak>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperTimingPhone {
    pub p: String,
    pub source_word_index: usize,
    pub target_duration_ms: u64,
    pub nucleus: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiperTimingBreak {
    pub after_word_index: usize,
    pub millis: u64,
    pub reason: BreakReason,
}

pub fn prosody_plan_to_piper_timing(plan: &ProsodyTimingPlan) -> PiperTimingPlan {
    let mut phonemes = Vec::new();
    let mut breaks = Vec::new();
    for (word_index, segment) in plan.segments.iter().enumerate() {
        for phone in &segment.phones {
            phonemes.push(PiperTimingPhone {
                p: phone.p.clone(),
                source_word_index: word_index,
                target_duration_ms: phone
                    .pace_target_ms
                    .unwrap_or_else(|| seconds_to_ms((phone.t1 - phone.t0).max(0.0))),
                nucleus: phone.nucleus,
            });
        }
        if let (Some(millis), Some(reason)) = (segment.break_hint_ms, segment.break_reason) {
            breaks.push(PiperTimingBreak {
                after_word_index: word_index,
                millis,
                reason,
            });
        }
    }
    PiperTimingPlan { phonemes, breaks }
}

fn segment_from_alignment(
    word: AlignedWord,
    praat: &PraatProsodyAnalysis,
    config: &ProsodyTimingConfig,
) -> ProsodySegment {
    let phones = word
        .phones
        .into_iter()
        .filter(|phone| finite_ordered_span(phone.t0, phone.t1))
        .map(|phone| {
            let vowel = is_vowel_phone(&phone.p);
            let nucleus = vowel
                || praat
                    .nuclei
                    .iter()
                    .any(|nucleus| nucleus.t >= phone.t0 && nucleus.t <= phone.t1);
            let duration_ms = seconds_to_ms((phone.t1 - phone.t0).max(0.0));
            ProsodyPhone {
                p: phone.p,
                t0: phone.t0,
                t1: phone.t1,
                nucleus,
                pace_target_ms: if nucleus {
                    Some(((duration_ms as f64) * config.vowel_stretch_factor).round() as u64)
                } else {
                    Some(duration_ms)
                },
            }
        })
        .collect();

    ProsodySegment {
        word: word.word,
        t0: word.t0,
        t1: word.t1,
        phones,
        break_hint_ms: None,
        break_reason: None,
    }
}

fn apply_gap_and_punctuation_breaks(
    segments: &mut [ProsodySegment],
    praat: &PraatProsodyAnalysis,
    config: &ProsodyTimingConfig,
) {
    for index in 0..segments.len() {
        let mut hint = punctuation_break(&segments[index].word, config)
            .map(|ms| (ms, BreakReason::Punctuation));

        let next_start = segments.get(index + 1).map(|next| next.t0);
        if let Some(next_start) = next_start {
            let gap_ms = seconds_to_ms((next_start - segments[index].t1).max(0.0));
            if gap_ms >= config.min_silence_break_ms {
                let silence_ms = gap_ms.clamp(config.min_break_ms, config.max_break_ms);
                hint = stronger_break(hint, (silence_ms, BreakReason::Silence));
            }
            for silence in &praat.silences {
                let overlaps_boundary =
                    silence.t0 <= next_start && silence.t1 >= segments[index].t1;
                let silence_ms = seconds_to_ms((silence.t1 - silence.t0).max(0.0));
                if overlaps_boundary && silence_ms >= config.min_silence_break_ms {
                    hint = stronger_break(
                        hint,
                        (
                            silence_ms.clamp(config.min_break_ms, config.max_break_ms),
                            BreakReason::Silence,
                        ),
                    );
                }
            }
        }

        if let Some((millis, reason)) = hint {
            segments[index].break_hint_ms = Some(millis);
            segments[index].break_reason = Some(reason);
        }
    }
}

fn apply_long_nucleus_breaks(segments: &mut [ProsodySegment], config: &ProsodyTimingConfig) {
    let mut durations = segments
        .iter()
        .flat_map(|segment| {
            segment
                .phones
                .iter()
                .filter(|phone| phone.nucleus)
                .map(|phone| phone.t1 - phone.t0)
        })
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .collect::<Vec<_>>();
    if durations.is_empty() {
        return;
    }
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = durations[durations.len() / 2];
    let threshold = median * config.long_nucleus_factor;

    for segment in segments {
        let has_long_nucleus = segment
            .phones
            .iter()
            .any(|phone| phone.nucleus && (phone.t1 - phone.t0) >= threshold);
        if has_long_nucleus && segment.break_hint_ms.is_none() && !ends_with_sentence(&segment.word)
        {
            segment.break_hint_ms = Some(config.long_nucleus_break_ms);
            segment.break_reason = Some(BreakReason::LongNucleus);
        }
    }
}

fn apply_syllable_cap_breaks(segments: &mut [ProsodySegment], config: &ProsodyTimingConfig) {
    if config.max_breath_group_syllables == 0 {
        return;
    }
    let mut syllables = 0usize;
    let mut last_soft_break = None;

    for index in 0..segments.len() {
        syllables += segments[index]
            .phones
            .iter()
            .filter(|phone| phone.nucleus)
            .count()
            .max(1);

        if segments[index].break_hint_ms.is_some() {
            syllables = 0;
            last_soft_break = None;
            continue;
        }

        if syllables >= config.max_breath_group_syllables {
            let target = last_soft_break.unwrap_or(index);
            if segments[target].break_hint_ms.is_none() {
                segments[target].break_hint_ms = Some(config.min_break_ms);
                segments[target].break_reason = Some(BreakReason::SyllableCap);
            }
            syllables = 0;
            last_soft_break = None;
        } else if syllables >= config.max_breath_group_syllables / 2 {
            last_soft_break = Some(index);
        }
    }
}

fn breath_groups_from_segments(segments: &[ProsodySegment]) -> Vec<BreathGroup> {
    let Some(first) = segments.first() else {
        return Vec::new();
    };

    let mut groups = Vec::new();
    let mut group_start = first.t0;
    for (index, segment) in segments.iter().enumerate() {
        if segment.break_hint_ms.is_some() {
            groups.push(BreathGroup {
                t0: group_start,
                t1: segment.t1,
            });
            if let Some(next) = segments.get(index + 1) {
                group_start = next.t0;
            }
        }
    }

    let last = segments.last().expect("segments is non-empty");
    if groups
        .last()
        .map(|group| group.t1 < last.t1)
        .unwrap_or(true)
    {
        groups.push(BreathGroup {
            t0: group_start,
            t1: last.t1,
        });
    }
    groups
}

fn punctuation_break(word: &str, config: &ProsodyTimingConfig) -> Option<u64> {
    let trimmed = word.trim();
    let last = trimmed.chars().next_back()?;
    match last {
        ',' | ';' | ':' => Some(config.comma_break_ms),
        '.' | '!' | '?' => Some(config.sentence_break_ms),
        _ => None,
    }
}

fn stronger_break(
    current: Option<(u64, BreakReason)>,
    candidate: (u64, BreakReason),
) -> Option<(u64, BreakReason)> {
    match current {
        Some(current) if current.0 >= candidate.0 => Some(current),
        _ => Some(candidate),
    }
}

fn finite_ordered_span(t0: f64, t1: f64) -> bool {
    t0.is_finite() && t1.is_finite() && t1 >= t0
}

fn seconds_to_ms(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).round() as u64
}

fn ends_with_sentence(word: &str) -> bool {
    word.trim()
        .chars()
        .next_back()
        .map(|ch| matches!(ch, '.' | '!' | '?'))
        .unwrap_or(false)
}

fn is_vowel_phone(phone: &str) -> bool {
    let normalized = phone
        .trim()
        .trim_matches(|ch: char| ch.is_ascii_digit() || ch == '"' || ch == '\'')
        .to_ascii_uppercase();
    matches!(
        normalized.as_str(),
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    ) || phone.chars().any(|ch| {
        matches!(
            ch,
            'a' | 'e' | 'i' | 'o' | 'u' | 'ɑ' | 'æ' | 'ʌ' | 'ɔ' | 'ɛ' | 'ɝ' | 'ɪ' | 'ʊ' | 'ə' | 'ɚ'
        )
    })
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_xml_attr(value: &str) -> String {
    escape_xml_text(value).replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_marks_nuclei_breaks_and_breath_groups() {
        let alignment = ForcedAlignment {
            utterance_id: Some("utt-1".to_string()),
            words: vec![
                AlignedWord {
                    word: "going".to_string(),
                    t0: 1.24,
                    t1: 1.52,
                    phones: vec![
                        AlignedPhone {
                            p: "g".to_string(),
                            t0: 1.24,
                            t1: 1.29,
                        },
                        AlignedPhone {
                            p: "OW1".to_string(),
                            t0: 1.29,
                            t1: 1.45,
                        },
                    ],
                },
                AlignedWord {
                    word: "now.".to_string(),
                    t0: 1.86,
                    t1: 2.10,
                    phones: vec![AlignedPhone {
                        p: "aʊ".to_string(),
                        t0: 1.91,
                        t1: 2.04,
                    }],
                },
            ],
        };
        let praat = PraatProsodyAnalysis {
            silences: vec![PraatSilence { t0: 1.52, t1: 1.86 }],
            nuclei: vec![PraatNucleus {
                t: 1.34,
                intensity_db: Some(71.0),
            }],
        };

        let plan = plan_prosody_timing(alignment, praat, &ProsodyTimingConfig::default());

        assert_eq!(plan.utterance_id, "utt-1");
        assert!(plan.segments[0].phones[1].nucleus);
        assert_eq!(plan.segments[0].break_reason, Some(BreakReason::Silence));
        assert_eq!(plan.segments[0].break_hint_ms, Some(320));
        assert_eq!(
            plan.segments[1].break_reason,
            Some(BreakReason::Punctuation)
        );
        assert_eq!(plan.breath_groups.len(), 2);
    }

    #[test]
    fn emitters_produce_ssml_marks_and_piper_breaks() {
        let plan = ProsodyTimingPlan {
            utterance_id: "utt".to_string(),
            segments: vec![ProsodySegment {
                word: "hello".to_string(),
                t0: 0.0,
                t1: 0.4,
                phones: vec![ProsodyPhone {
                    p: "EH1".to_string(),
                    t0: 0.1,
                    t1: 0.25,
                    nucleus: true,
                    pace_target_ms: Some(162),
                }],
                break_hint_ms: Some(160),
                break_reason: Some(BreakReason::Punctuation),
            }],
            breath_groups: vec![BreathGroup { t0: 0.0, t1: 0.4 }],
        };

        let ssml = prosody_plan_to_ssml(&plan);
        assert!(ssml.contains("<mark name=\"w0\"/>hello"));
        assert!(ssml.contains("<break time=\"160ms\"/>"));

        let piper = prosody_plan_to_piper_timing(&plan);
        assert_eq!(piper.phonemes[0].target_duration_ms, 162);
        assert_eq!(piper.breaks[0].after_word_index, 0);
    }
}
