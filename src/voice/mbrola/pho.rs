use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::symbols::MbrolaSymbolMap;
use crate::speech::prosody_timing::ProsodyTimingPlan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneTimedPlan {
    pub phones: Vec<MbrolaPhone>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MbrolaPhone {
    pub symbol: String,
    pub duration_ms: u32,
    #[serde(default)]
    pub pitch_targets: Vec<MbrolaPitchTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MbrolaPitchTarget {
    pub percent: u8,
    pub hz: f32,
}

impl PhoneTimedPlan {
    pub fn new(phones: Vec<MbrolaPhone>) -> Self {
        Self { phones }
    }

    pub fn from_source_symbols(
        phones: impl IntoIterator<Item = MbrolaPhone>,
        symbol_map: &MbrolaSymbolMap,
    ) -> Result<Self> {
        let mut mapped = Vec::new();
        for mut phone in phones {
            phone.symbol = symbol_map.map_phone(&phone.symbol)?;
            mapped.push(phone);
        }
        Ok(Self { phones: mapped })
    }

    pub fn total_duration_ms(&self) -> u64 {
        self.phones
            .iter()
            .map(|phone| u64::from(phone.duration_ms))
            .sum()
    }
}

impl MbrolaPhone {
    pub fn new(symbol: impl Into<String>, duration_ms: u32) -> Self {
        Self {
            symbol: symbol.into(),
            duration_ms,
            pitch_targets: Vec::new(),
        }
    }

    pub fn with_pitch_targets(mut self, pitch_targets: Vec<MbrolaPitchTarget>) -> Self {
        self.pitch_targets = pitch_targets;
        self
    }
}

pub fn phone_timed_plan_to_pho(plan: &PhoneTimedPlan) -> String {
    let mut out = String::new();
    for phone in &plan.phones {
        let _ = write!(out, "{} {}", phone.symbol, phone.duration_ms);
        for target in &phone.pitch_targets {
            let hz = if target.hz.fract().abs() < 0.005 {
                format!("{:.0}", target.hz)
            } else {
                format!("{:.2}", target.hz)
            };
            let _ = write!(out, " {} {}", target.percent.min(100), hz);
        }
        out.push('\n');
    }
    out
}

pub fn read_pho_file(path: &Path) -> Result<PhoneTimedPlan> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read MBROLA .pho file at {}", path.display()))?;
    parse_pho(&text).with_context(|| format!("failed to parse MBROLA .pho at {}", path.display()))
}

pub fn write_pho_file(path: &Path, plan: &PhoneTimedPlan) -> Result<()> {
    std::fs::write(path, phone_timed_plan_to_pho(plan))
        .with_context(|| format!("failed to write MBROLA .pho at {}", path.display()))
}

pub fn prosody_timing_plan_to_phone_timed_plan(
    plan: &ProsodyTimingPlan,
    symbol_map: &MbrolaSymbolMap,
) -> Result<PhoneTimedPlan> {
    let mut phones = Vec::new();
    for segment in &plan.segments {
        for phone in &segment.phones {
            let duration_ms = phone
                .pace_target_ms
                .unwrap_or_else(|| ((phone.t1 - phone.t0).max(0.0) * 1000.0).round() as u64)
                .clamp(1, u64::from(u32::MAX)) as u32;
            let mut mbrola_phone = MbrolaPhone::new(symbol_map.map_phone(&phone.p)?, duration_ms);
            if phone.nucleus {
                mbrola_phone.pitch_targets = vec![
                    MbrolaPitchTarget {
                        percent: 0,
                        hz: 125.0,
                    },
                    MbrolaPitchTarget {
                        percent: 60,
                        hz: 135.0,
                    },
                    MbrolaPitchTarget {
                        percent: 100,
                        hz: 128.0,
                    },
                ];
            }
            phones.push(mbrola_phone);
        }
        if let Some(break_ms) = segment.break_hint_ms {
            phones.push(MbrolaPhone::new(
                "_",
                break_ms.clamp(1, u64::from(u32::MAX)) as u32,
            ));
        }
    }
    Ok(PhoneTimedPlan::new(phones))
}

pub fn parse_pho(text: &str) -> std::result::Result<PhoneTimedPlan, MbrolaPhoParseError> {
    let mut phones = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 2 {
            return Err(MbrolaPhoParseError::MissingDuration { line: line_number });
        }
        let duration_ms =
            parts[1]
                .parse::<u32>()
                .map_err(|_| MbrolaPhoParseError::BadDuration {
                    line: line_number,
                    value: parts[1].to_string(),
                })?;
        let pitch_parts = &parts[2..];
        if pitch_parts.len() % 2 != 0 {
            return Err(MbrolaPhoParseError::OddPitchTargetCount { line: line_number });
        }
        let mut pitch_targets = Vec::new();
        for pair in pitch_parts.chunks_exact(2) {
            let percent =
                pair[0]
                    .parse::<u8>()
                    .map_err(|_| MbrolaPhoParseError::BadPitchPercent {
                        line: line_number,
                        value: pair[0].to_string(),
                    })?;
            let hz = pair[1]
                .parse::<f32>()
                .map_err(|_| MbrolaPhoParseError::BadPitchHz {
                    line: line_number,
                    value: pair[1].to_string(),
                })?;
            pitch_targets.push(MbrolaPitchTarget { percent, hz });
        }
        phones.push(MbrolaPhone {
            symbol: parts[0].to_string(),
            duration_ms,
            pitch_targets,
        });
    }
    Ok(PhoneTimedPlan { phones })
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum MbrolaPhoParseError {
    #[error("line {line}: missing duration")]
    MissingDuration { line: usize },
    #[error("line {line}: invalid duration `{value}`")]
    BadDuration { line: usize, value: String },
    #[error("line {line}: pitch targets must be percent/hz pairs")]
    OddPitchTargetCount { line: usize },
    #[error("line {line}: invalid pitch target percent `{value}`")]
    BadPitchPercent { line: usize, value: String },
    #[error("line {line}: invalid pitch target Hz `{value}`")]
    BadPitchHz { line: usize, value: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pho_round_trip_preserves_pitch_targets() {
        let plan = PhoneTimedPlan::new(vec![
            MbrolaPhone::new("h", 80),
            MbrolaPhone::new("@", 120).with_pitch_targets(vec![
                MbrolaPitchTarget {
                    percent: 0,
                    hz: 120.0,
                },
                MbrolaPitchTarget {
                    percent: 50,
                    hz: 130.0,
                },
            ]),
        ]);

        let pho = phone_timed_plan_to_pho(&plan);
        assert_eq!(pho, "h 80\n@ 120 0 120 50 130\n");
        assert_eq!(parse_pho(&pho).unwrap(), plan);
    }
}
