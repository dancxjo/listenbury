use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use thiserror::Error;

const MAGIC: &[u8; 6] = b"MBROLA";
const HEADER_LEN: usize = 27;
const VOICING_MASK: u8 = 2;
const V_REG: u8 = VOICING_MASK;
const DEFAULT_WRITER_VERSION: &[u8; 5] = b"2.060";

#[derive(Debug, Clone)]
pub struct MbrolaDatabase {
    pub path: PathBuf,
    pub version: String,
    pub sample_rate_hz: u32,
    pub mbr_period: usize,
    pub coding: u8,
    pub size_raw_bytes: usize,
    raw_offset: usize,
    pitch_marks: Vec<u8>,
    diphones: BTreeMap<(String, String), MbrolaDiphone>,
    phonemes: BTreeSet<String>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbrolaDiphone {
    pub left: String,
    pub right: String,
    pub pos_wave_samples: usize,
    pub halfseg_samples: usize,
    pub pos_pitch_mark: usize,
    pub logical_frames: usize,
    pub physical_frames: usize,
}

impl MbrolaDatabase {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MbrolaDatabaseError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| MbrolaDatabaseError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_bytes(path.to_path_buf(), bytes)
    }

    pub fn from_bytes(path: PathBuf, bytes: Vec<u8>) -> Result<Self, MbrolaDatabaseError> {
        let mut cursor = Cursor::new(&bytes);
        let magic = cursor.read_exact(6)?;
        if magic != MAGIC {
            return Err(MbrolaDatabaseError::BadMagic);
        }

        let version = String::from_utf8_lossy(cursor.read_exact(5)?).to_string();
        let nb_diphone = cursor.read_i16_le()? as usize;
        let old_size_mrk = cursor.read_u16_le()?;
        let size_pitch_marks = if old_size_mrk == 0 {
            cursor.read_i32_le()? as usize
        } else {
            old_size_mrk as usize
        };
        let size_raw_bytes = cursor.read_i32_le()? as usize;
        let sample_rate_hz = cursor.read_i16_le()? as u32;
        let mbr_period = cursor.read_u8()? as usize;
        let coding = cursor.read_u8()?;
        if coding != 1 {
            return Err(MbrolaDatabaseError::UnsupportedCoding(coding));
        }
        if cursor.pos() != HEADER_LEN {
            return Err(MbrolaDatabaseError::InternalHeaderLength(cursor.pos()));
        }

        let mut pitch_cursor = 0usize;
        let mut wave_cursor = 0usize;
        let mut entries = Vec::new();
        let mut phonemes = BTreeSet::new();
        let mut i = 0usize;
        while pitch_cursor != size_pitch_marks && i < nb_diphone {
            let left = cursor.read_zstring()?;
            let right = cursor.read_zstring()?;
            let halfseg_samples = cursor.read_i16_le()?.max(0) as usize;
            let logical_frames = cursor.read_u8()? as usize;
            let physical_frames = cursor.read_u8()? as usize;
            let diphone = MbrolaDiphone {
                left: left.clone(),
                right: right.clone(),
                pos_wave_samples: wave_cursor,
                halfseg_samples,
                pos_pitch_mark: pitch_cursor,
                logical_frames,
                physical_frames,
            };
            pitch_cursor += logical_frames;
            wave_cursor += physical_frames * mbr_period;
            phonemes.insert(left);
            phonemes.insert(right);
            entries.push(diphone);
            i += 1;
        }

        let mut diphones = BTreeMap::new();
        for diphone in entries {
            diphones.insert((diphone.left.clone(), diphone.right.clone()), diphone);
        }

        while i < nb_diphone {
            let old_left = cursor.read_zstring()?;
            let old_right = cursor.read_zstring()?;
            let Some(original) = diphones.get(&(old_left, old_right)).cloned() else {
                return Err(MbrolaDatabaseError::MissingReplacementSource);
            };
            let left = cursor.read_zstring()?;
            let right = cursor.read_zstring()?;
            let replacement = MbrolaDiphone {
                left: left.clone(),
                right: right.clone(),
                ..original
            };
            phonemes.insert(left.clone());
            phonemes.insert(right.clone());
            diphones.insert((left, right), replacement);
            i += 1;
        }

        let pitch_mark_bytes = size_pitch_marks.div_ceil(4);
        let pitch_marks = cursor.read_exact(pitch_mark_bytes)?.to_vec();
        let raw_offset = cursor.pos();
        if raw_offset + size_raw_bytes > bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }

        Ok(Self {
            path,
            version,
            sample_rate_hz,
            mbr_period,
            coding,
            size_raw_bytes,
            raw_offset,
            pitch_marks,
            diphones,
            phonemes,
            bytes,
        })
    }

    pub fn phonemes(&self) -> impl Iterator<Item = &str> {
        self.phonemes.iter().map(String::as_str)
    }

    pub fn diphone(&self, left: &str, right: &str) -> Option<&MbrolaDiphone> {
        self.diphones.get(&(left.to_string(), right.to_string()))
    }

    pub fn diphone_samples(
        &self,
        left: &str,
        right: &str,
    ) -> Result<Vec<f32>, MbrolaDatabaseError> {
        let diphone =
            self.diphone(left, right)
                .ok_or_else(|| MbrolaDatabaseError::MissingDiphone {
                    left: left.to_string(),
                    right: right.to_string(),
                })?;
        self.samples_for_diphone(diphone)
    }

    pub fn samples_for_diphone(
        &self,
        diphone: &MbrolaDiphone,
    ) -> Result<Vec<f32>, MbrolaDatabaseError> {
        let physical_frames = self.physical_frames(diphone);
        let sample_count = physical_frames * self.mbr_period;
        let start = self.raw_offset + diphone.pos_wave_samples * 2;
        let end = start + sample_count * 2;
        if end > self.bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        Ok(self.bytes[start..end]
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect())
    }

    pub fn physical_frames(&self, diphone: &MbrolaDiphone) -> usize {
        let mut total = 1usize;
        let mut pred_type = V_REG;
        for frame in 1..=diphone.logical_frames {
            let mark = self.pitch_mark(diphone, frame);
            if pred_type & VOICING_MASK == 0 && mark & VOICING_MASK != 0 {
                total += 1;
            }
            total += 1;
            pred_type = mark;
        }
        total -= 1;
        if pred_type & VOICING_MASK == 0 {
            total += 1;
        }
        total
    }

    pub fn frame_center_samples(&self, diphone: &MbrolaDiphone) -> Vec<usize> {
        let mut centers = Vec::with_capacity(diphone.logical_frames);
        let mut real_frame = 1usize;
        let mut pred_type = V_REG;
        for frame in 1..=diphone.logical_frames {
            let mark = self.pitch_mark(diphone, frame);
            if pred_type & VOICING_MASK == 0 && mark & VOICING_MASK != 0 {
                real_frame += 1;
            }
            centers.push((real_frame - 1) * self.mbr_period + self.mbr_period);
            real_frame += 1;
            pred_type = mark;
        }
        centers
    }

    fn pitch_mark(&self, diphone: &MbrolaDiphone, one_based_frame: usize) -> u8 {
        let idx = diphone.pos_pitch_mark + one_based_frame.saturating_sub(1);
        let byte = self.pitch_marks[idx / 4];
        (byte >> (2 * (idx % 4))) & 0x03
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MbrolaDatabaseUnit {
    pub left: String,
    pub right: String,
    pub samples: Vec<f32>,
    pub halfseg_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbrolaDatabaseWriteOptions {
    pub sample_rate_hz: u32,
    pub mbr_period: usize,
}

impl Default for MbrolaDatabaseWriteOptions {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            mbr_period: 80,
        }
    }
}

pub fn write_mbrola_database(
    path: impl AsRef<Path>,
    units: &[MbrolaDatabaseUnit],
    options: &MbrolaDatabaseWriteOptions,
) -> Result<(), MbrolaDatabaseError> {
    let path = path.as_ref();
    let bytes = encode_mbrola_database(units, options)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|source| MbrolaDatabaseError::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    std::fs::write(path, bytes).map_err(|source| MbrolaDatabaseError::Write {
        path: path.to_path_buf(),
        source,
    })
}

pub fn encode_mbrola_database(
    units: &[MbrolaDatabaseUnit],
    options: &MbrolaDatabaseWriteOptions,
) -> Result<Vec<u8>, MbrolaDatabaseError> {
    if units.is_empty() {
        return Err(MbrolaDatabaseError::NoDiphones);
    }
    if units.len() > i16::MAX as usize {
        return Err(MbrolaDatabaseError::TooManyDiphones(units.len()));
    }
    if options.sample_rate_hz > i16::MAX as u32 {
        return Err(MbrolaDatabaseError::UnsupportedSampleRate(
            options.sample_rate_hz,
        ));
    }
    if options.mbr_period == 0 || options.mbr_period > u8::MAX as usize {
        return Err(MbrolaDatabaseError::UnsupportedMbrPeriod(
            options.mbr_period,
        ));
    }

    let mut normalized = Vec::<EncodedUnit>::with_capacity(units.len());
    let mut total_pitch_marks = 0usize;
    let mut raw = Vec::<u8>::new();
    let mut seen = BTreeSet::<(String, String)>::new();

    for unit in units {
        if unit.left.is_empty() || unit.right.is_empty() {
            return Err(MbrolaDatabaseError::EmptyPhoneSymbol);
        }
        if unit.left.as_bytes().contains(&0) || unit.right.as_bytes().contains(&0) {
            return Err(MbrolaDatabaseError::NulPhoneSymbol);
        }
        if !seen.insert((unit.left.clone(), unit.right.clone())) {
            return Err(MbrolaDatabaseError::DuplicateDiphone {
                left: unit.left.clone(),
                right: unit.right.clone(),
            });
        }

        let logical_frames = unit.samples.len().div_ceil(options.mbr_period).max(1);
        if logical_frames > u8::MAX as usize {
            return Err(MbrolaDatabaseError::TooManyFrames {
                left: unit.left.clone(),
                right: unit.right.clone(),
                frames: logical_frames,
            });
        }
        let physical_frames = logical_frames;
        let padded_samples = logical_frames * options.mbr_period;
        for idx in 0..padded_samples {
            let sample = unit.samples.get(idx).copied().unwrap_or_default();
            raw.extend_from_slice(&f32_to_i16(sample).to_le_bytes());
        }

        let halfseg_samples = unit
            .halfseg_samples
            .min(padded_samples)
            .min(i16::MAX as usize);
        normalized.push(EncodedUnit {
            left: unit.left.clone(),
            right: unit.right.clone(),
            halfseg_samples,
            logical_frames,
            physical_frames,
        });
        total_pitch_marks += logical_frames;
    }

    let mut table = Vec::<u8>::new();
    for unit in &normalized {
        table.extend_from_slice(unit.left.as_bytes());
        table.push(0);
        table.extend_from_slice(unit.right.as_bytes());
        table.push(0);
        table.extend_from_slice(&(unit.halfseg_samples as i16).to_le_bytes());
        table.push(unit.logical_frames as u8);
        table.push(unit.physical_frames as u8);
    }

    if raw.len() > i32::MAX as usize || total_pitch_marks > i32::MAX as usize {
        return Err(MbrolaDatabaseError::DatabaseTooLarge);
    }

    let pitch_marks = pack_pitch_marks(total_pitch_marks, V_REG);
    let mut out = Vec::with_capacity(HEADER_LEN + table.len() + pitch_marks.len() + raw.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(DEFAULT_WRITER_VERSION);
    out.extend_from_slice(&(units.len() as i16).to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&(total_pitch_marks as i32).to_le_bytes());
    out.extend_from_slice(&(raw.len() as i32).to_le_bytes());
    out.extend_from_slice(&(options.sample_rate_hz as i16).to_le_bytes());
    out.push(options.mbr_period as u8);
    out.push(1);
    debug_assert_eq!(out.len(), HEADER_LEN);
    out.extend_from_slice(&table);
    out.extend_from_slice(&pitch_marks);
    out.extend_from_slice(&raw);
    Ok(out)
}

#[derive(Debug)]
struct EncodedUnit {
    left: String,
    right: String,
    halfseg_samples: usize,
    logical_frames: usize,
    physical_frames: usize,
}

fn pack_pitch_marks(count: usize, mark: u8) -> Vec<u8> {
    let mut packed = vec![0_u8; count.div_ceil(4)];
    for idx in 0..count {
        packed[idx / 4] |= (mark & 0x03) << (2 * (idx % 4));
    }
    packed
}

fn f32_to_i16(sample: f32) -> i16 {
    let scaled = sample.clamp(-1.0, 1.0) * i16::MAX as f32;
    scaled.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[derive(Debug, Error)]
pub enum MbrolaDatabaseError {
    #[error("failed to read MBROLA database {path}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write MBROLA database {path}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("not an MBROLA database")]
    BadMagic,
    #[error("unsupported MBROLA database coding {0}")]
    UnsupportedCoding(u8),
    #[error("MBROLA database ended unexpectedly")]
    UnexpectedEof,
    #[error("replacement diphone source was missing")]
    MissingReplacementSource,
    #[error("missing MBROLA diphone {left}-{right}")]
    MissingDiphone { left: String, right: String },
    #[error("internal header parser ended at byte {0}")]
    InternalHeaderLength(usize),
    #[error("cannot write MBROLA database with no diphones")]
    NoDiphones,
    #[error("too many diphones for MBROLA header: {0}")]
    TooManyDiphones(usize),
    #[error("unsupported MBROLA sample rate {0}")]
    UnsupportedSampleRate(u32),
    #[error("unsupported MBROLA mbr period {0}")]
    UnsupportedMbrPeriod(usize),
    #[error("database is too large for MBROLA 32-bit size fields")]
    DatabaseTooLarge,
    #[error("phone symbols must not be empty")]
    EmptyPhoneSymbol,
    #[error("phone symbols must not contain NUL bytes")]
    NulPhoneSymbol,
    #[error("duplicate MBROLA diphone {left}-{right}")]
    DuplicateDiphone { left: String, right: String },
    #[error("too many frames for MBROLA diphone {left}-{right}: {frames}")]
    TooManyFrames {
        left: String,
        right: String,
        frames: usize,
    },
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], MbrolaDatabaseError> {
        let end = self.pos + len;
        if end > self.bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, MbrolaDatabaseError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16_le(&mut self) -> Result<u16, MbrolaDatabaseError> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16_le(&mut self) -> Result<i16, MbrolaDatabaseError> {
        let bytes = self.read_exact(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i32_le(&mut self) -> Result<i32, MbrolaDatabaseError> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_zstring(&mut self) -> Result<String, MbrolaDatabaseError> {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != 0 {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(MbrolaDatabaseError::UnexpectedEof);
        }
        let out = String::from_utf8_lossy(&self.bytes[start..self.pos]).to_string();
        self.pos += 1;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_us3_database_when_fetched() {
        let path = PathBuf::from("data/mbrola/us3/us3");
        if !path.is_file() {
            eprintln!("skipping us3 database parse test; run `just fetch`");
            return;
        }

        let db = MbrolaDatabase::load(&path).expect("load us3 database");
        assert_eq!(db.version, "2.060");
        assert_eq!(db.sample_rate_hz, 16_000);
        assert!(db.diphone("h", "@").is_some());
        assert!(db.phonemes().any(|phoneme| phoneme == "AI"));
        let samples = db.diphone_samples("h", "@").expect("h-@ samples");
        assert!(!samples.is_empty());
        assert!(
            samples.iter().any(|sample| sample.abs() > 0.001),
            "real diphone samples should not be silent"
        );
    }

    #[test]
    fn encoded_database_round_trips_units() {
        let units = vec![
            MbrolaDatabaseUnit {
                left: "_".to_string(),
                right: "h".to_string(),
                samples: vec![0.0; 160],
                halfseg_samples: 80,
            },
            MbrolaDatabaseUnit {
                left: "h".to_string(),
                right: "@".to_string(),
                samples: (0..240).map(|i| (i as f32 * 0.05).sin() * 0.25).collect(),
                halfseg_samples: 96,
            },
        ];
        let options = MbrolaDatabaseWriteOptions::default();
        let bytes = encode_mbrola_database(&units, &options).expect("encode database");
        let db = MbrolaDatabase::from_bytes(PathBuf::from("test"), bytes).expect("parse database");

        assert_eq!(db.sample_rate_hz, 16_000);
        assert_eq!(db.mbr_period, 80);
        assert!(db.diphone("_", "h").is_some());
        let diphone = db.diphone("h", "@").expect("h-@ diphone");
        assert_eq!(diphone.halfseg_samples, 96);
        assert_eq!(db.samples_for_diphone(diphone).unwrap().len(), 240);
    }
}
