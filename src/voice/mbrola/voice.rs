use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use super::symbols::MbrolaSymbolMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MbrolaVoice {
    pub path: PathBuf,
    pub name: String,
    pub sample_rate: Option<u32>,
    pub symbol_map: MbrolaSymbolMap,
}

impl MbrolaVoice {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        Self::load_with_symbol_map(path, MbrolaSymbolMap::default())
    }

    pub fn load_with_symbol_map(
        path: impl Into<PathBuf>,
        symbol_map: MbrolaSymbolMap,
    ) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            bail!("MBROLA voice database not found at {}", path.display());
        }
        if !path.is_file() {
            bail!("MBROLA voice path is not a file: {}", path.display());
        }
        let name = voice_name(&path);
        let symbol_map = if symbol_map == MbrolaSymbolMap::default() {
            if name.eq_ignore_ascii_case("en1") {
                MbrolaSymbolMap::en1_starter()
            } else if name.eq_ignore_ascii_case("us3") {
                MbrolaSymbolMap::us3_starter()
            } else {
                symbol_map
            }
        } else {
            symbol_map
        };
        Ok(Self {
            path,
            name,
            sample_rate: None,
            symbol_map,
        })
    }
}

fn voice_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or("mbrola")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en1_voice_uses_rp_datapack_symbol_map() {
        let path = PathBuf::from("data/mbrola/en1/en1");
        if !path.is_file() {
            eprintln!("skipping en1 voice map test; run `just fetch`");
            return;
        }

        let voice = MbrolaVoice::load(path).expect("load en1 voice");
        assert_eq!(voice.symbol_map.map_phone("OW1").unwrap(), "@U");
        assert_eq!(voice.symbol_map.map_phone("IY1").unwrap(), "i:");
    }
}
