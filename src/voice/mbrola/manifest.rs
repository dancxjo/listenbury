//! Voice database provenance and licensing manifest.
//!
//! A [`VoiceManifest`] describes where a voice database came from, who holds
//! the copyright, and what redistribution rights the project has.  It is
//! intentionally separate from the binary voice database format so that
//! licensing information can be kept alongside without touching the upstream
//! file.
//!
//! Manifests can be loaded from TOML (`manifest.toml`) or JSON
//! (`manifest.json`) files placed next to a voice database, or constructed
//! programmatically for in-process voice validation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Provenance and licensing record for a MBROLA (or compatible) voice database.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub struct VoiceManifest {
    /// Short, human-readable name for the voice (e.g. `"us3"`).
    pub name: Option<String>,

    /// Path to the binary voice database file, relative to the manifest.
    pub path: Option<PathBuf>,

    /// Human-readable hint for how to obtain the voice database
    /// (e.g. a URL or `just fetch` instruction).
    pub download_hint: Option<String>,

    /// Canonical upstream URL for the voice database.
    pub upstream_url: Option<String>,

    /// Short identifier for the license (e.g. `"CC BY-SA 4.0"`, `"non-commercial"`).
    pub license_name: Option<String>,

    /// URL to the full license text.
    pub license_url: Option<String>,

    /// Whether the voice database may be redistributed.
    ///
    /// `None` means unknown / not specified.
    pub redistribution_allowed: Option<bool>,

    /// Whether commercial use is permitted.
    ///
    /// `None` means unknown / not specified.
    pub commercial_allowed: Option<bool>,

    /// Whether attribution to the original author is required.
    ///
    /// `None` means unknown / not specified.
    pub attribution_required: Option<bool>,

    /// Free-form notes about provenance, restrictions, or known issues.
    pub notes: Option<String>,
}

impl VoiceManifest {
    /// Load a manifest from a TOML file.
    pub fn load_toml(path: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        toml::from_str(&text).map_err(|source| ManifestError::ParseToml {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Load a manifest from a JSON file.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| ManifestError::ParseJson {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Try to load a manifest from the conventional locations next to
    /// `voice_path`: first `manifest.toml`, then `manifest.json` in the same
    /// directory.  Returns `Ok(None)` when no manifest file exists, rather than
    /// an error.
    pub fn find_for_voice(voice_path: impl AsRef<Path>) -> Result<Option<Self>, ManifestError> {
        let dir = voice_path.as_ref().parent().unwrap_or(Path::new("."));

        let toml_path = dir.join("manifest.toml");
        if toml_path.is_file() {
            return Self::load_toml(&toml_path).map(Some);
        }

        let json_path = dir.join("manifest.json");
        if json_path.is_file() {
            return Self::load_json(&json_path).map(Some);
        }

        Ok(None)
    }

    /// Returns a list of human-readable licensing warnings for this voice.
    ///
    /// An empty vec means no licensing concerns were detected.  Warnings are
    /// informational: callers decide whether to treat them as fatal.
    pub fn license_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.license_name.is_none() && self.license_url.is_none() {
            warnings.push("voice manifest does not specify a license".to_string());
        }
        if self.redistribution_allowed == Some(false) {
            warnings.push(
                "voice license does not permit redistribution; do not bundle this database"
                    .to_string(),
            );
        }
        if self.redistribution_allowed.is_none() {
            warnings.push(
                "voice manifest does not state whether redistribution is allowed".to_string(),
            );
        }
        if self.commercial_allowed.is_none() {
            warnings.push(
                "voice manifest does not state whether commercial use is allowed".to_string(),
            );
        }
        if self.attribution_required == Some(true) && self.upstream_url.is_none() {
            warnings.push(
                "attribution is required but the manifest does not include an upstream URL"
                    .to_string(),
            );
        }

        warnings
    }
}

/// Errors that can occur while loading a manifest file.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("failed to read manifest {path}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse TOML manifest {path}")]
    ParseToml {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("failed to parse JSON manifest {path}")]
    ParseJson {
        path: PathBuf,
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_toml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f
    }

    #[test]
    fn manifest_parses_full_toml() {
        let tmp = write_toml(
            r#"
name = "us3"
upstream_url = "https://example.com/us3"
license_name = "non-commercial"
license_url = "https://example.com/license"
redistribution_allowed = false
commercial_allowed = false
attribution_required = true
notes = "For testing only."
"#,
        );
        let m = VoiceManifest::load_toml(tmp.path()).expect("should parse");
        assert_eq!(m.name.as_deref(), Some("us3"));
        assert_eq!(m.redistribution_allowed, Some(false));
        assert_eq!(m.commercial_allowed, Some(false));
        assert_eq!(m.attribution_required, Some(true));
    }

    #[test]
    fn manifest_parses_minimal_toml() {
        let tmp = write_toml("name = \"en1\"\n");
        let m = VoiceManifest::load_toml(tmp.path()).expect("should parse minimal");
        assert_eq!(m.name.as_deref(), Some("en1"));
        assert!(m.license_name.is_none());
    }

    #[test]
    fn license_warnings_emitted_when_no_license_specified() {
        let m = VoiceManifest::default();
        let warnings = m.license_warnings();
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("does not specify a license")),
            "expected a 'no license' warning, got: {warnings:?}"
        );
    }

    #[test]
    fn license_warnings_include_redistribution_false() {
        let m = VoiceManifest {
            license_name: Some("proprietary".to_string()),
            license_url: Some("https://example.com".to_string()),
            redistribution_allowed: Some(false),
            commercial_allowed: Some(false),
            attribution_required: Some(false),
            ..VoiceManifest::default()
        };
        let warnings = m.license_warnings();
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("does not permit redistribution")),
            "expected redistribution warning, got: {warnings:?}"
        );
    }

    #[test]
    fn license_warnings_absent_when_fully_specified_and_allowed() {
        let m = VoiceManifest {
            license_name: Some("CC0".to_string()),
            license_url: Some("https://creativecommons.org/publicdomain/zero/1.0/".to_string()),
            redistribution_allowed: Some(true),
            commercial_allowed: Some(true),
            attribution_required: Some(false),
            ..VoiceManifest::default()
        };
        let warnings = m.license_warnings();
        assert!(
            warnings.is_empty(),
            "expected no warnings for fully-licensed voice, got: {warnings:?}"
        );
    }

    #[test]
    fn find_for_voice_returns_none_when_no_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let voice_path = dir.path().join("myvoice");
        let result = VoiceManifest::find_for_voice(&voice_path).expect("should not error");
        assert!(result.is_none());
    }

    #[test]
    fn find_for_voice_loads_toml_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let voice_path = dir.path().join("myvoice");
        let manifest_path = dir.path().join("manifest.toml");
        std::fs::write(&manifest_path, "name = \"myvoice\"\n").unwrap();
        let m = VoiceManifest::find_for_voice(&voice_path)
            .expect("should not error")
            .expect("should find manifest.toml");
        assert_eq!(m.name.as_deref(), Some("myvoice"));
    }
}
