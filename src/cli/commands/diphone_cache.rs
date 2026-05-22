use anyhow::{Context, Result};

use crate::cli::{DiphoneCacheBuildCommand, DiphoneCacheCommand, DiphoneCacheForgeCommand};

pub(crate) fn run_diphone_cache(command: DiphoneCacheCommand) -> Result<()> {
    match command {
        DiphoneCacheCommand::Forge(cmd) => run_forge(cmd),
        DiphoneCacheCommand::Build(cmd) => run_build(cmd),
    }
}

#[cfg(feature = "tts-riper")]
fn run_forge(cmd: DiphoneCacheForgeCommand) -> Result<()> {
    use listenbury::voice::diphone::{DiphoneCache, ForgeSettings};
    use listenbury::voice::diphone::forge::forge_diphone;
    use listenbury::mouth::riper::{PiperVoiceConfig, RiperBackend};

    let config_json = std::fs::read_to_string(&cmd.config).with_context(|| {
        format!(
            "failed to read Piper voice config {}",
            cmd.config.display()
        )
    })?;
    let config = PiperVoiceConfig::from_json_str(&config_json).with_context(|| {
        format!(
            "failed to parse Piper voice config {}",
            cmd.config.display()
        )
    })?;

    let mut backend = RiperBackend::load(&cmd.model, config)
        .with_context(|| format!("failed to load Riper backend from {}", cmd.model.display()))?;

    let cache = DiphoneCache::open(&cmd.cache_dir).with_context(|| {
        format!("failed to open diphone cache at {}", cmd.cache_dir.display())
    })?;

    let settings = ForgeSettings::default();
    let forged = forge_diphone(&mut backend, &cmd.left, &cmd.right, &settings).with_context(
        || {
            format!(
                "failed to forge diphone {}-{}",
                cmd.left, cmd.right
            )
        },
    )?;

    // Store in cache
    use listenbury::voice::diphone::cache::{CacheEntryMetadata, CacheKey};
    use listenbury::voice::diphone::forge::{
        CARRIER_STRATEGY_VERSION, FORGE_SETTINGS_VERSION, NORMALIZATION_VERSION,
        fingerprint_config, fingerprint_path,
    };

    let key = CacheKey {
        model_fingerprint: fingerprint_path(backend.model_path()),
        config_fingerprint: fingerprint_config(backend.config()),
        speaker_id: String::new(),
        left: cmd.left.clone(),
        right: cmd.right.clone(),
        carrier_strategy_version: CARRIER_STRATEGY_VERSION.to_string(),
        forge_settings_version: FORGE_SETTINGS_VERSION.to_string(),
        sample_rate_hz: backend.config().sample_rate_hz,
        normalization_version: NORMALIZATION_VERSION.to_string(),
    };

    let meta = CacheEntryMetadata {
        key: key.clone(),
        generated_at: forged
            .unit
            .metadata
            .forge_provenance
            .as_ref()
            .map(|p| p.generated_at.clone())
            .unwrap_or_default(),
        carrier_sequence: forged.carrier_sequence.clone(),
        halfseg_samples: forged.unit.halfseg_samples,
        segmentation_confidence: forged.segmentation_confidence,
        sample_count: forged.unit.samples.len(),
        license_note: concat!(
            "Generated locally from a Piper ONNX model. ",
            "Do not redistribute without checking the model license."
        )
        .to_string(),
    };

    cache.store(&key, &forged.unit, meta)?;

    println!(
        "Forged diphone {}-{}: {} samples, halfseg={}, confidence={:.2}, cache={}",
        cmd.left,
        cmd.right,
        forged.unit.samples.len(),
        forged.unit.halfseg_samples,
        forged.segmentation_confidence,
        cmd.cache_dir.display(),
    );
    Ok(())
}

#[cfg(not(feature = "tts-riper"))]
fn run_forge(_cmd: DiphoneCacheForgeCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-riper` feature")
}

#[cfg(feature = "tts-riper")]
fn run_build(cmd: DiphoneCacheBuildCommand) -> Result<()> {
    use listenbury::voice::diphone::{DiphoneCache, ForgeSettings};
    use listenbury::voice::diphone::cache::{CacheEntryMetadata, CacheKey};
    use listenbury::voice::diphone::forge::{
        CARRIER_STRATEGY_VERSION, FORGE_SETTINGS_VERSION, NORMALIZATION_VERSION,
        fingerprint_config, fingerprint_path, forge_diphone,
    };
    use listenbury::mouth::riper::{PiperVoiceConfig, RiperBackend};

    let config_json = std::fs::read_to_string(&cmd.config).with_context(|| {
        format!(
            "failed to read Piper voice config {}",
            cmd.config.display()
        )
    })?;
    let config = PiperVoiceConfig::from_json_str(&config_json).with_context(|| {
        format!(
            "failed to parse Piper voice config {}",
            cmd.config.display()
        )
    })?;

    let mut backend = RiperBackend::load(&cmd.model, config)
        .with_context(|| format!("failed to load Riper backend from {}", cmd.model.display()))?;

    let cache = DiphoneCache::open(&cmd.cache_dir).with_context(|| {
        format!("failed to open diphone cache at {}", cmd.cache_dir.display())
    })?;

    let inventory_text = std::fs::read_to_string(&cmd.inventory).with_context(|| {
        format!(
            "failed to read phone inventory from {}",
            cmd.inventory.display()
        )
    })?;

    let phones: Vec<&str> = inventory_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    let settings = ForgeSettings::default();
    let model_fp = fingerprint_path(backend.model_path());
    let config_fp = fingerprint_config(backend.config());
    let sample_rate = backend.config().sample_rate_hz;

    let mut forged_count = 0usize;
    let mut skip_count = 0usize;
    let mut error_count = 0usize;

    // Build all diphones: each phone with each other phone (N² pairs) plus silence boundaries
    let mut all_phones: Vec<String> = phones.iter().map(|s| s.to_string()).collect();
    if !all_phones.contains(&"_".to_string()) {
        all_phones.push("_".to_string());
    }

    for left in &all_phones {
        for right in &all_phones {
            if left == "_" && right == "_" {
                continue;
            }

            let key = CacheKey {
                model_fingerprint: model_fp.clone(),
                config_fingerprint: config_fp.clone(),
                speaker_id: String::new(),
                left: left.clone(),
                right: right.clone(),
                carrier_strategy_version: CARRIER_STRATEGY_VERSION.to_string(),
                forge_settings_version: FORGE_SETTINGS_VERSION.to_string(),
                sample_rate_hz: sample_rate,
                normalization_version: NORMALIZATION_VERSION.to_string(),
            };

            if cache.get(&key).is_some() {
                skip_count += 1;
                continue;
            }

            match forge_diphone(&mut backend, left, right, &settings) {
                Ok(forged) => {
                    let meta = CacheEntryMetadata {
                        key: key.clone(),
                        generated_at: forged
                            .unit
                            .metadata
                            .forge_provenance
                            .as_ref()
                            .map(|p| p.generated_at.clone())
                            .unwrap_or_default(),
                        carrier_sequence: forged.carrier_sequence.clone(),
                        halfseg_samples: forged.unit.halfseg_samples,
                        segmentation_confidence: forged.segmentation_confidence,
                        sample_count: forged.unit.samples.len(),
                        license_note: concat!(
                            "Generated locally from a Piper ONNX model. ",
                            "Do not redistribute without checking the model license."
                        )
                        .to_string(),
                    };
                    if let Err(err) = cache.store(&key, &forged.unit, meta) {
                        eprintln!("warning: failed to cache {left}-{right}: {err}");
                        error_count += 1;
                    } else {
                        forged_count += 1;
                    }
                }
                Err(err) => {
                    eprintln!("warning: failed to forge {left}-{right}: {err}");
                    error_count += 1;
                }
            }
        }
    }

    println!(
        "Diphone cache build complete: {} forged, {} skipped (cached), {} errors → {}",
        forged_count,
        skip_count,
        error_count,
        cmd.cache_dir.display()
    );
    Ok(())
}

#[cfg(not(feature = "tts-riper"))]
fn run_build(_cmd: DiphoneCacheBuildCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-riper` feature")
}
