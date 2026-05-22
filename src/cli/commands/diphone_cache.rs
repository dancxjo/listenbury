use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::{
    DiphoneAuditPlanCommand, DiphoneCacheBuildCommand, DiphoneCacheCommand,
    DiphoneCacheForgeCommand, DiphoneCacheListCommand, DiphoneCommand,
};

pub(crate) fn run_diphone(command: DiphoneCommand) -> Result<()> {
    match command {
        DiphoneCommand::Forge(cmd) => run_forge(cmd),
        DiphoneCommand::CacheBuild(cmd) => run_build(cmd),
        DiphoneCommand::CacheList(cmd) => run_list(cmd),
        DiphoneCommand::AuditPlan(cmd) => run_audit_plan(cmd),
    }
}

pub(crate) fn run_diphone_cache(command: DiphoneCacheCommand) -> Result<()> {
    match command {
        DiphoneCacheCommand::Forge(cmd) => run_diphone(DiphoneCommand::Forge(cmd)),
        DiphoneCacheCommand::Build(cmd) => run_diphone(DiphoneCommand::CacheBuild(cmd)),
        DiphoneCacheCommand::List(cmd) => run_diphone(DiphoneCommand::CacheList(cmd)),
        DiphoneCacheCommand::AuditPlan(cmd) => run_diphone(DiphoneCommand::AuditPlan(cmd)),
    }
}

#[cfg(feature = "tts-riper")]
fn run_forge(cmd: DiphoneCacheForgeCommand) -> Result<()> {
    use listenbury::voice::diphone::ForgeSettings;
    use listenbury::voice::diphone::cache::{CacheLookup, DiphoneCache};

    let mut backend = load_backend(&cmd.model, &cmd.config)?;
    let cache = DiphoneCache::open(&cmd.cache_dir).with_context(|| {
        format!(
            "failed to open diphone cache at {}",
            cmd.cache_dir.display()
        )
    })?;
    let settings = ForgeSettings::default();
    let key = cache_key_for_backend(&backend, &cmd.left, &cmd.right);

    let cache_state = cache.lookup_state(&key);
    match cache_state {
        CacheLookup::Hit => {
            let unit = cache.read(&key)?.ok_or_else(|| {
                anyhow::anyhow!("cache lookup returned hit but read returned none")
            })?;
            let confidence = unit
                .metadata
                .forge_provenance
                .as_ref()
                .map(|p| p.segmentation_confidence)
                .unwrap_or_default();
            let duration_ms = duration_ms(unit.samples.len(), unit.sample_rate_hz);
            println!(
                "Status: cache hit ({left}-{right})",
                left = cmd.left,
                right = cmd.right
            );
            println!(
                "Metadata: sample_rate={}Hz duration={:.2}ms halfseg={} confidence={:.2} carrier={}",
                unit.sample_rate_hz,
                duration_ms,
                unit.halfseg_samples,
                confidence,
                unit.metadata
                    .forge_provenance
                    .as_ref()
                    .map(|p| p.carrier_sequence.join(" "))
                    .unwrap_or_else(|| "(unknown)".to_string())
            );
        }
        CacheLookup::Corrupt { reason } => {
            eprintln!(
                "warning: cache entry was corrupt for {}-{}: {reason}",
                cmd.left, cmd.right
            );
            let (forged, carrier_samples) =
                forge_for_cli(&mut backend, &cmd.left, &cmd.right, &settings)?;
            let license = resolved_model_license(backend.config());
            let meta = cache_metadata(&key, &forged, &license);
            cache.store(&key, &forged.unit, meta)?;
            emit_forge_result(
                "regenerated after corrupt cache",
                &forged.unit,
                &forged.carrier_sequence,
                forged.segmentation_confidence,
            );
            maybe_write_debug_outputs(&cmd, &forged, &carrier_samples)?;
        }
        CacheLookup::Miss => {
            let (forged, carrier_samples) =
                forge_for_cli(&mut backend, &cmd.left, &cmd.right, &settings)?;
            let license = resolved_model_license(backend.config());
            let meta = cache_metadata(&key, &forged, &license);
            cache.store(&key, &forged.unit, meta)?;
            emit_forge_result(
                "newly generated",
                &forged.unit,
                &forged.carrier_sequence,
                forged.segmentation_confidence,
            );
            maybe_write_debug_outputs(&cmd, &forged, &carrier_samples)?;
        }
    }

    print_license_notice(backend.config());
    println!("Cache: {}", cmd.cache_dir.display());
    Ok(())
}

#[cfg(not(feature = "tts-riper"))]
fn run_forge(_cmd: DiphoneCacheForgeCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-riper` feature")
}

#[cfg(feature = "tts-riper")]
fn run_build(cmd: DiphoneCacheBuildCommand) -> Result<()> {
    use listenbury::voice::diphone::ForgeSettings;
    use listenbury::voice::diphone::cache::{CacheLookup, DiphoneCache};

    let mut backend = load_backend(&cmd.model, &cmd.config)?;
    let cache = DiphoneCache::open(&cmd.cache_dir).with_context(|| {
        format!(
            "failed to open diphone cache at {}",
            cmd.cache_dir.display()
        )
    })?;
    let phones = resolve_inventory(&cmd.inventory)?;
    let settings = ForgeSettings::default();
    let mut all_phones = phones;
    if !all_phones.iter().any(|p| p == "_") {
        all_phones.push("_".to_string());
    }

    let total_pairs = all_phones.len() * all_phones.len() - 1;
    let mut pair_index = 0usize;
    let mut generated = 0usize;
    let mut cached = 0usize;
    let mut failed = 0usize;
    let mut low_confidence = 0usize;
    let mut failures = Vec::new();

    for left in &all_phones {
        for right in &all_phones {
            if left == "_" && right == "_" {
                continue;
            }
            pair_index += 1;
            if pair_index % 50 == 0 || pair_index == 1 {
                println!("[{pair_index}/{total_pairs}] prewarming diphones…");
            }

            let key = cache_key_for_backend(&backend, left, right);
            if !cmd.force && matches!(cache.lookup_state(&key), CacheLookup::Hit) {
                cached += 1;
                continue;
            }

            match forge_for_cli(&mut backend, left, right, &settings) {
                Ok((forged, _carrier_samples)) => {
                    if forged.segmentation_confidence < 0.5 {
                        low_confidence += 1;
                    }
                    let meta =
                        cache_metadata(&key, &forged, &resolved_model_license(backend.config()));
                    if let Err(err) = cache.store(&key, &forged.unit, meta) {
                        failed += 1;
                        failures.push(format!("{left}-{right}: cache write failed: {err}"));
                    } else {
                        generated += 1;
                    }
                }
                Err(err) => {
                    failed += 1;
                    failures.push(format!("{left}-{right}: {err}"));
                }
            }
        }
    }

    println!("Diphone cache prewarm complete:");
    println!("  Generated      : {generated}");
    println!("  Cache hits     : {cached}");
    println!("  Failed         : {failed}");
    println!("  Low confidence : {low_confidence}");
    println!("  Cache dir      : {}", cmd.cache_dir.display());
    if !failures.is_empty() {
        println!("Failures:");
        for failure in failures {
            println!("  - {failure}");
        }
    }
    print_license_notice(backend.config());
    Ok(())
}

#[cfg(not(feature = "tts-riper"))]
fn run_build(_cmd: DiphoneCacheBuildCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-riper` feature")
}

#[cfg(feature = "tts-riper")]
fn run_list(cmd: DiphoneCacheListCommand) -> Result<()> {
    use listenbury::voice::diphone::cache::{CacheEntryMetadata, DiphoneCache};
    use listenbury::voice::diphone::forge::{fingerprint_config, fingerprint_path};

    let config = load_config(&cmd.config)?;
    let model_fingerprint = fingerprint_path(&cmd.model);
    let config_fingerprint = fingerprint_config(&config);
    let cache = DiphoneCache::open(&cmd.cache_dir)?;

    let mut entries = Vec::<CacheEntryMetadata>::new();
    for path in std::fs::read_dir(cache.dir())
        .with_context(|| format!("failed to read cache directory {}", cache.dir().display()))?
    {
        let path = path?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let bytes = std::fs::read(&path)
            .with_context(|| format!("failed to read cache metadata {}", path.display()))?;
        let meta: CacheEntryMetadata = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid cache metadata JSON {}", path.display()))?;
        if meta.key.model_fingerprint == model_fingerprint
            && meta.key.config_fingerprint == config_fingerprint
        {
            entries.push(meta);
        }
    }
    entries.sort_by(|a, b| {
        (&a.key.left, &a.key.right, &a.generated_at).cmp(&(
            &b.key.left,
            &b.key.right,
            &b.generated_at,
        ))
    });

    println!(
        "Cache entries for model={} config={} ({})",
        &model_fingerprint[..8.min(model_fingerprint.len())],
        &config_fingerprint[..8.min(config_fingerprint.len())],
        cache.dir().display()
    );
    for entry in &entries {
        let duration = duration_ms(entry.sample_count, entry.key.sample_rate_hz);
        println!(
            "{}-{} ts={} conf={:.2} duration={:.2}ms license={}",
            entry.key.left,
            entry.key.right,
            entry.generated_at,
            entry.segmentation_confidence,
            duration,
            entry.model_license
        );
        if cmd.verbose {
            println!(
                "  sample_rate={} halfseg={} carrier={} provenance={}",
                entry.key.sample_rate_hz,
                entry.halfseg_samples,
                entry.carrier_sequence.join(" "),
                entry.provenance_note
            );
        }
    }
    if entries.is_empty() {
        println!("(no cache entries for this model/config)");
    } else {
        println!("Total: {}", entries.len());
    }
    print_license_notice(&config);
    Ok(())
}

#[cfg(not(feature = "tts-riper"))]
fn run_list(_cmd: DiphoneCacheListCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-riper` feature")
}

#[cfg(feature = "tts-riper")]
fn run_audit_plan(cmd: DiphoneAuditPlanCommand) -> Result<()> {
    use listenbury::voice::diphone::cache::{CacheLookup, DiphoneCache};
    use listenbury::voice::mbrola::MbrolaDatabase;

    let config = load_config(&cmd.config)?;
    let cache = DiphoneCache::open(&cmd.cache_dir)?;
    let maybe_db = if let Some(voice) = &cmd.mbrola_voice {
        Some(MbrolaDatabase::load(voice)?)
    } else {
        None
    };
    let plan = load_phone_plan(&cmd.plan)?;
    let pairs = plan_pairs(&plan);
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut non_exact = Vec::new();
    let mut generatable_cache: HashMap<(String, String), bool> = HashMap::new();

    println!("=== Diphone Plan Audit ===");
    println!("Plan pairs : {}", pairs.len());
    println!("Plan path  : {}", cmd.plan.display());
    println!("Cache dir  : {}", cmd.cache_dir.display());
    if let Some(path) = &cmd.mbrola_voice {
        println!("MBROLA db  : {}", path.display());
    } else {
        println!("MBROLA db  : (none)");
    }

    for (left, right) in pairs {
        let class = if maybe_db
            .as_ref()
            .and_then(|db| db.diphone(&left, &right))
            .is_some()
        {
            "mbrola-exact"
        } else if matches!(
            cache.lookup_state(&cache_key_from_parts(&cmd.model, &config, &left, &right)),
            CacheLookup::Hit
        ) {
            "cache-hit"
        } else if *generatable_cache
            .entry((left.clone(), right.clone()))
            .or_insert_with(|| is_generatable_pair(&config, &left, &right))
        {
            "neural-forge"
        } else {
            "fallback"
        };
        *counts.entry(class).or_insert(0) += 1;
        if class != "mbrola-exact" && class != "cache-hit" {
            non_exact.push(format!("{left}-{right}: {class}"));
        }
    }

    println!("Coverage:");
    println!(
        "  MBROLA exact : {}",
        counts.get("mbrola-exact").copied().unwrap_or(0)
    );
    println!(
        "  Cache hit    : {}",
        counts.get("cache-hit").copied().unwrap_or(0)
    );
    println!(
        "  Neural forge : {}",
        counts.get("neural-forge").copied().unwrap_or(0)
    );
    println!(
        "  Fallback     : {}",
        counts.get("fallback").copied().unwrap_or(0)
    );
    if !non_exact.is_empty() {
        println!("Non-exact pairs:");
        for line in non_exact {
            println!("  - {line}");
        }
    }

    print_license_notice(&config);
    Ok(())
}

#[cfg(not(feature = "tts-riper"))]
fn run_audit_plan(_cmd: DiphoneAuditPlanCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-riper` feature")
}

#[cfg(feature = "tts-riper")]
fn load_backend(
    model: &Path,
    config_path: &Path,
) -> Result<listenbury::mouth::riper::RiperBackend> {
    use listenbury::mouth::riper::RiperBackend;
    let config = load_config(config_path)?;
    RiperBackend::load(model, config)
        .with_context(|| format!("failed to load Riper backend from {}", model.display()))
}

#[cfg(feature = "tts-riper")]
fn load_config(config_path: &Path) -> Result<listenbury::mouth::riper::PiperVoiceConfig> {
    use listenbury::mouth::riper::PiperVoiceConfig;
    let config_json = std::fs::read_to_string(config_path).with_context(|| {
        format!(
            "failed to read Piper voice config {}",
            config_path.display()
        )
    })?;
    PiperVoiceConfig::from_json_str(&config_json).with_context(|| {
        format!(
            "failed to parse Piper voice config {}",
            config_path.display()
        )
    })
}

#[cfg(feature = "tts-riper")]
fn cache_key_for_backend(
    backend: &listenbury::mouth::riper::RiperBackend,
    left: &str,
    right: &str,
) -> listenbury::voice::diphone::cache::CacheKey {
    use listenbury::voice::diphone::cache::CacheKey;
    use listenbury::voice::diphone::forge::{
        CARRIER_STRATEGY_VERSION, FORGE_SETTINGS_VERSION, NORMALIZATION_VERSION,
        fingerprint_config, fingerprint_path,
    };
    CacheKey {
        model_fingerprint: fingerprint_path(backend.model_path()),
        config_fingerprint: fingerprint_config(backend.config()),
        speaker_id: String::new(),
        left: left.to_string(),
        right: right.to_string(),
        carrier_strategy_version: CARRIER_STRATEGY_VERSION.to_string(),
        forge_settings_version: FORGE_SETTINGS_VERSION.to_string(),
        sample_rate_hz: backend.config().sample_rate_hz,
        normalization_version: NORMALIZATION_VERSION.to_string(),
    }
}

#[cfg(feature = "tts-riper")]
fn cache_key_from_parts(
    model: &Path,
    config: &listenbury::mouth::riper::PiperVoiceConfig,
    left: &str,
    right: &str,
) -> listenbury::voice::diphone::cache::CacheKey {
    use listenbury::voice::diphone::cache::CacheKey;
    use listenbury::voice::diphone::forge::{
        CARRIER_STRATEGY_VERSION, FORGE_SETTINGS_VERSION, NORMALIZATION_VERSION,
        fingerprint_config, fingerprint_path,
    };
    CacheKey {
        model_fingerprint: fingerprint_path(model),
        config_fingerprint: fingerprint_config(config),
        speaker_id: String::new(),
        left: left.to_string(),
        right: right.to_string(),
        carrier_strategy_version: CARRIER_STRATEGY_VERSION.to_string(),
        forge_settings_version: FORGE_SETTINGS_VERSION.to_string(),
        sample_rate_hz: config.sample_rate_hz,
        normalization_version: NORMALIZATION_VERSION.to_string(),
    }
}

#[cfg(feature = "tts-riper")]
fn forge_for_cli(
    backend: &mut listenbury::mouth::riper::RiperBackend,
    left: &str,
    right: &str,
    settings: &listenbury::voice::diphone::ForgeSettings,
) -> Result<(listenbury::voice::diphone::ForgedUnit, Vec<f32>)> {
    use listenbury::mouth::riper::{PiperPhoneme, PiperPhonemeSequence};
    use listenbury::voice::diphone::forge::{build_carrier_sequence, forge_from_samples};

    let carrier_sequence = build_carrier_sequence(left, right);
    let phonemes = carrier_sequence
        .iter()
        .map(|symbol| PiperPhoneme(symbol.clone()))
        .collect();
    let ids = PiperPhonemeSequence { phonemes }
        .to_piper_ids_compatible(backend.config())
        .with_context(|| format!("failed to map carrier sequence for {left}-{right}"))?;
    let carrier_pcm = backend.synthesize_ids(&ids)?;
    let forged = forge_from_samples(
        left,
        right,
        &carrier_pcm.samples,
        carrier_pcm.sample_rate_hz,
        &carrier_sequence,
        &listenbury::voice::diphone::forge::fingerprint_path(backend.model_path()),
        &listenbury::voice::diphone::forge::fingerprint_config(backend.config()),
        settings,
    )?;
    Ok((forged, carrier_pcm.samples))
}

#[cfg(feature = "tts-riper")]
fn cache_metadata(
    key: &listenbury::voice::diphone::cache::CacheKey,
    forged: &listenbury::voice::diphone::ForgedUnit,
    model_license: &str,
) -> listenbury::voice::diphone::cache::CacheEntryMetadata {
    use listenbury::voice::diphone::cache::CacheEntryMetadata;
    CacheEntryMetadata {
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
        extraction_start_sample: forged.segmentation.source_start_sample,
        extraction_end_sample: forged.segmentation.source_end_sample,
        model_license: model_license.to_string(),
        provenance_note: if model_license == "unknown" {
            concat!(
                "Generated locally from a Piper ONNX model; license metadata unknown. ",
                "Do not redistribute without checking upstream model terms."
            )
            .to_string()
        } else {
            format!(
                "Generated locally from a Piper ONNX model under `{model_license}`. Verify redistribution rights for derived audio before publishing."
            )
        },
    }
}

#[cfg(feature = "tts-riper")]
fn emit_forge_result(
    status: &str,
    unit: &listenbury::voice::mbrola::diphone_provider::DiphoneUnit,
    carrier_sequence: &[String],
    confidence: f32,
) {
    println!("Status: {status}");
    println!(
        "Metadata: sample_rate={}Hz duration={:.2}ms halfseg={} confidence={:.2} carrier={}",
        unit.sample_rate_hz,
        duration_ms(unit.samples.len(), unit.sample_rate_hz),
        unit.halfseg_samples,
        confidence,
        carrier_sequence.join(" ")
    );
}

#[cfg(feature = "tts-riper")]
fn maybe_write_debug_outputs(
    cmd: &DiphoneCacheForgeCommand,
    forged: &listenbury::voice::diphone::ForgedUnit,
    carrier_samples: &[f32],
) -> Result<()> {
    let Some(debug_dir) = &cmd.debug_dir else {
        return Ok(());
    };
    std::fs::create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug directory {}", debug_dir.display()))?;
    let stem = format!(
        "{}-{}",
        sanitize_phone_symbol(&cmd.left),
        sanitize_phone_symbol(&cmd.right)
    );
    let carrier_path = debug_dir.join(format!("{stem}-carrier.wav"));
    let extracted_path = debug_dir.join(format!("{stem}-extracted.wav"));
    let normalized_path = debug_dir.join(format!("{stem}-normalized.wav"));
    let json_path = debug_dir.join(format!("{stem}.json"));

    write_debug_wav(&carrier_path, carrier_samples, forged.unit.sample_rate_hz)?;
    let pre_normalized = carrier_samples
        .get(forged.segmentation.source_start_sample..forged.segmentation.source_end_sample)
        .map(|slice| slice.to_vec())
        .unwrap_or_else(|| {
            eprintln!(
                "warning: invalid segmentation range {}..{} for debug extraction (carrier samples={}); using normalized samples",
                forged.segmentation.source_start_sample,
                forged.segmentation.source_end_sample,
                carrier_samples.len()
            );
            forged.unit.samples.clone()
        });
    write_debug_wav(&extracted_path, &pre_normalized, forged.unit.sample_rate_hz)?;
    write_debug_wav(
        &normalized_path,
        &forged.unit.samples,
        forged.unit.sample_rate_hz,
    )?;

    let debug_json = serde_json::json!({
        "left": cmd.left,
        "right": cmd.right,
        "carrierSequence": forged.carrier_sequence,
        "segmentation": {
            "confidence": forged.segmentation.confidence,
            "warnings": forged.segmentation.warnings,
            "sourceStartSample": forged.segmentation.source_start_sample,
            "sourceEndSample": forged.segmentation.source_end_sample,
            "halfsegSamples": forged.segmentation.halfseg_samples
        },
        "normalization": {
            "dcOffsetRemoved": forged.normalization.dc_offset_removed,
            "fadeSamplesApplied": forged.normalization.fade_samples_applied,
            "rmsBefore": forged.normalization.rms_before,
            "rmsAfter": forged.normalization.rms_after
        }
    });
    std::fs::write(&json_path, serde_json::to_vec_pretty(&debug_json)?)
        .with_context(|| format!("failed to write {}", json_path.display()))?;
    println!("Debug outputs: {}", debug_dir.display());
    Ok(())
}

#[cfg(feature = "tts-riper")]
fn write_debug_wav(path: &Path, samples: &[f32], sample_rate_hz: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create {}", path.display()))?;
    for sample in samples {
        let scaled = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(scaled)?;
    }
    writer.finalize()?;
    Ok(())
}

fn sanitize_phone_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "tts-riper")]
fn print_license_notice(config: &listenbury::mouth::riper::PiperVoiceConfig) {
    let license = resolved_model_license(config);
    if license == "unknown" {
        println!("License: unknown (model/config metadata does not declare license)");
        println!("         generated outputs may be non-redistributable; verify upstream terms.");
    } else {
        println!("License: {license}");
    }
}

#[cfg(feature = "tts-riper")]
fn resolved_model_license(config: &listenbury::mouth::riper::PiperVoiceConfig) -> String {
    for (key, value) in &config.model_metadata {
        let lower = key.to_ascii_lowercase();
        if lower.contains("license") && !value.trim().is_empty() {
            return value.trim().to_string();
        }
    }
    "unknown".to_string()
}

fn duration_ms(sample_count: usize, sample_rate_hz: u32) -> f32 {
    if sample_rate_hz == 0 {
        eprintln!("warning: sample_rate_hz=0; duration calculation returns 0.0");
        return 0.0;
    }
    (sample_count as f32 * 1000.0) / sample_rate_hz as f32
}

fn resolve_inventory(inventory_arg: &str) -> Result<Vec<String>> {
    let inventory_path = PathBuf::from(inventory_arg);
    if inventory_path.is_file() {
        let text = std::fs::read_to_string(&inventory_path).with_context(|| {
            format!(
                "failed to read phone inventory from {}",
                inventory_path.display()
            )
        })?;
        return Ok(parse_inventory_lines(&text));
    }
    if let Some(inventory) = builtin_inventory(inventory_arg) {
        return Ok(inventory.iter().map(|value| value.to_string()).collect());
    }
    anyhow::bail!(
        "unknown inventory `{inventory_arg}`; pass a file path or one of: {}",
        builtin_inventory_names().join(", ")
    )
}

fn parse_inventory_lines(text: &str) -> Vec<String> {
    let mut uniq = BTreeSet::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        uniq.insert(line.to_string());
    }
    uniq.into_iter().collect()
}

fn builtin_inventory(name: &str) -> Option<&'static [&'static str]> {
    match name {
        // IPA/espeak-ish symbols chosen as a compact US-English baseline for cache prewarm.
        "en-us-basic" => Some(&[
            "_", "a", "e", "i", "o", "u", "@", "p", "b", "t", "d", "k", "g", "f", "v", "s", "z",
            "h", "m", "n", "l", "r", "w", "j",
        ]),
        _ => None,
    }
}

fn builtin_inventory_names() -> Vec<&'static str> {
    vec!["en-us-basic"]
}

#[cfg(feature = "tts-riper")]
fn load_phone_plan(path: &Path) -> Result<listenbury::voice::mbrola::PhoneTimedPlan> {
    use listenbury::voice::mbrola::pho::parse_pho;
    use listenbury::voice::mbrola::{PhoneTimedPlan, read_pho_file};
    if path.extension().and_then(|ext| ext.to_str()) == Some("pho") {
        return read_pho_file(path);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read phone plan {}", path.display()))?;
    if let Ok(plan) = serde_json::from_str::<PhoneTimedPlan>(&text) {
        return Ok(plan);
    }
    parse_pho(&text).with_context(|| format!("failed to parse phone plan {}", path.display()))
}

#[cfg(feature = "tts-riper")]
fn plan_pairs(plan: &listenbury::voice::mbrola::PhoneTimedPlan) -> Vec<(String, String)> {
    let mut symbols = Vec::with_capacity(plan.phones.len() + 2);
    symbols.push("_".to_string());
    symbols.extend(plan.phones.iter().map(|phone| phone.symbol.clone()));
    symbols.push("_".to_string());
    symbols
        .windows(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect()
}

#[cfg(feature = "tts-riper")]
fn is_generatable_pair(
    config: &listenbury::mouth::riper::PiperVoiceConfig,
    left: &str,
    right: &str,
) -> bool {
    use listenbury::mouth::riper::{PiperPhoneme, PiperPhonemeSequence};
    use listenbury::voice::diphone::build_carrier_sequence;
    let phonemes = build_carrier_sequence(left, right)
        .into_iter()
        .map(PiperPhoneme)
        .collect();
    PiperPhonemeSequence { phonemes }
        .to_piper_ids_compatible(config)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inventory_deduplicates_and_ignores_comments() {
        let parsed = parse_inventory_lines(
            r#"
            # comment
            h
            @
            h
            "#,
        );
        assert_eq!(parsed, vec!["@", "h"]);
    }

    #[test]
    fn resolve_builtin_inventory_name() {
        let parsed = resolve_inventory("en-us-basic").expect("builtin inventory should resolve");
        assert!(parsed.contains(&"h".to_string()));
        assert!(parsed.contains(&"@".to_string()));
    }

    #[test]
    fn duration_ms_handles_zero_sample_rate() {
        assert_eq!(duration_ms(100, 0), 0.0);
    }

    #[test]
    fn sanitize_phone_symbol_replaces_path_unsafe_chars() {
        assert_eq!(sanitize_phone_symbol("h/@"), "h__");
    }
}
