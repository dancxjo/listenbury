//! `dev mbrola-inventory` — inspect a MBROLA voice database.
//!
//! Reports:
//! - Voice metadata (sample rate, period, version, database path)
//! - License manifest status
//! - Full phoneme inventory
//! - Diphone count and halfseg statistics
//! - Suspicious (short or empty) units
//! - If `--plan <file.pho>` is provided: which diphones are missing for that
//!   plan and what fallback strategy would be used

use std::collections::BTreeMap;

use anyhow::Result;

use crate::cli::{MbrolaAuditCommand, MbrolaInventoryCommand};

pub(crate) fn run_mbrola_inventory(cmd: MbrolaInventoryCommand) -> Result<()> {
    let voice_path = &cmd.voice;
    if !voice_path.is_file() {
        anyhow::bail!(
            "MBROLA voice database not found at {}",
            voice_path.display()
        );
    }

    let db = listenbury::voice::mbrola::MbrolaDatabase::load(voice_path)?;

    println!("=== MBROLA Voice Inventory ===");
    println!("Path         : {}", voice_path.display());
    println!("Version      : {}", db.version);
    println!("Sample rate  : {} Hz", db.sample_rate_hz);
    println!("Period       : {} samples", db.mbr_period);
    println!("Coding       : {}", db.coding);
    println!("Raw PCM size : {} bytes", db.size_raw_bytes);

    // Manifest status
    match listenbury::voice::mbrola::VoiceManifest::find_for_voice(voice_path) {
        Ok(Some(manifest)) => {
            println!();
            println!("=== License Manifest ===");
            println!(
                "Name         : {}",
                manifest.name.as_deref().unwrap_or("(not set)")
            );
            println!(
                "License      : {}",
                manifest.license_name.as_deref().unwrap_or("(not specified)")
            );
            println!(
                "License URL  : {}",
                manifest.license_url.as_deref().unwrap_or("(not specified)")
            );
            println!(
                "Upstream URL : {}",
                manifest.upstream_url.as_deref().unwrap_or("(not specified)")
            );
            println!(
                "Redistrib.   : {}",
                fmt_tribool(manifest.redistribution_allowed)
            );
            println!(
                "Commercial   : {}",
                fmt_tribool(manifest.commercial_allowed)
            );
            println!(
                "Attribution  : {}",
                fmt_tribool(manifest.attribution_required)
            );
            if let Some(notes) = &manifest.notes {
                println!("Notes        : {notes}");
            }
            let warnings = manifest.license_warnings();
            if warnings.is_empty() {
                println!("License OK   : no warnings");
            } else {
                for w in &warnings {
                    println!("⚠  {w}");
                }
            }
        }
        Ok(None) => {
            println!();
            println!("=== License Manifest ===");
            println!("⚠  No manifest.toml or manifest.json found next to the voice database.");
            println!("   Add a manifest to record license and provenance information.");
        }
        Err(e) => {
            println!();
            println!("⚠  Failed to load manifest: {e}");
        }
    }

    // Phoneme inventory
    let phonemes: Vec<&str> = db.phonemes().collect();
    println!();
    println!("=== Phoneme Inventory ({} symbols) ===", phonemes.len());
    let cols = 12;
    for chunk in phonemes.chunks(cols) {
        println!("  {}", chunk.join("  "));
    }

    // Diphone statistics
    let mut halfseg_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut short_units = Vec::new();
    let mut total_diphones = 0usize;

    for left in db.phonemes() {
        for right in db.phonemes() {
            if let Some(diphone) = db.diphone(left, right) {
                total_diphones += 1;
                let physical = db.physical_frames(diphone);
                let total_samples = physical * db.mbr_period;
                let bucket = diphone.halfseg_samples / 10 * 10; // round to 10
                *halfseg_hist.entry(bucket).or_insert(0) += 1;
                if total_samples < db.mbr_period * 2 {
                    short_units.push(format!(
                        "{}-{} (halfseg={}, frames={})",
                        left, right, diphone.halfseg_samples, physical
                    ));
                }
            }
        }
    }

    println!();
    println!("=== Diphone Statistics ===");
    println!("Total diphones: {total_diphones}");
    println!("Halfseg sample distribution (bucket size=10):");
    for (bucket, count) in &halfseg_hist {
        println!("  {:>5}-{:<5}: {count}", bucket, bucket + 9);
    }

    if short_units.is_empty() {
        println!("No suspiciously short or empty diphone units.");
    } else {
        println!("⚠  Short/empty units ({}):", short_units.len());
        for u in &short_units {
            println!("  {u}");
        }
    }

    Ok(())
}

pub(crate) fn run_mbrola_audit(cmd: MbrolaAuditCommand) -> Result<()> {
    let voice_path = &cmd.voice;
    if !voice_path.is_file() {
        anyhow::bail!(
            "MBROLA voice database not found at {}",
            voice_path.display()
        );
    }

    let db = listenbury::voice::mbrola::MbrolaDatabase::load(voice_path)?;

    let plan_path = &cmd.plan;
    if !plan_path.is_file() {
        anyhow::bail!("MBROLA .pho plan not found at {}", plan_path.display());
    }

    let plan = listenbury::voice::mbrola::read_pho_file(plan_path)?;

    println!("=== MBROLA Audit ===");
    println!("Voice : {}", voice_path.display());
    println!("Plan  : {}", plan_path.display());
    println!("Phones: {}", plan.phones.len());

    let mut exact_count = 0usize;
    let mut fallback_count = 0usize;
    let mut silence_count = 0usize;
    let mut missing_diphones = Vec::new();

    // For each pair in the plan, check what diphone fallback would be used
    let phones = &plan.phones;
    for i in 0..phones.len() {
        let phone = &phones[i];
        if phone.symbol == "_" {
            continue;
        }
        let prev = if i > 0 { phones[i - 1].symbol.as_str() } else { "_" };
        let next = phones.get(i + 1).map(|p| p.symbol.as_str()).unwrap_or("_");

        // Check right context: (prev, phone)
        audit_diphone_pair(&db, prev, &phone.symbol, &mut exact_count, &mut fallback_count, &mut silence_count, &mut missing_diphones);
        // Check left context: (phone, next)
        audit_diphone_pair(&db, &phone.symbol, next, &mut exact_count, &mut fallback_count, &mut silence_count, &mut missing_diphones);
    }

    println!();
    println!("=== Diphone Coverage ===");
    let total_pairs = exact_count + fallback_count + silence_count;
    println!("Total diphone lookups : {total_pairs}");
    println!("  Exact              : {exact_count}");
    println!("  Boundary fallback  : {fallback_count}");
    println!("  Synthetic silence  : {silence_count}");

    if missing_diphones.is_empty() {
        println!("All diphones covered exactly.");
    } else {
        println!();
        println!("⚠  Non-exact diphone lookups ({}):", missing_diphones.len());
        for entry in &missing_diphones {
            println!("  {entry}");
        }
    }

    Ok(())
}

fn audit_diphone_pair(
    db: &listenbury::voice::mbrola::MbrolaDatabase,
    left: &str,
    right: &str,
    exact_count: &mut usize,
    fallback_count: &mut usize,
    silence_count: &mut usize,
    missing_diphones: &mut Vec<String>,
) {
    if db.diphone(left, right).is_some() {
        *exact_count += 1;
        return;
    }

    // Try boundary half: (_, right) for right-half context
    if left != "_" && right != "_" {
        if db.diphone("_", right).is_some() {
            *fallback_count += 1;
            missing_diphones.push(format!(
                "{left}-{right}: boundary fallback (_-{right})"
            ));
            return;
        }
        // Try (left, _) for left-half context
        if db.diphone(left, "_").is_some() {
            *fallback_count += 1;
            missing_diphones.push(format!(
                "{left}-{right}: boundary fallback ({left}-_)"
            ));
            return;
        }
    }

    *silence_count += 1;
    missing_diphones.push(format!(
        "{left}-{right}: synthetic silence (no fallback available)"
    ));
}

fn fmt_tribool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "unknown",
    }
}
