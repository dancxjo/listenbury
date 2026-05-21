# eSpeak-ng → Riper rule import audit (seed pass)

## Scope

This seed pass imports **typed, inspectable** eSpeak-ng-inspired rule data for Riper, with emphasis on:

- weak-form handling (`to` reduction)
- strong/contrastive overrides for `to`
- punctuation prosody defaults (`!`, `?`)
- phoneme/voice metadata placeholders for future expansion

The importer does **not** embed eSpeak-ng runtime behavior.

## Audited eSpeak-ng source families

| eSpeak-ng source area | Why it matters to Riper | Seed-pass action |
|---|---|---|
| `dictsource/en_rules` | pronunciation/word-level rewrite rules, weak forms, citation behavior | translated into typed `WeakFormRule` and `PronunciationOverrideRule` seeds |
| `phsource/en_us` (and related voice files) | language/voice defaults and variety behavior | translated to `VoiceVariantRule` seed defaults |
| `phsource/phonemes` | phoneme symbol mapping guidance | translated to `PhonemeMappingRule` seed entries |
| punctuation/prosody behavior from language rules | phrase-final/terminal contour cues | translated to `PunctuationProsodyRule` seeds |

## Licensing and attribution notes

- Upstream eSpeak-ng materials are treated as **reference inputs** and documented as `GPL-3.0-or-later`.
- This pass stores **hand-translated seed rules** (not raw copied runtime internals) in a typed JSON table.
- Every imported seed rule includes provenance fields:
  - `source`
  - `source_file`
  - `source_license`
  - `imported_at`

If future work imports broader direct data slices, re-check compatibility and preserve attribution metadata per rule.

## Seed converter details

- Seed data file: `src/mouth/riper/data/espeak_ng_seed_rules.json`
- Typed loader/converter: `src/mouth/riper/espeak_ng_rules.rs`
  - `import_rule_table_from_str`
  - `export_rule_table_to_json`
  - `load_seed_rule_table`

Output is deterministic JSON and test-covered for round-trip stability.
