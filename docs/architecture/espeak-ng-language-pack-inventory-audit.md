# eSpeak-ng → language-pack inventory audit (rule-catalog pass)

## Scope

This rule-catalog pass inventories eSpeak-ng-inspired linguistic behavior into Listenbury top-level language-pack data, with emphasis on:

- weak-form handling (`to` reduction)
- strong/contrastive overrides for `to`
- punctuation prosody defaults (`!`, `?`)
- phoneme/voice metadata placeholders for future expansion

The importer does **not** embed eSpeak-ng runtime behavior and does not define Listenbury's internal ontology around eSpeak categories.

## Audited eSpeak-ng source families

| eSpeak-ng source area | Listenbury language-pack interpretation | Rule-catalog pass action |
|---|---|---|
| `dictsource/en_rules` | pronunciation + lexical rewrite behavior | restated as native pronunciation/weak-form rule entries with provenance metadata |
| `phsource/en_us` (and related voice files) | language/voice defaults and variety behavior | restated as variety-level defaults and backend-map guidance |
| `phsource/phonemes` | phoneme inventory and backend mapping hints | restated into language-pack inventory + backend-map starter entries |
| punctuation/prosody behavior from language rules | phrase-boundary contour cues | restated into punctuation/prosody rule entries |

## Licensing and attribution notes

- Upstream eSpeak-ng materials are treated as **reference inputs** and documented as `GPL-3.0-or-later`.
- This pass stores **hand-translated rule-catalog entries** (not an eSpeak runtime compatibility layer).
- Language-pack rule-catalog entries carry source provenance metadata (`source`, `source_file`, `source_license`, `imported_at`).

## Rule-catalog converter details

- Rule-catalog data file: `data/language-varieties/en-US/pronunciation-rules.json`
- Typed loader/converter: `src/linguistic/language_pack_rules.rs`
  - `import_rule_catalog_from_str`
  - `export_rule_catalog_to_json`
  - `load_pronunciation_rule_catalog`

Output is deterministic JSON and test-covered for round-trip stability.
