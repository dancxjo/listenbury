# Language-variety datapacks

Listenbury now carries an explicit `en-US` language-variety datapack at:

- `data/language-varieties/en-US/manifest.toml`
- `data/language-varieties/en-US/phonology.toml`
- `data/language-varieties/en-US/phoneme-inventory.toml`
- `data/language-varieties/en-US/vowel-targets.toml`
- `data/language-varieties/en-US/acoustic-tweaks.toml`
- `data/language-varieties/en-US/spelling-rules.toml`
- `data/language-varieties/en-US/pronunciation-rules.toml`
- `data/language-varieties/en-US/pronunciation-rules.json`
- `data/language-varieties/en-US/morphophonology.toml`
- `data/language-varieties/en-US/prosody-rules.toml`
- `data/language-varieties/en-US/punctuation-rules.toml`
- `data/language-varieties/en-US/backend-maps/riper.toml`
- `data/language-varieties/en-US/backend-maps/mbrola-us1.toml`
- `data/language-varieties/en-US/backend-maps/mbrola-us3.toml`

The runtime loader is in:

- `src/linguistic/language_variety.rs`
- `src/linguistic/language_pack.rs`

## Current English-specific paths inventoried

- `src/speech/prosody_timing.rs` (`is_vowel_phone`)
- `src/mouth/riper/g2p.rs` (`is_nucleus_symbol`)
- `src/mouth/riper/phoneme.rs` (`is_arpabet_vowel` stress stripping)
- `src/voice/mbrola/symbols.rs` (`us1_starter`, `us3_starter`)
- `src/linguistic/inventory.rs` (`english_phoneme_table`)
- `src/linguistic/rule_registry.rs` (English inventory + phonotactic fragments)

## Adding or overriding a variety

1. Create a new `data/language-varieties/<id>/` folder with:
   - `manifest.toml`
   - `phonology.toml` (compatibility classifier layer)
   - `phoneme-inventory.toml`
   - `vowel-targets.toml`
   - `acoustic-tweaks.toml`
   - `spelling-rules.toml`
   - `pronunciation-rules.toml`
   - `morphophonology.toml`
   - `prosody-rules.toml`
   - `punctuation-rules.toml`
   - `backend-maps/*.toml` for backend-specific symbol maps
2. Add a loader entry in `src/linguistic/language_pack.rs` (and `language_variety.rs` only for compatibility lookups).
3. Wire call sites to use the selected variety lookup rather than hard-coded symbol tables.
4. Add tests covering:
   - datapack load validation
   - vowel/nucleus classification
   - backend mapping parity and unknown-symbol errors
