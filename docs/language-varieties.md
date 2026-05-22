# Language-variety datapacks

Listenbury now carries an explicit `en-US` language-variety datapack at:

- `data/language-varieties/en-US/manifest.toml`
- `data/language-varieties/en-US/phonology.toml`
- `data/language-varieties/en-US/backend-maps/mbrola-us1.toml`
- `data/language-varieties/en-US/backend-maps/mbrola-us3.toml`

The runtime loader is in:

- `src/linguistic/language_variety.rs`

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
   - `phonology.toml`
   - `backend-maps/*.toml` for backend-specific symbol maps
2. Add a loader entry in `src/linguistic/language_variety.rs`.
3. Wire call sites to use the selected variety lookup rather than hard-coded symbol tables.
4. Add tests covering:
   - datapack load validation
   - vowel/nucleus classification
   - backend mapping parity and unknown-symbol errors
