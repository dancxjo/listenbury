# Language pack source inventory

This document inventories useful data families in eSpeak-ng and related speech-synthesis sources for Listenbury language packs.

It is deliberately **not** a plan to copy eSpeak-ng's rule model or build an eSpeak compatibility layer. eSpeak-ng is source material: a field notebook of useful linguistic and acoustic observations. Listenbury should restate the underlying behaviors in its own language-pack concepts and code.

## Core principle

```text
eSpeak-ng / related source material
  -> inventory the behavior
  -> name the behavior in Listenbury terms
  -> place it in the appropriate language-pack section
  -> preserve provenance as metadata
  -> implement with Listenbury-native types and APIs
```

Not:

```text
eSpeak-ng rule syntax
  -> mirrored internally
  -> renderer-specific compatibility layer
```

Riper, Klatt, MBROLA, Piper, singing, and future ONNX renderers should consume the same top-level language-pack data.

## Upstream/source families to inventory

### 1. Language identity and voice/language metadata

Primary upstream locations:

- `espeak-ng-data/lang/**`
- `espeak-ng-data/voices/**`
- `docs/voices.md`
- `docs/add_language.md`

Observed data families:

- BCP-47-ish language/accent tags
- language family grouping
- display name
- preferred/default voice selection
- language fallback and dialect preference
- selected phoneme table
- selected dictionary
- dictionary rule overrides
- stress defaults
- intonation defaults
- number/symbol behavior hooks
- voice variant vs real language distinction

Listenbury-native destination:

- `LanguagePackManifest`
- `LanguageVarietyId`
- `LanguageFamily`
- `VarietyImplementationStatus`
- `LanguagePackInheritance`
- `VoicePolicy` only for renderer/voice-quality facts

Do not let voice files become phonology. Voice settings can select or condition pack behavior only through explicit metadata.

### 2. Phoneme inventory and feature metadata

Primary upstream locations:

- `phsource/phonemes`
- `phsource/ph_*`
- `docs/phontab.md`
- `docs/phonemes.md`
- `docs/phoneme_model.md`

Observed data families:

- phoneme tables
- inherited phoneme tables
- imported/redefined phonemes
- mnemonic phoneme symbols
- utility symbols for stress, pauses, word boundary, syllable behavior
- phoneme types and properties
- manner/place/voicing-like properties
- rhotic/unstressed/etc. flags
- length and length modifiers
- start/end type groups
- voicing-switch relationships
- conditional phoneme changes

Listenbury-native destination:

- `PhonemicInventory`
- `PhonemeDefinition`
- `FeatureBundle`
- `PhonemeClass`
- `PhoneString`
- `Stress`
- `PauseKind`
- `PhonemeInventoryInheritance`
- `AllophoneRule`
- `PhoneEnvironmentRule`

Important: inventory and feature facts are language-pack data. Renderer symbol maps are separate.

### 3. Vowel targets and acoustic production hints

Primary upstream locations:

- `phsource/phonemes`
- `phsource/ph_*`
- `phsource/voc*`, `phsource/vdiph*`, `phsource/vnasal`, etc.
- `docs/phontab.md`
- `src/libespeak-ng/` synthesis tables/code, only as reference behavior

Observed data families:

- vowel quality targets
- diphthong targets
- formant-transition instructions
- vowel start/end/in/out behavior
- length/default duration
- RMS/amplitude hints
- nasal pole/zero-like hints
- consonant burst/frication/aspiration hints
- transition/coarticulation hints
- conditional acoustic changes by neighboring phones, stress, pause, etc.

Listenbury-native destination:

- `VowelTargetTable`
- `PhoneAcousticTarget`
- `AcousticTweak`
- `CoarticulationProfile`
- `DurationHint`
- `SourceFilterHint`
- `KlattAdapter`
- `DiphoneAdapter`
- `MbrolaAdapter`

Important: acoustic targets should be pack-level and renderer-consumable. Klatt should not own the only English vowel table.

### 4. Spelling/G2P rule observations

Primary upstream locations:

- `dictsource/*_rules`
- `docs/dictionary.md`

Observed data families:

- grapheme-to-phoneme behavior
- left/right context
- letter groups
- vowel/consonant/digit/word-boundary predicates
- rule ranking/scoring ideas
- word-start/word-end behavior
- syllable-count-like conditions
- doubled-letter contexts
- foreign-language retranslation/island behavior
- character substitution

Listenbury-native destination:

- `SpellingRule`
- `GraphemePattern`
- `OrthographicContextPredicate`
- `RulePriority`
- `LanguageIslandRule`
- `CharacterSubstitutionRule`
- `SpellingRuleTrace`

Important: do not mirror eSpeak syntax. Restate each behavior using Listenbury pattern/matcher concepts.

### 5. Pronunciation dictionary and lexical facts

Primary upstream locations:

- `dictsource/*_list`
- `dictsource/*_extra`
- `dictsource/*_listx`
- `docs/dictionary.md`

Observed data families:

- explicit word pronunciations
- stress-only overrides
- exceptional pronunciations
- common function-word weak/unstressed facts
- pause/break facts before words or phrases
- abbreviation/acronym facts
- capitalization-sensitive facts
- dot/has-dot abbreviation facts
- part-of-speech/context disambiguation hints
- text replacement facts
- multi-pronunciation alternatives

Listenbury-native destination:

- `PronunciationLexicon`
- `PronunciationEntry`
- `StressOverride`
- `LexicalProsodyFact`
- `OrthographicEmphasisKind`
- `AbbreviationPolicy`
- `ContextualPronunciationRule`
- `WordSyntaxHint`
- `PronunciationProvenance`

Important: dictionary flags become meaningful facts, not imported flag tokens.

### 6. Multi-word entries and phrase-level facts

Primary upstream locations:

- `dictsource/*_list`
- `docs/dictionary.md`

Observed data families:

- explicit pronunciation for word groups
- phrase-level stress/prosody overrides
- run-together phrases
- phrase flags such as pause/break behavior
- hyphenated multi-part entries
- word-boundary markers inside phoneme strings

Listenbury-native destination:

- `PhrasePronunciationRule`
- `PhraseClump`
- `NoBreakPolicy`
- `PhraseProsodyOverride`
- `PhraseLexicalFact`

Important: phrase clumps are first-class. Do not flatten them into one fake word unless the pack explicitly says to.

### 7. Morphophonology and retranslation behavior

Primary upstream locations:

- `dictsource/*_rules`
- `dictsource/*_list`
- `docs/dictionary.md`

Observed data families:

- standard suffix handling
- standard prefix handling
- remove suffix/prefix and retranslate stem
- query dictionary for stem stress or attributes
- determine stress before adding affix
- doubled consonant repair
- dropped `e` repair
- `y -> i` repair
- suffix implies verb form
- suffix implies next word likely verb
- multiple suffix stripping

Listenbury-native destination:

- `MorphophonologyRule`
- `MorphemeKind`
- `StemPolicy`
- `SpellingRepairHint`
- `StressInheritancePolicy`
- `AffixPronunciation`
- `DerivedWordStructure`

Important: this is word anatomy. It belongs in language packs, not renderer code.

### 8. Numbers, ordinals, letters, symbols, punctuation names, emoji

Primary upstream locations:

- `dictsource/*_list`
- `dictsource/*_rules`
- `dictsource/*_emoji`
- `docs/numbers.md`
- `docs/dictionary.md`

Observed data families:

- cardinal number names
- ordinal names
- digit/tens/hundreds/thousands/millions behavior
- year-like numbers
- decimal behavior
- letter names
- acronym/initialism behavior
- symbol names
- punctuation names
- emoji names/descriptions
- character substitution

Listenbury-native destination:

- `NumberNormalizationRules`
- `OrdinalRules`
- `LetterNameTable`
- `SymbolNameTable`
- `EmojiNameTable`
- `InitialismPolicy`
- `NormalizedToken` with provenance

Important: numbers are tiny scripts. Normalize into structure before pronunciation and prosody touch them.

### 9. Prosody, rhythm, stress, and intonation

Primary upstream locations:

- `docs/intonation.md`
- `docs/voices.md`
- `dictsource/*_list`
- `dictsource/*_rules`
- `phsource/phonemes`
- `src/libespeak-ng/` intonation/rhythm code, only as reference behavior

Observed data families:

- default stress rule
- spelling stress behavior
- stress length adjustments
- stress amplitude adjustments
- stress addition behavior
- vowel length under stress/unstress
- clause-final stress behavior
- phrase-final/question/exclamation contours
- intonation tunes
- pause length categories
- rhythm tweaks by language

Listenbury-native destination:

- `StressRule`
- `ProsodyRule`
- `BoundaryProsodyRule`
- `RhythmProfile`
- `IntonationContour`
- `PhraseBoundaryPolicy`
- `DurationModel`

Important: punctuation and dictionary hints are prosody evidence, not just text-cleanup hacks.

### 10. Renderer/backend maps and adapters

Primary upstream locations:

- `phsource/mbrola`
- `docs/mbrola.md`
- `espeak-ng-data/voices/mb/mb-*`
- existing Listenbury `data/language-varieties/*/backend-maps/`

Observed data families:

- backend-specific phoneme symbol maps
- MBROLA voice mappings
- phoneme duration/pitch output expectations
- front-end PHO-style behavior
- renderer-specific limitations

Listenbury-native destination:

- `backend-maps/<backend>.toml`
- `RendererSymbolMap`
- `RendererCapabilityProfile`
- `MbrolaAdapter`
- `RiperAdapter`
- `KlattAdapter`
- `PiperAdapter`

Important: backend maps are allowed to be renderer-specific. Phonology and pronunciation rules are not.

### 11. Voice-quality and source/filter policy

Primary upstream locations:

- `espeak-ng-data/voices/**`
- `docs/voices.md`
- `phsource/klatt`
- `src/libespeak-ng/` synthesis references

Observed data families:

- pitch range
- formant shifts
- frequency additions
- echo
- tone
- flutter
- roughness
- voicing
- consonant strength
- breath/breath width
- speed
- word-gap behavior

Listenbury-native destination:

- `VoicePolicy`
- `SourcePolicy`
- `FormantPolicy`
- `NoisePolicy`
- `TimingPolicy`
- `RendererCapabilityProfile`

Important: voice policy is rendering/acoustic personality, not phoneme inventory.

### 12. Tests and regression fixtures

Primary upstream locations:

- `tests/**`
- `docs/add_language.md`

Observed data families:

- expected phoneme strings
- expected audio hashes
- language-specific regression examples
- pronunciation debug traces via `-X` / compiled debug dictionaries

Listenbury-native destination:

- `LanguagePackFixture`
- `PronunciationTraceFixture`
- `GoldenPhonemeText`
- `GoldenPhoneString`
- `RendererGoldenFixture`

Important: use fixtures to preserve observed behavior goals, not upstream rule syntax.

## First-pass Listenbury pack sections

A useful initial pack shape:

```text
data/language-varieties/en-US/
  manifest.toml
  inventory.toml
  phonology.toml
  vowel-targets.toml
  acoustic-tweaks.toml
  spelling-rules.toml
  lexicon.toml
  morphophonology.toml
  phrase-rules.toml
  prosody-rules.toml
  normalization/
    numbers.toml
    letters.toml
    symbols.toml
    emoji.toml
  backend-maps/
    riper.toml
    klatt.toml
    mbrola-us1.toml
    mbrola-us3.toml
    piper.toml
```

## Immediate inventory tasks

1. Catalog current eSpeak-ng source families by path and data type.
2. Catalog current Listenbury language/phonology/acoustic data by path and data type.
3. Define the minimal `LanguagePack` struct and pack section types.
4. Move Riper-local eSpeak seed data out of `src/mouth/riper` or deprecate it behind the new pack API.
5. Rewrite the Riper-specific eSpeak audit doc into a language-pack source-inventory doc.
6. Pick one tiny behavior, such as weak `to` or abbreviation-dot handling, and restate it as a Listenbury-native pack rule.

## Licensing/provenance policy

Do not copy wholesale upstream rule files into Listenbury-native packs without a deliberate license review. For now:

- record upstream paths as provenance;
- inventory behavior categories;
- hand-restate rules in Listenbury terms;
- keep generated/imported data clearly separated from original hand-authored Listenbury data;
- do not hide GPL-derived material inside renderer code.

## Glossary distinction

- **Source material**: upstream eSpeak-ng files and docs we inspect.
- **Inventory**: our catalog of behavior/data categories.
- **Translation**: human-designed restatement into Listenbury-native concepts.
- **Language pack**: Listenbury's top-level data model for a language/variety.
- **Renderer adapter**: backend-specific conversion from language-pack output into Riper/Klatt/MBROLA/Piper/etc.
