use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Phone {
    pub ipa: String,
    pub source_symbol: Option<String>,
    pub status: PhoneStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneStatus {
    Mapped,
    UnknownSymbol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneString {
    pub phones: Vec<Phone>,
}

impl Phone {
    /// Construct a [`Phone`] directly from an IPA string with no source symbol.
    ///
    /// Useful for constructing phones outside the ARPABET lookup path, e.g.
    /// in tests or when working with raw IPA input.
    pub fn new_ipa(ipa: impl Into<String>) -> Self {
        Self {
            ipa: ipa.into(),
            source_symbol: None,
            status: PhoneStatus::Mapped,
        }
    }

    /// Construct a [`Phone`] from a mapped IPA string.  Alias for [`new_ipa`].
    ///
    /// Preferred in phonotactic table construction and test assertions where
    /// the name `mapped` emphasises that the phone has a known IPA symbol.
    pub fn mapped(ipa: impl Into<String>) -> Self {
        Self::new_ipa(ipa)
    }
}

impl PhoneString {
    /// Construct an empty [`PhoneString`].
    pub fn empty() -> Self {
        Self { phones: vec![] }
    }

    /// Concatenate the IPA strings of all contained phones into a single
    /// `String`.
    ///
    /// # Example
    ///
    /// ```
    /// use listenbury::linguistic::phonology::{Phone, PhoneString};
    ///
    /// let ps = PhoneString { phones: vec![
    ///     Phone::new_ipa("s"),
    ///     Phone::new_ipa("t"),
    ///     Phone::new_ipa("ɹ"),
    ///     Phone::new_ipa("ʌ"),
    /// ]};
    /// assert_eq!(ps.to_ipa(), "stɹʌ");
    /// ```
    pub fn to_ipa(&self) -> String {
        self.phones.iter().map(|p| p.ipa.as_str()).collect()
    }

    /// Return the IPA string for each contained phone as a `Vec<&str>`.
    ///
    /// Useful for phonotactic table lookup and diagnostics without allocating
    /// a concatenated string.
    pub fn ipa_segments(&self) -> Vec<&str> {
        self.phones.iter().map(|p| p.ipa.as_str()).collect()
    }

    /// Build a single-phone [`PhoneString`] from one [`Phoneme`] by reading
    /// its current [`realization.ipa`][Realization] field.
    ///
    /// For a slice of phonemes, use [`from_phoneme_slice`].
    ///
    /// # Note — future multi-phone realizations
    ///
    /// Currently `Realization` stores a single IPA string.  When realizations
    /// become proper [`PhoneString`]s (affricates, diphthongs, syllabic
    /// consonants), this function will naturally extend to multi-phone output.
    pub fn from_realized(phoneme: &Phoneme) -> Self {
        Self {
            phones: vec![Phone {
                ipa: phoneme.realization.ipa.clone(),
                source_symbol: Some(phoneme.source_symbol.clone()),
                status: PhoneStatus::Mapped,
            }],
        }
    }

    /// Build a [`PhoneString`] from a slice of [`Phoneme`]s by reading each
    /// phoneme's current [`realization.ipa`][Realization] field.
    ///
    /// This is the primary bridge from a phoneme sequence to the phone
    /// representation used by the syllabifier.
    pub fn from_phoneme_slice(phonemes: &[Phoneme]) -> Self {
        Self {
            phones: phonemes
                .iter()
                .map(|p| Phone {
                    ipa: p.realization.ipa.clone(),
                    source_symbol: Some(p.source_symbol.clone()),
                    status: PhoneStatus::Mapped,
                })
                .collect(),
        }
    }
}

// ─── Phone comparison ─────────────────────────────────────────────────────────

/// Controls how two [`Phone`]s are compared for equality.
///
/// `ExactIpa` is the default: two phones match only if their `ipa` strings are
/// identical.  `Broad` applies the active [`PhoneEqualityOptions`] flags to
/// strip length marks, tie bars, and/or diacritics before comparing, allowing
/// `[tʰ]` to match `/t/` when aspiration is considered non-contrastive.
///
/// `Segmental` is reserved for a future feature-bundle comparison layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneComparisonMode {
    /// Two phones are equal only when their IPA strings are byte-identical.
    ExactIpa,
    /// Apply the [`PhoneEqualityOptions`] normalization flags before comparing.
    Broad,
    /// Reserved: future feature-bundle / distinctive-feature comparison.
    Segmental,
}

/// Policy for comparing two [`Phone`]s for "same segment" equality.
///
/// The default compares IPA strings exactly.  Set flags to strip phonetic
/// detail that the calling context does not care about.
///
/// # Example
///
/// ```
/// use listenbury::linguistic::phonology::{Phone, PhoneComparisonMode, PhoneEqualityOptions, phones_equivalent};
///
/// let t    = Phone::mapped("t");
/// let t_h  = Phone::mapped("tʰ");
///
/// // Exact mode: aspirated /tʰ/ ≠ plain /t/
/// assert!(!phones_equivalent(&t, &t_h, &PhoneEqualityOptions::default()));
///
/// // Broad mode + ignore_diacritics: /tʰ/ ≈ /t/
/// let broad = PhoneEqualityOptions {
///     mode: PhoneComparisonMode::Broad,
///     ignore_diacritics: true,
///     ..Default::default()
/// };
/// assert!(phones_equivalent(&t, &t_h, &broad));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneEqualityOptions {
    /// Which comparison algorithm to use.
    pub mode: PhoneComparisonMode,
    /// Strip IPA stress diacritics (ˈ ˌ) before comparing.
    pub ignore_stress: bool,
    /// Strip length marks (ː ˑ) before comparing.
    pub ignore_length: bool,
    /// Strip tie bars (͡ ͜) before comparing.
    pub ignore_tie_bars: bool,
    /// Strip superscript modifier diacritics (aspiration ʰ, labialization ʷ,
    /// palatalization ʲ, etc.) and combining marks before comparing.
    pub ignore_diacritics: bool,
}

impl Default for PhoneEqualityOptions {
    fn default() -> Self {
        Self {
            mode: PhoneComparisonMode::ExactIpa,
            ignore_stress: false,
            ignore_length: false,
            ignore_tie_bars: false,
            ignore_diacritics: false,
        }
    }
}

/// Normalize a phone's IPA string according to `options`, producing a
/// comparison key.
pub fn phone_comparison_key(phone: &Phone, options: &PhoneEqualityOptions) -> String {
    if options.mode == PhoneComparisonMode::ExactIpa {
        return phone.ipa.clone();
    }
    let mut s = phone.ipa.clone();
    if options.ignore_stress {
        // ˈ U+02C8, ˌ U+02CC
        s = s.replace('\u{02C8}', "").replace('\u{02CC}', "");
    }
    if options.ignore_length {
        // ː U+02D0, ˑ U+02D1
        s = s.replace('\u{02D0}', "").replace('\u{02D1}', "");
    }
    if options.ignore_tie_bars {
        // ͡ U+0361, ͜ U+035C
        s = s.replace('\u{0361}', "").replace('\u{035C}', "");
    }
    if options.ignore_diacritics {
        // Strip superscript modifier letters (U+02B0..=U+02FF) and combining
        // diacritical marks (U+0300..=U+036F), but only those that represent
        // phonetic detail rather than base segment identity.
        s = s
            .chars()
            .filter(|c| {
                let cp = *c as u32;
                // Keep: regular IPA base characters, length (already stripped above),
                // stress (already stripped above).
                // Remove: superscript modifiers (aspiration, labialization, …) and
                // combining diacritics (dental, nasalization, …).
                // Spacing modifier letters (02B0–02FF) excluding stress marks
                // ˈ (02C8) and ˌ (02CC), plus combining diacritics (0300–036F).
                !matches!(cp, 0x02B0..=0x02C7 | 0x02C9..=0x02CB | 0x02CD..=0x02FF | 0x0300..=0x036F)
            })
            .collect();
    }
    s
}

/// Return `true` if `left` and `right` represent the same phone segment under
/// the given comparison `options`.
///
/// This is the canonical comparison function for phonotactic lookup: it lets
/// `[tʰ]` match `/t/` when `ignore_diacritics` is set, while keeping `/t/`
/// distinct from `/ɾ/` (a different segment, not a diacritic variant).
pub fn phones_equivalent(left: &Phone, right: &Phone, options: &PhoneEqualityOptions) -> bool {
    phone_comparison_key(left, options) == phone_comparison_key(right, options)
}

// ─── Phonemic inventory ───────────────────────────────────────────────────────

/// Stable identifier for a phonological variety.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VarietyId(pub String);

impl VarietyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Stable identifier for a phoneme within a variety's inventory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PhonemeId(pub String);

impl PhonemeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// A source-schema symbol that maps to a phoneme in the inventory.
///
/// For example, the CMU Pronouncing Dictionary symbol `"AE"` maps to the IPA
/// phoneme `/æ/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceSymbol {
    /// The schema this symbol comes from.
    pub schema: PhonemeSchema,
    /// The symbol string, e.g. `"AE"`, `"æ"`.
    pub symbol: String,
}

/// A single phoneme entry in a [`PhonemicInventory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhonemeDefinition {
    /// Stable identifier for this phoneme.
    pub id: PhonemeId,
    /// Canonical IPA representation.
    pub ipa: String,
    /// Source symbols across different encoding schemas.
    pub source_symbols: Vec<SourceSymbol>,
    /// Default phone realization as a sequence of zero or more phones.
    ///
    /// Most phonemes realize as a single [`Phone`]; affricates, diphthongs,
    /// and syllabic consonants may realize as multiple phones.
    pub default_phone_string: PhoneString,
    /// Broad phonological class(es) for this phoneme.
    pub classes: Vec<PhonemeClass>,
}

impl PhonemeDefinition {
    /// Return the canonical default [`Phone`] (first element of
    /// `default_phone_string`), or a freshly constructed phone from the IPA
    /// string if the phone string is empty.
    pub fn default_phone(&self) -> Phone {
        self.default_phone_string
            .phones
            .first()
            .cloned()
            .unwrap_or_else(|| Phone::mapped(self.ipa.clone()))
    }
}

/// The phoneme inventory and phone comparison policy for a specific linguistic
/// variety.
///
/// This is the phonological backbone consumed by
/// [`crate::prosody::phonotactics::EnglishPhonotactics`] and the syllabifier.  It is *not* the same as
/// [`crate::linguistic::variety::LinguisticVariety`], which handles runtime
/// configuration; `PhonemicInventory` is purely about phonological facts.
///
/// Construct via [`crate::linguistic::variety::EnglishVariety::phonemic_inventory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhonemicInventory {
    /// Stable identifier for this variety.
    pub id: VarietyId,
    /// ISO 639 language code, e.g. `"en"`.
    pub language: String,
    /// Human-readable label, e.g. `"General American English"`.
    pub label: String,
    /// All phonemes in this variety's inventory.
    pub phonemes: Vec<PhonemeDefinition>,
    /// Phone comparison policy.
    pub phone_equality: PhoneEqualityOptions,
}

impl PhonemicInventory {
    /// Return all phoneme definitions that belong to `class`.
    pub fn phonemes_of_class(&self, class: PhonemeClass) -> Vec<&PhonemeDefinition> {
        self.phonemes
            .iter()
            .filter(|def| def.classes.contains(&class))
            .collect()
    }

    /// Look up the phoneme definition whose canonical IPA matches `ipa`
    /// using this inventory's equality policy.
    pub fn find_by_ipa(&self, ipa: &str) -> Option<&PhonemeDefinition> {
        let query = Phone::mapped(ipa);
        self.phonemes.iter().find(|def| {
            let canonical = Phone::mapped(def.ipa.clone());
            phones_equivalent(&query, &canonical, &self.phone_equality)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhonemeSchema {
    Arpabet,
    Cmudict,
    ArpabetSurface,
    Ipa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WordPosition {
    Singleton,
    WordInitial,
    WordMedial,
    WordFinal,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    pub left_phone: Option<String>,
    pub right_phone: Option<String>,
    pub left_class: Option<String>,
    pub right_class: Option<String>,
    pub word_position: Option<WordPosition>,
    pub syllable_position: Option<String>,
    pub stress_context: Option<String>,
    pub phrase_position: Option<String>,
    pub language: Option<String>,
    pub dialect: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhonemeClass {
    Vowel,
    Consonant,
    AlveolarStop,
    AlveolarNasal,
    VelarStop,
    VelarConsonant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionPattern {
    Any,
    Singleton,
    Initial,
    Medial,
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StressPattern {
    Any,
    Stressed,
    Unstressed,
    Primary,
    Secondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanState {
    Candidate,
    Stable,
    Committed,
    Invalidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchCommitment {
    Provisional,
    Stable,
    Committed,
    Invalidated,
}

impl MatchCommitment {
    fn from_span_state(state: SpanState) -> Self {
        match state {
            SpanState::Candidate => Self::Provisional,
            SpanState::Stable => Self::Stable,
            SpanState::Committed => Self::Committed,
            SpanState::Invalidated => Self::Invalidated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphemeKind {
    Prefix,
    Stem,
    Suffix,
    Clitic,
    CompoundMember,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartOfSpeech {
    Noun,
    Verb,
    Auxiliary,
    Determiner,
    Preposition,
    Pronoun,
    Adverb,
    Adjective,
    Conjunction,
    Particle,
    ProperName,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProsodicRole {
    Content,
    FunctionWeak,
    FunctionStrong,
    Contrastive,
    Focus,
    DirectAddress,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhraseBoundaryKind {
    None,
    Minor,
    Major,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TargetPattern {
    Symbol(String),
    Symbols(Vec<String>),
    PhonemeClass(PhonemeClass),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TimingPredicate {
    AtOrBeforeNow,
    AtNow,
    SpanState(SpanState),
    ConfidenceAtLeast(f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ContextPredicate {
    Symbol(String),
    PhoneIpa(String),
    PhonemeClass(PhonemeClass),
    Stress(StressPattern),
    MorphemeKind(MorphemeKind),
    WordText(String),
    Pos(PartOfSpeech),
    ProsodicRole(ProsodicRole),
    BoundaryKind(PhraseBoundaryKind),
    SpanState(SpanState),
    ConfidenceAtLeast(f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentPattern {
    pub target: TargetPattern,
    pub left: Vec<ContextPredicate>,
    pub right: Vec<ContextPredicate>,
    pub contains: Vec<ContextPredicate>,
    pub overlaps: Vec<ContextPredicate>,
    pub word_position: Option<PositionPattern>,
    pub syllable_position: Option<PositionPattern>,
    pub phrase_position: Option<PositionPattern>,
    pub stress: Option<StressPattern>,
    pub language: Option<String>,
    pub variety: Option<String>,
    pub timing: Vec<TimingPredicate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextSide {
    Target,
    Left,
    Right,
    Contains,
    Overlaps,
    Timing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredicateDiagnostic {
    pub side: ContextSide,
    pub predicate: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentMatch {
    pub rule: String,
    pub target: String,
    pub matched_environment: Environment,
    pub matched_predicates: Vec<PredicateDiagnostic>,
    pub commitment: MatchCommitment,
    pub result: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Phoneme,
    Syllable,
    Word,
    Clause,
    Prosody,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpanRef {
    pub id: String,
    pub kind: SpanKind,
    pub start_index: usize,
    pub end_index: usize,
    pub state: SpanState,
    pub confidence: f32,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineCursor {
    pub now_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct RuleMatchContext<'a> {
    pub sequence: &'a [Phoneme],
    pub index: usize,
    pub now: TimelineCursor,
    pub target_span: SpanRef,
    pub previous_span: Option<SpanRef>,
    pub next_span: Option<SpanRef>,
    pub parent_word_span: Option<SpanRef>,
    pub containing_syllable_span: Option<SpanRef>,
    pub containing_clause_span: Option<SpanRef>,
    pub neighboring_word_spans: Vec<SpanRef>,
    pub phrase_boundary: Option<PhraseBoundaryKind>,
    pub prosodic_role: Option<ProsodicRole>,
    pub morphology: Option<MorphemeKind>,
    pub part_of_speech: Option<PartOfSpeech>,
    pub span_state: SpanState,
    pub confidence: f32,
    pub language: String,
    pub variety: String,
}

impl<'a> RuleMatchContext<'a> {
    pub fn from_sequence(
        sequence: &'a [Phoneme],
        index: usize,
        config: &RealizationConfig,
    ) -> Self {
        let confidence = config.confidence.clamp(0.0, 1.0);
        let target_span = SpanRef {
            id: format!("phoneme:{index}"),
            kind: SpanKind::Phoneme,
            start_index: index,
            end_index: index + 1,
            state: config.span_state,
            confidence,
            text: Some(sequence[index].symbol.clone()),
        };
        let previous_span = index.checked_sub(1).map(|left| SpanRef {
            id: format!("phoneme:{left}"),
            kind: SpanKind::Phoneme,
            start_index: left,
            end_index: left + 1,
            state: config.span_state,
            confidence,
            text: Some(sequence[left].symbol.clone()),
        });
        let next_span = (index + 1 < sequence.len()).then(|| SpanRef {
            id: format!("phoneme:{}", index + 1),
            kind: SpanKind::Phoneme,
            start_index: index + 1,
            end_index: index + 2,
            state: config.span_state,
            confidence,
            text: Some(sequence[index + 1].symbol.clone()),
        });
        let parent_word_span = (!sequence.is_empty()).then(|| SpanRef {
            id: "word:0".to_string(),
            kind: SpanKind::Word,
            start_index: 0,
            end_index: sequence.len(),
            state: config.span_state,
            confidence,
            text: Some(
                sequence
                    .iter()
                    .map(|phoneme| phoneme.symbol.as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
        });

        Self {
            sequence,
            index,
            now: TimelineCursor {
                now_index: config.now_index,
            },
            target_span,
            previous_span,
            next_span,
            parent_word_span,
            containing_syllable_span: None,
            containing_clause_span: None,
            neighboring_word_spans: Vec::new(),
            phrase_boundary: config.phrase_boundary,
            prosodic_role: config.prosodic_role,
            morphology: config.morpheme_kind,
            part_of_speech: config.part_of_speech,
            span_state: config.span_state,
            confidence,
            language: config.language.clone(),
            variety: config.dialect.clone(),
        }
    }

    pub fn target(&self) -> &Phoneme {
        &self.sequence[self.index]
    }

    pub fn left(&self) -> Option<&Phoneme> {
        self.index.checked_sub(1).map(|index| &self.sequence[index])
    }

    pub fn right(&self) -> Option<&Phoneme> {
        (self.index + 1 < self.sequence.len()).then(|| &self.sequence[self.index + 1])
    }

    pub fn commitment(&self) -> MatchCommitment {
        MatchCommitment::from_span_state(self.span_state)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllophoneRule {
    pub id: String,
    pub applies_to_symbols: Vec<String>,
    pub output_ipa: String,
    pub environment_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationMethod {
    Default,
    AllophoneRule,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Realization {
    pub ipa: String,
    pub method: RealizationMethod,
    pub rule: Option<String>,
    pub environment: Option<Environment>,
    pub environment_match: Option<EnvironmentMatch>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Phoneme {
    pub symbol: String,
    pub source_symbol: String,
    pub source: String,
    pub stress: Option<Stress>,
    pub default_phone_string: PhoneString,
    pub realization: Realization,
}

impl Phoneme {
    pub fn new(symbol: impl Into<String>) -> Self {
        let symbol = symbol.into();
        phoneme_from_arpabet(&symbol, "manual")
    }

    pub fn symbols_in_schema(&self, schema: PhonemeSchema) -> Vec<String> {
        match schema {
            PhonemeSchema::Arpabet => vec![self.source_symbol.clone()],
            PhonemeSchema::Cmudict => vec![self.source_symbol.clone()],
            PhonemeSchema::ArpabetSurface => {
                if self.is_realized_american_english_tap() {
                    vec!["DX".to_string()]
                } else {
                    vec![self.source_symbol.clone()]
                }
            }
            PhonemeSchema::Ipa => vec![self.realization.ipa.clone()],
        }
    }

    pub fn symbol_in_schema(&self, schema: PhonemeSchema) -> String {
        self.symbols_in_schema(schema).join(" ")
    }

    fn is_realized_american_english_tap(&self) -> bool {
        matches!(self.realization.method, RealizationMethod::AllophoneRule)
            && self.symbol == "T"
            && self.realization.ipa == "ɾ"
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealizationConfig {
    pub enable_allophone_rules: bool,
    pub language: String,
    pub dialect: String,
    pub now_index: Option<usize>,
    pub span_state: SpanState,
    pub confidence: f32,
    pub phrase_boundary: Option<PhraseBoundaryKind>,
    pub prosodic_role: Option<ProsodicRole>,
    pub morpheme_kind: Option<MorphemeKind>,
    pub part_of_speech: Option<PartOfSpeech>,
    pub careful_style: bool,
}

impl Default for RealizationConfig {
    fn default() -> Self {
        Self {
            enable_allophone_rules: false,
            language: "en".to_string(),
            dialect: "american_english".to_string(),
            now_index: None,
            span_state: SpanState::Candidate,
            confidence: 1.0,
            phrase_boundary: None,
            prosodic_role: None,
            morpheme_kind: None,
            part_of_speech: None,
            careful_style: false,
        }
    }
}

pub fn phoneme_from_arpabet(symbol: &str, source: &str) -> Phoneme {
    let (base, stress) = split_arpabet_symbol(symbol);
    let phone = default_phone_for_arpabet(&base, symbol);
    let default_phone_string = PhoneString {
        phones: vec![phone],
    };
    let ipa = default_phone_string.phones[0].ipa.clone();
    Phoneme {
        symbol: base,
        source_symbol: symbol.to_string(),
        source: source.to_string(),
        stress,
        default_phone_string,
        realization: Realization {
            ipa,
            method: RealizationMethod::Default,
            rule: None,
            environment: None,
            environment_match: None,
        },
    }
}

pub fn realize_sequence(sequence: &[Phoneme], config: &RealizationConfig) -> Vec<Phoneme> {
    if !config.enable_allophone_rules {
        return sequence.to_vec();
    }

    let mut realized = sequence.to_vec();
    let rules = declarative_environment_rules(config);
    for i in 0..realized.len() {
        let context = RuleMatchContext::from_sequence(&realized, i, config);
        for rule in &rules {
            if let Some(mut environment_match) = match_environment_pattern(rule, &context) {
                if rule.id == "american_english_intervocalic_flapping" {
                    if config.careful_style
                        || matches!(config.phrase_boundary, Some(PhraseBoundaryKind::Major))
                    {
                        continue;
                    }
                    environment_match.matched_environment.stress_context =
                        Some("between stressed vowel and unstressed vowel".to_string());
                }

                realized[i].realization = Realization {
                    ipa: rule.output_ipa.clone(),
                    method: RealizationMethod::AllophoneRule,
                    rule: Some(rule.id.clone()),
                    environment: Some(environment_match.matched_environment.clone()),
                    environment_match: Some(environment_match),
                };
                break;
            }
        }
    }
    realized
}

pub fn realize_sequence_as_schema(
    sequence: &[Phoneme],
    config: &RealizationConfig,
    schema: PhonemeSchema,
) -> Vec<String> {
    realize_sequence(sequence, config)
        .iter()
        .flat_map(|phoneme| phoneme.symbols_in_schema(schema))
        .collect()
}

#[derive(Debug, Clone)]
struct DeclarativeAllophoneRule {
    id: String,
    output_ipa: String,
    pattern: EnvironmentPattern,
}

fn declarative_environment_rules(config: &RealizationConfig) -> Vec<DeclarativeAllophoneRule> {
    vec![
        DeclarativeAllophoneRule {
            id: "american_english_intervocalic_flapping".to_string(),
            output_ipa: "ɾ".to_string(),
            pattern: EnvironmentPattern {
                target: TargetPattern::Symbol("T".to_string()),
                left: vec![
                    ContextPredicate::PhonemeClass(PhonemeClass::Vowel),
                    ContextPredicate::Stress(StressPattern::Stressed),
                ],
                right: vec![
                    ContextPredicate::PhonemeClass(PhonemeClass::Vowel),
                    ContextPredicate::Stress(StressPattern::Unstressed),
                ],
                contains: Vec::new(),
                overlaps: Vec::new(),
                word_position: Some(PositionPattern::Medial),
                syllable_position: None,
                phrase_position: None,
                stress: None,
                language: Some(config.language.clone()),
                variety: Some(config.dialect.clone()),
                timing: vec![TimingPredicate::AtOrBeforeNow],
            },
        },
        DeclarativeAllophoneRule {
            id: "alveolar_nasal_velar_assimilation".to_string(),
            output_ipa: "ŋ".to_string(),
            pattern: EnvironmentPattern {
                target: TargetPattern::PhonemeClass(PhonemeClass::AlveolarNasal),
                left: Vec::new(),
                right: vec![ContextPredicate::PhonemeClass(PhonemeClass::VelarStop)],
                contains: Vec::new(),
                overlaps: Vec::new(),
                word_position: None,
                syllable_position: None,
                phrase_position: None,
                stress: None,
                language: Some(config.language.clone()),
                variety: Some(config.dialect.clone()),
                timing: vec![TimingPredicate::AtOrBeforeNow],
            },
        },
    ]
}

fn match_environment_pattern(
    rule: &DeclarativeAllophoneRule,
    context: &RuleMatchContext<'_>,
) -> Option<EnvironmentMatch> {
    let mut diagnostics = Vec::new();
    let target = context.target();
    if !target_pattern_matches(&rule.pattern.target, target) {
        return None;
    }
    diagnostics.push(PredicateDiagnostic {
        side: ContextSide::Target,
        predicate: format!("{:?}", rule.pattern.target),
    });

    for predicate in &rule.pattern.left {
        if !context_predicate_matches(predicate, context.left(), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Left,
            predicate: format!("{:?}", predicate),
        });
    }

    for predicate in &rule.pattern.right {
        if !context_predicate_matches(predicate, context.right(), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Right,
            predicate: format!("{:?}", predicate),
        });
    }

    for predicate in &rule.pattern.contains {
        if !context_predicate_matches(predicate, Some(target), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Contains,
            predicate: format!("{:?}", predicate),
        });
    }

    for predicate in &rule.pattern.overlaps {
        if !context_predicate_matches(predicate, Some(target), context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Overlaps,
            predicate: format!("{:?}", predicate),
        });
    }

    if let Some(position) = rule.pattern.word_position {
        if !position_pattern_matches(
            position,
            word_position(context.index, context.sequence.len()),
        ) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Target,
            predicate: format!("word_position:{position:?}"),
        });
    }

    if let Some(stress_pattern) = rule.pattern.stress {
        if !stress_pattern_matches(stress_pattern, target.stress) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Target,
            predicate: format!("stress:{stress_pattern:?}"),
        });
    }

    if let Some(language) = &rule.pattern.language
        && language != &context.language
    {
        return None;
    }
    if let Some(variety) = &rule.pattern.variety
        && variety != &context.variety
    {
        return None;
    }

    for timing in &rule.pattern.timing {
        if !timing_predicate_matches(timing, context) {
            return None;
        }
        diagnostics.push(PredicateDiagnostic {
            side: ContextSide::Timing,
            predicate: format!("{:?}", timing),
        });
    }

    let left = context.left();
    let right = context.right();
    let matched_environment = Environment {
        left_phone: left.map(|phone| phone.realization.ipa.clone()),
        right_phone: right.map(|phone| phone.realization.ipa.clone()),
        left_class: left.map(phoneme_class_label),
        right_class: right.map(phoneme_class_label),
        word_position: Some(word_position(context.index, context.sequence.len())),
        syllable_position: None,
        stress_context: None,
        phrase_position: context
            .phrase_boundary
            .map(|boundary| format!("{boundary:?}").to_lowercase()),
        language: Some(context.language.clone()),
        dialect: Some(context.variety.clone()),
    };

    Some(EnvironmentMatch {
        rule: rule.id.clone(),
        target: target.symbol.clone(),
        matched_environment,
        matched_predicates: diagnostics,
        commitment: context.commitment(),
        result: rule.output_ipa.clone(),
    })
}

fn target_pattern_matches(pattern: &TargetPattern, target: &Phoneme) -> bool {
    match pattern {
        TargetPattern::Symbol(symbol) => &target.symbol == symbol,
        TargetPattern::Symbols(symbols) => symbols.iter().any(|symbol| symbol == &target.symbol),
        TargetPattern::PhonemeClass(class) => phoneme_class_matches(*class, target),
    }
}

fn context_predicate_matches(
    predicate: &ContextPredicate,
    phone: Option<&Phoneme>,
    context: &RuleMatchContext<'_>,
) -> bool {
    match predicate {
        ContextPredicate::Symbol(symbol) => phone.is_some_and(|entry| entry.symbol == *symbol),
        ContextPredicate::PhoneIpa(ipa) => phone.is_some_and(|entry| entry.realization.ipa == *ipa),
        ContextPredicate::PhonemeClass(class) => {
            phone.is_some_and(|entry| phoneme_class_matches(*class, entry))
        }
        ContextPredicate::Stress(pattern) => {
            phone.is_some_and(|entry| stress_pattern_matches(*pattern, entry.stress))
        }
        ContextPredicate::MorphemeKind(kind) => context.morphology == Some(*kind),
        ContextPredicate::WordText(text) => context
            .parent_word_span
            .as_ref()
            .and_then(|span| span.text.as_ref())
            .is_some_and(|surface| surface == text),
        ContextPredicate::Pos(pos) => context.part_of_speech == Some(*pos),
        ContextPredicate::ProsodicRole(role) => context.prosodic_role == Some(*role),
        ContextPredicate::BoundaryKind(boundary) => context.phrase_boundary == Some(*boundary),
        ContextPredicate::SpanState(state) => context.span_state == *state,
        ContextPredicate::ConfidenceAtLeast(minimum) => context.confidence >= *minimum,
    }
}

fn timing_predicate_matches(predicate: &TimingPredicate, context: &RuleMatchContext<'_>) -> bool {
    match predicate {
        TimingPredicate::AtOrBeforeNow => context
            .now
            .now_index
            .is_none_or(|now_index| context.index <= now_index),
        TimingPredicate::AtNow => context.now.now_index == Some(context.index),
        TimingPredicate::SpanState(state) => &context.span_state == state,
        TimingPredicate::ConfidenceAtLeast(minimum) => context.confidence >= *minimum,
    }
}

fn position_pattern_matches(pattern: PositionPattern, position: WordPosition) -> bool {
    matches!(
        (pattern, position),
        (PositionPattern::Any, _)
            | (PositionPattern::Singleton, WordPosition::Singleton)
            | (PositionPattern::Initial, WordPosition::WordInitial)
            | (PositionPattern::Medial, WordPosition::WordMedial)
            | (PositionPattern::Final, WordPosition::WordFinal)
    )
}

fn stress_pattern_matches(pattern: StressPattern, stress: Option<Stress>) -> bool {
    match pattern {
        StressPattern::Any => true,
        StressPattern::Stressed => matches!(stress, Some(Stress::Primary | Stress::Secondary)),
        StressPattern::Unstressed => matches!(stress, Some(Stress::Unstressed)),
        StressPattern::Primary => matches!(stress, Some(Stress::Primary)),
        StressPattern::Secondary => matches!(stress, Some(Stress::Secondary)),
    }
}

fn classify_symbol(base: &str) -> PhonemeClass {
    if is_vowel_symbol(base) {
        return PhonemeClass::Vowel;
    }
    match base {
        "T" | "D" => PhonemeClass::AlveolarStop,
        "N" => PhonemeClass::AlveolarNasal,
        "K" | "G" => PhonemeClass::VelarStop,
        "NG" => PhonemeClass::VelarConsonant,
        _ => PhonemeClass::Consonant,
    }
}

fn phoneme_class_name(class: PhonemeClass) -> &'static str {
    match class {
        PhonemeClass::Vowel => "vowel",
        PhonemeClass::Consonant => "consonant",
        PhonemeClass::AlveolarStop => "alveolar_stop",
        PhonemeClass::AlveolarNasal => "alveolar_nasal",
        PhonemeClass::VelarStop => "velar_stop",
        PhonemeClass::VelarConsonant => "velar_consonant",
    }
}

fn phoneme_class_label(phoneme: &Phoneme) -> String {
    phoneme_class_name(classify_symbol(&phoneme.symbol)).to_string()
}

fn phoneme_class_matches(class: PhonemeClass, phoneme: &Phoneme) -> bool {
    let actual = classify_symbol(&phoneme.symbol);
    let actual_is_consonant = matches!(
        actual,
        PhonemeClass::Consonant
            | PhonemeClass::AlveolarStop
            | PhonemeClass::AlveolarNasal
            | PhonemeClass::VelarStop
            | PhonemeClass::VelarConsonant
    );
    matches!(
        (class, actual),
        (PhonemeClass::Vowel, PhonemeClass::Vowel)
            | (PhonemeClass::AlveolarStop, PhonemeClass::AlveolarStop)
            | (PhonemeClass::AlveolarNasal, PhonemeClass::AlveolarNasal)
            | (PhonemeClass::VelarStop, PhonemeClass::VelarStop)
            | (PhonemeClass::VelarConsonant, PhonemeClass::VelarConsonant)
            // VelarConsonant intentionally includes velar stops as a subtype.
            | (PhonemeClass::VelarConsonant, PhonemeClass::VelarStop)
    ) || (class == PhonemeClass::Consonant && actual_is_consonant)
}

fn word_position(index: usize, len: usize) -> WordPosition {
    if len <= 1 {
        WordPosition::Singleton
    } else if index == 0 {
        WordPosition::WordInitial
    } else if index == len - 1 {
        WordPosition::WordFinal
    } else {
        WordPosition::WordMedial
    }
}

fn split_arpabet_symbol(symbol: &str) -> (String, Option<Stress>) {
    match symbol.chars().last() {
        Some('1') => (
            symbol[..symbol.len() - 1].to_string(),
            Some(Stress::Primary),
        ),
        Some('2') => (
            symbol[..symbol.len() - 1].to_string(),
            Some(Stress::Secondary),
        ),
        Some('0') => (
            symbol[..symbol.len() - 1].to_string(),
            Some(Stress::Unstressed),
        ),
        _ => (symbol.to_string(), None),
    }
}

fn default_phone_for_arpabet(base: &str, source_symbol: &str) -> Phone {
    let mapped = match base {
        "AA" => Some("ɑ"),
        "AE" => Some("æ"),
        "AH" => Some("ʌ"),
        "AO" => Some("ɔ"),
        "AW" => Some("aʊ"),
        "AY" => Some("aɪ"),
        "B" => Some("b"),
        "CH" => Some("tʃ"),
        "D" => Some("d"),
        "DH" => Some("ð"),
        "EH" => Some("ɛ"),
        "ER" => Some("ɝ"),
        "EY" => Some("eɪ"),
        "F" => Some("f"),
        "G" => Some("ɡ"),
        "HH" => Some("h"),
        "IH" => Some("ɪ"),
        "IY" => Some("iː"),
        "JH" => Some("dʒ"),
        "K" => Some("k"),
        "L" => Some("l"),
        "M" => Some("m"),
        "N" => Some("n"),
        "NG" => Some("ŋ"),
        "OW" => Some("oʊ"),
        "OY" => Some("ɔɪ"),
        "P" => Some("p"),
        "R" => Some("ɹ"),
        "S" => Some("s"),
        "SH" => Some("ʃ"),
        "T" => Some("t"),
        "TH" => Some("θ"),
        "UH" => Some("ʊ"),
        "UW" => Some("uː"),
        "V" => Some("v"),
        "W" => Some("w"),
        "Y" => Some("j"),
        "Z" => Some("z"),
        "ZH" => Some("ʒ"),
        _ => None,
    };
    match mapped {
        Some(ipa) => Phone {
            ipa: ipa.to_string(),
            source_symbol: Some(source_symbol.to_string()),
            status: PhoneStatus::Mapped,
        },
        None => Phone {
            ipa: format!("?{base}"),
            source_symbol: Some(source_symbol.to_string()),
            status: PhoneStatus::UnknownSymbol,
        },
    }
}

fn is_vowel_symbol(base: &str) -> bool {
    matches!(
        base,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arpabet_to_ipa_mapping_preserves_stress_metadata() {
        let phoneme = phoneme_from_arpabet("IY1", "cmudict");
        assert_eq!(phoneme.symbol, "IY");
        assert_eq!(phoneme.source_symbol, "IY1");
        assert_eq!(phoneme.stress, Some(Stress::Primary));
        assert_eq!(phoneme.default_phone_string.phones[0].ipa, "iː");
        assert_eq!(phoneme.realization.ipa, "iː");
        assert_eq!(phoneme.realization.method, RealizationMethod::Default);
    }

    #[test]
    fn unknown_symbol_falls_back_safely() {
        let phoneme = phoneme_from_arpabet("QH9", "cmudict");
        assert_eq!(phoneme.symbol, "QH9");
        assert_eq!(phoneme.stress, None);
        assert_eq!(phoneme.default_phone_string.phones[0].ipa, "?QH9");
        assert_eq!(
            phoneme.default_phone_string.phones[0].status,
            PhoneStatus::UnknownSymbol
        );
    }

    #[test]
    fn opt_in_flapping_rule_realizes_t_between_stressed_and_unstressed_vowels() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "ɾ");
        assert_eq!(
            realized[1].realization.method,
            RealizationMethod::AllophoneRule
        );
        assert_eq!(
            realized[1].realization.rule.as_deref(),
            Some("american_english_intervocalic_flapping")
        );
        assert_eq!(
            realized[1]
                .realization
                .environment
                .as_ref()
                .and_then(|env| env.stress_context.as_deref()),
            Some("between stressed vowel and unstressed vowel")
        );
        assert!(
            realized[1]
                .realization
                .environment_match
                .as_ref()
                .is_some_and(|m| m.commitment == MatchCommitment::Provisional)
        );
    }

    #[test]
    fn central_phoneme_presents_realized_tap_by_requested_schema() {
        let seq = vec![
            phoneme_from_arpabet("EY2", "morphophonology"),
            phoneme_from_arpabet("T", "morphophonology"),
            phoneme_from_arpabet("IH0", "morphophonology"),
        ];

        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );

        assert_eq!(realized[1].symbol, "T");
        assert_eq!(realized[1].realization.ipa, "ɾ");
        assert_eq!(
            realized[1].realization.rule.as_deref(),
            Some("american_english_intervocalic_flapping")
        );
        assert_eq!(realized[1].symbol_in_schema(PhonemeSchema::Arpabet), "T");
        assert_eq!(
            realized[1].symbol_in_schema(PhonemeSchema::ArpabetSurface),
            "DX"
        );
        assert_eq!(realized[1].symbol_in_schema(PhonemeSchema::Ipa), "ɾ");
        assert_eq!(
            realize_sequence_as_schema(
                &seq,
                &RealizationConfig {
                    enable_allophone_rules: true,
                    ..RealizationConfig::default()
                },
                PhonemeSchema::ArpabetSurface,
            ),
            vec!["EY2", "DX", "IH0"]
        );
    }

    #[test]
    fn flapping_rule_does_not_apply_to_d() {
        let seq = vec![
            phoneme_from_arpabet("EH1", "cmudict"),
            phoneme_from_arpabet("D", "cmudict"),
            phoneme_from_arpabet("IY0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].symbol, "D");
        assert_eq!(realized[1].realization.ipa, "d");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn flapping_rule_requires_following_unstressed_vowel() {
        let seq = vec![
            phoneme_from_arpabet("AH0", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("IH2", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn allophone_rules_are_opt_in() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(&seq, &RealizationConfig::default());
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn nasal_assimilation_realizes_n_before_velars() {
        let seq = vec![
            phoneme_from_arpabet("IH0", "cmudict"),
            phoneme_from_arpabet("N", "cmudict"),
            phoneme_from_arpabet("K", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "ŋ");
        assert_eq!(
            realized[1].realization.rule.as_deref(),
            Some("alveolar_nasal_velar_assimilation")
        );
        assert_eq!(
            realized[1]
                .realization
                .environment
                .as_ref()
                .and_then(|env| env.right_class.as_deref()),
            Some("velar_stop")
        );
    }

    #[test]
    fn nasal_assimilation_does_not_apply_before_non_velars() {
        let seq = vec![
            phoneme_from_arpabet("IH0", "cmudict"),
            phoneme_from_arpabet("N", "cmudict"),
            phoneme_from_arpabet("D", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "n");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn commitment_follows_span_state() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                span_state: SpanState::Committed,
                ..RealizationConfig::default()
            },
        );
        assert!(
            realized[1]
                .realization
                .environment_match
                .as_ref()
                .is_some_and(|m| m.commitment == MatchCommitment::Committed)
        );
    }

    #[test]
    fn flapping_rule_is_blocked_in_careful_style() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                careful_style: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }
}
