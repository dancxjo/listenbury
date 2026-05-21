use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

/// A unique identifier for an analysis claim.
pub type ClaimId = u64;

static CLAIM_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Allocate a fresh, globally unique [`ClaimId`].
pub fn next_claim_id() -> ClaimId {
    CLAIM_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// ── Target ────────────────────────────────────────────────────────────────────

/// The linguistic object that an [`AnalysisClaim`] is about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisTarget {
    /// A single word slot by word index.
    WordIndex(usize),
    /// A range of word slots (e.g., both poles of a contrast pair).
    WordRange(Vec<usize>),
    /// A single token slot (includes punctuation / phrase breaks).
    TokenIndex(usize),
    /// A phoneme within a word.
    PhonemeIndex { word: usize, phoneme: usize },
    /// The boundary point between two words (e.g., a comma site).
    Boundary {
        left_word: Option<usize>,
        right_word: Option<usize>,
    },
}

// ── Source categories ─────────────────────────────────────────────────────────

/// Broad category of the analyser that produced an [`AnalysisClaim`].
///
/// The category determines the default priority used during conflict resolution:
/// explicit user intent outranks acoustic evidence, which outranks lexical
/// knowledge, which outranks heuristic rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisSourceKind {
    /// A dictionary or lexical database lookup (e.g., CMUdict exception entries).
    Lexicon,
    /// A productive morphology rule (e.g., suffix stripping + re-stress).
    MorphologyRule,
    /// A syntactic or grammar-island rule (e.g., infinitival-marker detection).
    SyntaxRule,
    /// A prosodic rule (e.g., nuclear stress, focus projection).
    ProsodyRule,
    /// A punctuation-derived heuristic (e.g., comma → minor pause).
    PunctuationRule,
    /// A phonological environment-pattern match (e.g., nasal assimilation).
    EnvironmentPattern,
    /// Evidence derived from live acoustic / ASR analysis.
    AcousticEvidence,
    /// Explicit markup supplied by the user in the input text.
    UserMarkup,
    /// A manual, authoritative override (highest priority).
    ManualOverride,
    /// An imported third-party rule table (e.g., eSpeak-NG rule import).
    ImportedRuleTable,
}

/// Default conflict-resolution priority for a source kind.
///
/// Higher values beat lower values.  Ties are broken by [`AnalysisClaim::confidence`].
///
/// Policy summary:
/// - Manual overrides always win.
/// - Explicit user markup beats all automatic analysis.
/// - Committed acoustic evidence can override projected phonology.
/// - Dictionary exceptions override productive morphology.
/// - Syntax rules outrank prosodic defaults, which outrank punctuation defaults.
pub fn source_default_priority(source: AnalysisSourceKind) -> i32 {
    match source {
        AnalysisSourceKind::ManualOverride => 100,
        AnalysisSourceKind::UserMarkup => 90,
        AnalysisSourceKind::AcousticEvidence => 80,
        AnalysisSourceKind::Lexicon => 70,
        AnalysisSourceKind::SyntaxRule => 60,
        AnalysisSourceKind::ProsodyRule => 55,
        AnalysisSourceKind::PunctuationRule => 50,
        AnalysisSourceKind::MorphologyRule => 45,
        AnalysisSourceKind::EnvironmentPattern => 40,
        AnalysisSourceKind::ImportedRuleTable => 35,
    }
}

// ── Span-state lifecycle ──────────────────────────────────────────────────────

/// Lifecycle state of an [`AnalysisClaim`].
///
/// ```text
/// Hypothesis ──► Stable ──► Committed
///      │              │
///      └──────────────┴──► Revised / Invalidated
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanState {
    /// Tentative: emitted but not yet corroborated.
    Hypothesis,
    /// Corroborated and stable: context has not changed.
    Stable,
    /// Locked in for realization; will not be revised.
    Committed,
    /// Superseded by a higher-priority or newer claim; kept for diagnostics.
    Revised,
    /// No longer valid because the text or timeline changed.
    Invalidated,
}

// ── Claim kind ────────────────────────────────────────────────────────────────

/// What kind of linguistic fact an [`AnalysisClaim`] asserts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    // Syntactic / grammatical
    /// The word is an infinitival marker ("to" before a bare verb).
    InfinitivalMarker,
    /// The word is a candidate for weak-form reduction.
    WeakFunctionCandidate,
    /// Two words form a contrastive pair.
    ContrastPair,
    /// The word introduces or follows a vocative boundary.
    VocativeBoundary,
    /// The word is inside a parenthetical phrase.
    ParentheticalBoundary,
    /// The word introduces a non-restrictive appositive phrase.
    AppositionBoundary,

    // Prosodic
    /// The prosodic role assigned to a word (content, weak function, contrastive, …).
    ProsodicRole,
    /// The phonological reduction applied or blocked for a word.
    Reduction,
    /// The phrase-boundary kind at a word edge or punctuation site.
    BoundaryKind,

    // Phonological / phonetic
    /// The surface phoneme realization predicted by an environment pattern.
    PhonemeRealization,

    // Morphological
    /// The morphological analysis and pronunciation form of a word.
    MorphologicalForm,
}

// ── Claim value ───────────────────────────────────────────────────────────────

/// The value asserted by an [`AnalysisClaim`].
///
/// For structural claims (e.g., "this is an infinitival marker") where the kind
/// is sufficient, use [`ClaimValue::Syntactic`].  For claims that assert a
/// specific realization or category string, use the appropriate variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimValue {
    /// No additional value beyond the claim kind itself.
    Syntactic,
    /// A part-of-speech label (e.g., `"InfinitivalMarker"`).
    PartOfSpeech(String),
    /// A reduction form (e.g., `"WeakTo"`, `"None"`).
    Reduction(String),
    /// A phrase-boundary kind (e.g., `"Vocative"`, `"MinorPhrasePause"`).
    BoundaryKind(String),
    /// A prosodic role string (e.g., `"Contrastive"`, `"FunctionWeak"`).
    ProsodicRole(String),
    /// An IPA surface realization (e.g., `"[ŋ]"`).
    PhonemeRealization(String),
    /// A morphological pronunciation form.
    MorphologicalForm(String),
}

// ── Core claim struct ─────────────────────────────────────────────────────────

/// A provisional, ranked, inspectable assertion made by one analyser about one
/// linguistic object.
///
/// Claims are never deleted when superseded; they are marked [`SpanState::Revised`]
/// or [`SpanState::Invalidated`] and kept for diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisClaim {
    /// Unique identifier for this claim.
    pub id: ClaimId,
    /// The word / token / phoneme / boundary this claim is about.
    pub target: AnalysisTarget,
    /// What kind of linguistic fact is being asserted.
    pub kind: ClaimKind,
    /// The specific value asserted.
    pub value: ClaimValue,
    /// The analyser category that produced this claim.
    pub source: AnalysisSourceKind,
    /// Confidence in [0, 1].
    pub confidence: f32,
    /// Explicit conflict-resolution priority (overrides source default when set).
    pub priority: i32,
    /// Lifecycle state of this claim.
    pub span_state: SpanState,
    /// Other claim IDs that support (corroborate) this one.
    pub support: Vec<ClaimId>,
    /// Other claim IDs that conflict with this one.
    pub conflicts: Vec<ClaimId>,
    /// Human-readable explanation of why this claim was made.
    pub rationale: String,
}

impl AnalysisClaim {
    /// Create a new hypothesis-state claim, assigning a fresh [`ClaimId`] and
    /// deriving [`priority`](Self::priority) from the source default.
    pub fn new(
        target: AnalysisTarget,
        kind: ClaimKind,
        value: ClaimValue,
        source: AnalysisSourceKind,
        confidence: f32,
        rationale: impl Into<String>,
    ) -> Self {
        let priority = source_default_priority(source);
        Self {
            id: next_claim_id(),
            target,
            kind,
            value,
            source,
            confidence,
            priority,
            span_state: SpanState::Hypothesis,
            support: Vec::new(),
            conflicts: Vec::new(),
            rationale: rationale.into(),
        }
    }

    /// Mark this claim as [`SpanState::Invalidated`].
    ///
    /// Call this when the text or timeline has changed such that the analysis
    /// is no longer valid.  The claim is retained for diagnostics.
    pub fn invalidate(&mut self) {
        self.span_state = SpanState::Invalidated;
    }

    /// Mark this claim as [`SpanState::Revised`].
    ///
    /// Call this when a newer or higher-priority claim supersedes it.
    pub fn revise(&mut self) {
        self.span_state = SpanState::Revised;
    }

    /// Promote from [`SpanState::Hypothesis`] to [`SpanState::Stable`].
    ///
    /// Call this when the surrounding context confirms the claim.
    pub fn stabilize(&mut self) {
        if self.span_state == SpanState::Hypothesis {
            self.span_state = SpanState::Stable;
        }
    }

    /// Promote to [`SpanState::Committed`].
    ///
    /// Call this at the realization stage when the claim is locked in.
    pub fn commit(&mut self) {
        self.span_state = SpanState::Committed;
    }
}

// ── Conflict resolution ───────────────────────────────────────────────────────

/// Record of one losing claim in a conflict-resolution result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConflictEntry {
    /// The losing claim's ID.
    pub claim: ClaimId,
    /// The source of the losing claim.
    pub source: AnalysisSourceKind,
    /// Human-readable explanation of why this claim lost.
    pub reason_lost: String,
}

/// The outcome of resolving a set of competing claims for the same target and
/// kind.
///
/// Both the winner and all losers (with reasons) are preserved so diagnostics
/// can show why a decision was made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedAnalysis<T> {
    /// The winning interpretation.
    pub selected: T,
    /// ID of the claim that produced `selected`.
    pub selected_claim: ClaimId,
    /// IDs of non-conflicting alternatives (same target/kind but different
    /// value, not in direct conflict with the winner).
    pub alternatives: Vec<ClaimId>,
    /// Claims that were directly in conflict with the winner and lost.
    pub conflicts: Vec<ConflictEntry>,
    /// Confidence of the winning claim.
    pub confidence: f32,
}

/// Resolve a set of competing [`AnalysisClaim`]s using the default policy.
///
/// Invalidated claims are excluded.  The remaining claims are ranked by
/// [`priority`](AnalysisClaim::priority) descending, then by
/// [`confidence`](AnalysisClaim::confidence) descending.  The top-ranked claim
/// wins; all others become [`ConflictEntry`] records.
///
/// Returns `None` if `claims` is empty or all claims are invalidated.
///
/// # Type parameter
///
/// `T` is the value extracted from each candidate claim by `extract`.  The
/// caller decides what to extract (e.g., a `ClaimValue`, a
/// `PhraseBoundaryKind`, …).
pub fn resolve_claims<T, F>(claims: &[AnalysisClaim], extract: F) -> Option<ResolvedAnalysis<T>>
where
    F: Fn(&AnalysisClaim) -> T,
{
    let mut active: Vec<&AnalysisClaim> = claims
        .iter()
        .filter(|c| c.span_state != SpanState::Invalidated)
        .collect();

    if active.is_empty() {
        return None;
    }

    // Higher priority wins; ties broken by confidence (descending).
    active.sort_by(|a, b| {
        b.priority.cmp(&a.priority).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let winner = active[0];
    let selected = extract(winner);

    let conflicts: Vec<ConflictEntry> = active[1..]
        .iter()
        .map(|loser| ConflictEntry {
            claim: loser.id,
            source: loser.source,
            reason_lost: if loser.priority < winner.priority {
                format!(
                    "{:?} source priority ({}) is lower than {:?} ({})",
                    loser.source, loser.priority, winner.source, winner.priority
                )
            } else {
                format!(
                    "equal priority but lower confidence ({:.2} vs {:.2})",
                    loser.confidence, winner.confidence
                )
            },
        })
        .collect();

    let alternatives: Vec<ClaimId> = active[1..].iter().map(|c| c.id).collect();

    Some(ResolvedAnalysis {
        selected,
        selected_claim: winner.id,
        alternatives,
        conflicts,
        confidence: winner.confidence,
    })
}

// ── EnvironmentPattern → claim conversion ─────────────────────────────────────

/// Create an [`AnalysisClaim`] from a phonological
/// [`EnvironmentMatch`](crate::linguistic::phonology::EnvironmentMatch).
///
/// The claim asserts that the phoneme at `phoneme_index` within word
/// `word_index` realizes as the IPA string in the match's `result` field.
///
/// `rule_name` is a short identifier for the rule (e.g.
/// `"alveolar_nasal_before_velar"`).
pub fn claim_from_environment_match(
    env_match: &crate::linguistic::phonology::EnvironmentMatch,
    word_index: usize,
    phoneme_index: usize,
    rule_name: &str,
) -> AnalysisClaim {
    AnalysisClaim::new(
        AnalysisTarget::PhonemeIndex {
            word: word_index,
            phoneme: phoneme_index,
        },
        ClaimKind::PhonemeRealization,
        ClaimValue::PhonemeRealization(env_match.result.clone()),
        AnalysisSourceKind::EnvironmentPattern,
        0.85,
        format!(
            "environment_pattern.{rule_name}: target /{target}/ realizes as [{result}]",
            rule_name = rule_name,
            target = env_match.target,
            result = env_match.result,
        ),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn weak_to_claim(id_override: Option<ClaimId>) -> AnalysisClaim {
        let mut c = AnalysisClaim::new(
            AnalysisTarget::WordIndex(2),
            ClaimKind::WeakFunctionCandidate,
            ClaimValue::Reduction("WeakTo".to_string()),
            AnalysisSourceKind::SyntaxRule,
            0.76,
            "infinitival to before verb is weak",
        );
        if let Some(id) = id_override {
            c.id = id;
        }
        c
    }

    fn contrastive_to_claim(id_override: Option<ClaimId>) -> AnalysisClaim {
        let mut c = AnalysisClaim::new(
            AnalysisTarget::WordIndex(2),
            ClaimKind::ProsodicRole,
            ClaimValue::ProsodicRole("Contrastive".to_string()),
            AnalysisSourceKind::SyntaxRule,
            0.91,
            "all-caps TO signals contrastive stress; blocks weak-form reduction",
        );
        if let Some(id) = id_override {
            c.id = id;
        }
        c
    }

    fn vocative_boundary_claim() -> AnalysisClaim {
        AnalysisClaim::new(
            AnalysisTarget::WordIndex(3),
            ClaimKind::VocativeBoundary,
            ClaimValue::BoundaryKind("Vocative".to_string()),
            AnalysisSourceKind::SyntaxRule,
            0.86,
            "comma-delimited proper name after verb is a vocative boundary",
        )
    }

    fn default_comma_pause_claim() -> AnalysisClaim {
        AnalysisClaim::new(
            AnalysisTarget::WordIndex(3),
            ClaimKind::BoundaryKind,
            ClaimValue::BoundaryKind("MinorPhrasePause".to_string()),
            AnalysisSourceKind::PunctuationRule,
            0.55,
            "comma defaults to a minor phrase pause",
        )
    }

    fn lexicon_pronunciation_claim() -> AnalysisClaim {
        AnalysisClaim::new(
            AnalysisTarget::WordIndex(0),
            ClaimKind::MorphologicalForm,
            ClaimValue::MorphologicalForm("K EH1 D Z".to_string()),
            AnalysisSourceKind::Lexicon,
            0.95,
            "dictionary entry: cadz -> K EH1 D Z",
        )
    }

    fn morphology_pronunciation_claim() -> AnalysisClaim {
        AnalysisClaim::new(
            AnalysisTarget::WordIndex(0),
            ClaimKind::MorphologicalForm,
            ClaimValue::MorphologicalForm("K AE1 D Z".to_string()),
            AnalysisSourceKind::MorphologyRule,
            0.60,
            "productive morphology: stem cad + plural -s",
        )
    }

    // ── source priority table ─────────────────────────────────────────────────

    #[test]
    fn manual_override_has_highest_priority() {
        assert!(
            source_default_priority(AnalysisSourceKind::ManualOverride)
                > source_default_priority(AnalysisSourceKind::UserMarkup)
        );
        assert!(
            source_default_priority(AnalysisSourceKind::UserMarkup)
                > source_default_priority(AnalysisSourceKind::AcousticEvidence)
        );
        assert!(
            source_default_priority(AnalysisSourceKind::AcousticEvidence)
                > source_default_priority(AnalysisSourceKind::Lexicon)
        );
        assert!(
            source_default_priority(AnalysisSourceKind::Lexicon)
                > source_default_priority(AnalysisSourceKind::SyntaxRule)
        );
        assert!(
            source_default_priority(AnalysisSourceKind::SyntaxRule)
                > source_default_priority(AnalysisSourceKind::PunctuationRule)
        );
        assert!(
            source_default_priority(AnalysisSourceKind::PunctuationRule)
                > source_default_priority(AnalysisSourceKind::MorphologyRule)
        );
        assert!(
            source_default_priority(AnalysisSourceKind::MorphologyRule)
                > source_default_priority(AnalysisSourceKind::EnvironmentPattern)
        );
    }

    // ── weak `to` vs contrastive `TO` ────────────────────────────────────────

    /// When "TO" is written in all-caps, a contrastive claim (SyntaxRule, high
    /// confidence) should beat the default weak-function-candidate claim.
    #[test]
    fn contrastive_to_beats_weak_to() {
        let weak = weak_to_claim(Some(1));
        let contrastive = contrastive_to_claim(Some(2));

        // Both claims target the same word; resolve them.
        let resolved = resolve_claims(&[weak, contrastive.clone()], |c| c.value.clone())
            .expect("should resolve");

        // Contrastive wins because it has higher confidence (0.91 > 0.76)
        // at equal SyntaxRule priority.
        assert_eq!(resolved.selected_claim, contrastive.id);
        assert_eq!(
            resolved.selected,
            ClaimValue::ProsodicRole("Contrastive".to_string())
        );
        assert_eq!(resolved.conflicts.len(), 1);
        assert_eq!(resolved.conflicts[0].claim, 1); // weak_to lost
    }

    // ── vocative comma vs default comma pause ─────────────────────────────────

    /// The VocativeBoundary claim (SyntaxRule, priority 60) overrides the
    /// default comma MinorPhrasePause (PunctuationRule, priority 50).
    #[test]
    fn vocative_boundary_beats_default_comma_pause() {
        let vocative = vocative_boundary_claim();
        let default_pause = default_comma_pause_claim();

        let vocative_id = vocative.id;
        let resolved = resolve_claims(&[default_pause, vocative], |c| c.value.clone())
            .expect("should resolve");

        assert_eq!(resolved.selected_claim, vocative_id);
        assert_eq!(
            resolved.selected,
            ClaimValue::BoundaryKind("Vocative".to_string())
        );
        let lost_reason = &resolved.conflicts[0].reason_lost;
        assert!(
            lost_reason.contains("lower than"),
            "reason should mention lower priority: {lost_reason}"
        );
    }

    // ── dictionary exception vs productive morphology ─────────────────────────

    /// Lexicon source (priority 70) beats MorphologyRule (priority 45).
    #[test]
    fn lexicon_beats_productive_morphology() {
        let lex = lexicon_pronunciation_claim();
        let morph = morphology_pronunciation_claim();

        let lex_id = lex.id;
        let resolved = resolve_claims(&[morph, lex], |c| c.value.clone()).expect("should resolve");

        assert_eq!(resolved.selected_claim, lex_id);
        assert_eq!(
            resolved.selected,
            ClaimValue::MorphologicalForm("K EH1 D Z".to_string())
        );
    }

    // ── provisional analysis invalidation ────────────────────────────────────

    /// If a claim is invalidated (e.g., because the text changed), it must be
    /// excluded from resolution.  If all claims are invalidated, `resolve_claims`
    /// returns `None`.
    #[test]
    fn invalidated_claims_are_excluded_from_resolution() {
        let mut claim = weak_to_claim(None);
        claim.invalidate();

        let result = resolve_claims(&[claim], |c| c.value.clone());
        assert!(
            result.is_none(),
            "invalidated claim must not participate in resolution"
        );
    }

    /// A mix of a valid and an invalidated claim resolves to the valid one only.
    #[test]
    fn valid_claim_wins_over_invalidated_claim() {
        let mut stale = weak_to_claim(Some(10));
        stale.invalidate();

        let fresh = contrastive_to_claim(Some(11));

        let resolved = resolve_claims(&[stale, fresh.clone()], |c| c.value.clone())
            .expect("should resolve");

        assert_eq!(resolved.selected_claim, fresh.id);
        assert!(
            resolved.conflicts.is_empty(),
            "no active conflicts — the stale claim was excluded"
        );
    }

    /// A claim begins as `Hypothesis` and can be advanced through the lifecycle.
    #[test]
    fn claim_lifecycle_transitions() {
        let mut claim = weak_to_claim(None);
        assert_eq!(claim.span_state, SpanState::Hypothesis);

        claim.stabilize();
        assert_eq!(claim.span_state, SpanState::Stable);

        claim.commit();
        assert_eq!(claim.span_state, SpanState::Committed);
    }

    #[test]
    fn revised_claim_is_still_active_in_resolution() {
        let mut old_claim = weak_to_claim(Some(20));
        old_claim.revise();
        assert_eq!(old_claim.span_state, SpanState::Revised);

        // `Revised` claims are NOT invalidated — they represent a superseded but
        // still-informative interpretation that diagnostics should show.
        let result = resolve_claims(&[old_claim.clone()], |c| c.value.clone());
        assert!(
            result.is_some(),
            "revised (not invalidated) claim should still participate"
        );
    }

    // ── EnvironmentPattern claim emission ─────────────────────────────────────

    #[test]
    fn environment_match_emits_phoneme_realization_claim() {
        let env_match = crate::linguistic::phonology::EnvironmentMatch {
            rule: "alveolar_nasal_before_velar".to_string(),
            target: "n".to_string(),
            matched_environment: crate::linguistic::phonology::Environment::default(),
            matched_predicates: Vec::new(),
            commitment: crate::linguistic::phonology::MatchCommitment::Provisional,
            result: "ŋ".to_string(),
        };

        let claim = claim_from_environment_match(&env_match, 1, 2, "alveolar_nasal_before_velar");

        assert_eq!(claim.kind, ClaimKind::PhonemeRealization);
        assert_eq!(
            claim.value,
            ClaimValue::PhonemeRealization("ŋ".to_string())
        );
        assert_eq!(claim.source, AnalysisSourceKind::EnvironmentPattern);
        assert_eq!(
            claim.target,
            AnalysisTarget::PhonemeIndex {
                word: 1,
                phoneme: 2
            }
        );
        assert_eq!(claim.span_state, SpanState::Hypothesis);
    }

    // ── claim IDs are unique ──────────────────────────────────────────────────

    #[test]
    fn next_claim_id_is_monotonically_increasing() {
        let a = next_claim_id();
        let b = next_claim_id();
        assert!(b > a);
    }
}
