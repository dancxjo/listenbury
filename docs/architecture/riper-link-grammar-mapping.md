# Inventory: Link Grammar English categories mapped to Riper link kinds

## Scope

This document maps *concepts* from the English Link Grammar model to Riper's native syntax/prosody analysis model.

It does **not** import Link Grammar dictionary entries or rule files.

## Riper vocabulary used by this mapping

- `SyntacticLinkKind` and `ContextPredicate::SyntacticLink(...)` (`src/mouth/riper/sentence_analysis.rs`)
- `EnvironmentPattern` (`src/mouth/riper/sentence_analysis.rs`)
- `AnalysisClaim` and `ClaimKind` (`src/mouth/riper/evidence.rs`)

Riper flow stays:

1. detect links,
2. represent them as `EnvironmentPattern` predicates,
3. emit `AnalysisClaim`s,
4. resolve claims,
5. realize prosody/phonology.

No rule in this mapping should directly mutate pronunciation or prosody state.

## Link Grammar English category inventory (speech-relevant)

From Link Grammar English examples and docs (`README.md`, `data/en/tiny.dict`, `data/en/4.0.affix` in `opencog/link-grammar`), the most useful categories for Riper speech decisions are:

- subject ↔ verb links (`S*` family, e.g. `Ss`) for clause head/focus
- object/complement links (`O*`, complement-like copular patterns)
- determiner ↔ noun links (`D*`, e.g. `Ds`)
- modifier links (`A`, `M*`, `MV`-family usage)
- auxiliary ↔ verb links (`I`, `SI*`, `PP`, `Pg` families)
- coordination/connector links (`C`, `CO`, conjunction behavior)
- contrastive structures (`not ... but ...` and related conjunction contrast)
- apposition/parenthetical comma islands (comma punctuation links and clause-attachment behavior)
- vocative/direct-address-like comma marking (imperative comma notes)
- punctuation links (`Xp`, `Xc`, plus punctuation tokenization classes)
- phonetic agreement links (`PH`; `AN` usage in noun entries)
- morphological split classes (prefix/suffix/punctuation split categories in `4.0.affix`)

## Mapping table

| Link Grammar concept (English) | Riper `SyntacticLinkKind` | Claim emission target (`AnalysisClaim`) | Speech use | Priority |
| --- | --- | --- | --- | --- |
| `S*` subject↔verb (e.g. `Ss`) | `Subject` | `ClaimKind::ProsodicRole` (focus/content), `ClaimKind::BoundaryKind` for clause edges | phrase grouping, focus placement, weak-function suppression around head verb | P1 |
| `O*` verb↔object | `Object` | `ClaimKind::ProsodicRole` (content/focus) | phrase grouping, object focus in broad/narrow focus alternation | P1 |
| complement/copular dependency patterns | `Complement` | `ClaimKind::ProsodicRole` | copular complement prominence and phrase planning | P2 |
| `D*` determiner↔noun (e.g. `Ds`) | `Determiner` | `ClaimKind::WeakFunctionCandidate`, `ClaimKind::ProsodicRole` | function-word de-emphasis, article+noun chunking | P0 |
| adjectival/adverbial modifier families (`A`, `M*`, `MV`) | `Modifier` | `ClaimKind::ProsodicRole` | content emphasis/de-emphasis and local focus contrast | P1 |
| infinitival `to` marker (`I`/`TO` behavior) | `InfinitivalMarker` | `ClaimKind::InfinitivalMarker`, `ClaimKind::WeakFunctionCandidate` | weak `to` and reduction gating | P0 |
| auxiliary support (`SI*`, `PP`, `Pg`, modal/aux patterns) | `Auxiliary` | `ClaimKind::ProsodicRole`, optional `ClaimKind::WeakFunctionCandidate` | auxiliary de-emphasis unless contrastive | P0 |
| coordination (`C`, `CO`, conjunction links) | `Coordination` | `ClaimKind::BoundaryKind`, `ClaimKind::ProsodicRole` | phrase grouping and coordinated rhythm | P1 |
| explicit contrast (not X but Y) | `ContrastPair` | `ClaimKind::ContrastPair`, `ClaimKind::ProsodicRole` | contrastive stress on X/Y pair | P0 |
| direct-address comma behavior | `Vocative` | `ClaimKind::VocativeBoundary`, `ClaimKind::BoundaryKind` | direct address phrasing and pause control | P0 |
| non-restrictive/appositive comma islands | `Apposition` | `ClaimKind::AppositionBoundary`, `ClaimKind::BoundaryKind` | side-channel phrase lowering and pause insertion | P0 |
| parenthetical comma pair islands | `Parenthetical` | `ClaimKind::ParentheticalBoundary`, `ClaimKind::BoundaryKind` | phrase break and parenthetical de-emphasis | P0 |
| punctuation links (`Xp`, comma connectors, affix punctuation classes) | `Parenthetical` / `Coordination` / `Vocative` depending on pattern | `ClaimKind::BoundaryKind` | break strength and boundary timing | P1 |
| phonetic agreement (`PH`, plus `AN`-related article selection evidence) | (near-term) `Determiner`; (future) add dedicated `PhoneticAgreement` link kind | `ClaimKind::PhonemeRealization` and/or `ClaimKind::MorphologicalForm` | `a/an`, article allomorphy, reduced-vs-strong article realization | P0 |
| morphology/token splitting classes (`PRE`, `SUF`, `LPUNC`, `RPUNC`, `MPUNC`) | (pre-link stage; can feed `Modifier`/`Parenthetical`/`InfinitivalMarker`) | `ClaimKind::MorphologicalForm`, `ClaimKind::BoundaryKind` | contraction handling, clitic-like reduction, punctuation-aware phrasing | P1 |

Priority legend: `P0` = first wave grammar islands, `P1` = next, `P2` = later.

## Must-have grammar islands for first native implementation wave

1. **Infinitival `to`** (`InfinitivalMarker` + weak-function claim)
2. **Phonetic agreement for `a/an`** (`Determiner` + phoneme/morphological claim path)
3. **Weak/strong `the`** (determiner link + prosodic role/reduction claims)
4. **Vocative/direct address** (`Vocative` + boundary claims)
5. **Parenthetical comma pair** (`Parenthetical` + boundary claims)
6. **Contrastive `not X but Y`** (`ContrastPair` claim)
7. **Determiner + noun** (`Determiner` links for chunking and de-emphasis)
8. **Auxiliary + verb** (`Auxiliary` links for stress/de-emphasis gating)

## Licensing/import boundary

Keep this boundary explicit in code review and architecture docs:

- **Allowed:** use Link Grammar as an optional LGPL backend process/library boundary.
- **Allowed:** study published Link Grammar categories and write Riper-native rules/types.
- **Not allowed in this issue:** copy English dictionary entries/rule files into Riper source.

That keeps this issue in conceptual mapping/reimplementation territory, not dictionary-data import.

## Recommended next steps

1. Implement each P0 grammar island as a small heuristic that emits links.
2. Convert links to `EnvironmentPattern` predicates only.
3. Add explicit `AnalysisClaim` emitters per island (no direct prosody mutation).
4. Add fixtures that verify claims (contrast/vocative/parenthetical/`a-an`/`to`).
5. If needed later, add a dedicated `SyntacticLinkKind::PhoneticAgreement` and keep it Riper-native.

## External source pointers used for this mapping

- `https://github.com/opencog/link-grammar/blob/master/README.md`
- `https://github.com/opencog/link-grammar/blob/master/data/en/tiny.dict`
- `https://github.com/opencog/link-grammar/blob/master/data/en/4.0.affix`
