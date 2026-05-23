# Native Link Grammar audit (Riper sentence analysis)

This audit documents the **existing native implementation** and where parity work remains.

## Native modules and current responsibilities

| Area | File(s) | Current behavior |
| --- | --- | --- |
| Link graph model | `src/mouth/riper/sentence_analysis.rs` (`SyntacticLink`, `SyntacticLinkKind`, `SyntacticLinkParse`) | Native link graph representation with ranked parses and source metadata (`HeuristicGrammarIsland` / `AmbiguityVariant`). |
| Connector/rule heuristics | `src/mouth/riper/sentence_analysis.rs` (`build_link_parses`, `emit_*` helpers) | Declarative-ish rule helpers for subject/object/complement, determiner, auxiliary, preposition, modifier, infinitival marker, contrast pair, vocative, apposition, parenthetical, noun compound. |
| EnvironmentPattern bridge | `src/mouth/riper/sentence_analysis.rs` (`as_environment_pattern`, `environment_patterns`) | Exposes parse links as `ContextPredicate::SyntacticLink(...)` for downstream pattern matching. |
| Speech claim emission | `src/mouth/riper/sentence_analysis.rs` + `src/mouth/riper/evidence.rs` | Emits `AnalysisClaim` objects (contrast, weak-form candidate, vocative/apposition/parenthetical boundaries, article agreement hooks) with confidence/rationale. |
| Claim conflict/resolution | `src/mouth/riper/evidence.rs` | Priority/conflict handling for competing claims before prosody realization. |
| Downstream use in prosody/G2P | `src/mouth/riper/prosody_planner.rs`, `src/mouth/riper/g2p.rs` | Consumes link-derived claims and syntax facts for boundary/stress/reduction decisions. |
| Mapping and licensing boundary | `docs/architecture/riper-link-grammar-mapping.md` | Upstream Link Grammar category → native `SyntacticLinkKind` mapping and explicit “no dictionary copy” policy. |

## Coverage status for first parity islands

| Sentence fixture | Link kinds/claims currently covered |
| --- | --- |
| `I want to go.` | `InfinitivalMarker` link + weak-function/infinitival claims + environment predicate |
| `I said TO, not FROM.` | `ContrastPair` link + contrast claims |
| `Thank you, Dave.` | `Vocative` link + vocative boundary claims |
| `The machine, unfortunately, exploded.` | `Parenthetical` link + boundary claims |
| `My brother, who lives in Tacoma, arrived.` | `Apposition` link + boundary claims |
| `a dog / an owl / the owl / the door` | `Determiner` link + article agreement/weak-form hooks |

These are validated in `src/mouth/riper/sentence_analysis.rs` tests, including the parity fixture suite.

## Gaps to continue toward broader Link Grammar parity

1. Expand connector inventory for richer coordination and relative-clause variants beyond current heuristics.
2. Add more ambiguity fixtures (coordination scope, PP attachment variants, punctuation-heavy clauses) with normalized expected links.
3. Add optional comparison helper that diffs normalized native links against an installed external Link Grammar backend output.
4. Keep provenance/license notes attached to parity fixtures and avoid importing upstream dictionary/rule files directly.
