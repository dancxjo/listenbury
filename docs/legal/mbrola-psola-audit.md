# MBROLA/PSOLA compliance audit (engineering checkpoint)

> This document is an engineering compliance checkpoint, not legal advice.

## Scope

This audit covers Listenbury's native MBROLA-style diphone/PSOLA work:

- MBROLA voice database parsing and symbol mapping
- `.pho` phone-duration + pitch-target rendering flow
- native Rust TD-PSOLA/diphone synthesis behavior
- generated neural diphone cache artifacts

## Current implementation boundary (known)

- Listenbury implements its own native Rust renderer and parser paths (`src/voice/mbrola/*`), and does not require embedding upstream MBROLA code to run.
- The public upstream MBROLA repository is AGPL-3.0 licensed; this project should treat that codebase as reference material unless we intentionally accept AGPL obligations.
- `mbrola` naming in this repository means compatibility with MBROLA data/flow conventions, not code derivation.
- Voice databases are external assets with voice-specific licenses/provenance that may differ by voice package.
- Generated neural diphone cache entries include provenance metadata (`CacheEntryMetadata`) and license notes, and are intended to remain local artifacts unless redistribution rights are confirmed.

## Patent status (known vs unknown)

### Known

- No formal freedom-to-operate search has been completed in this repository.
- Patent term is finite and must be checked by jurisdiction, filing date, and legal status records.

### Unknown / requires attorney review

- Whether any active patents currently cover the exact synthesis techniques used by this native implementation in jurisdictions where software is distributed.
- Which MBROLA-era patents (if any) are relevant to this implementation and whether each is expired, lapsed, or still enforceable in target jurisdictions.
- Whether any additional patent licenses are needed for distribution in specific regions.

## License/code-derivation boundary

- Do not copy AGPL MBROLA implementation code into Listenbury unless the project intentionally adopts compatible license obligations for that code path.
- Independent implementation of public ideas, documented file formats, and locally-authored parsing/synthesis logic remains the intended boundary.

## Voice database and cache artifact policy

- Do not vendor MBROLA voice databases into the repository unless the specific voice license explicitly permits redistribution and project policy approves it.
- Keep provenance metadata next to voice databases (for example, `manifest.toml`/`manifest.json`) including upstream URL, license identity, and redistribution/commercial/attribution flags.
- Treat generated/cached neural diphones as derived artifacts: keep local by default; only redistribute when model + voice licenses clearly allow it.
- CI/repository policy: do not commit non-redistributable voice databases or generated diphone caches.
  - Current guardrails: `/data/mbrola/` and `diphone-cache/` are gitignored in `.gitignore`.

## Naming/trademark confusion guidance

- Prefer wording such as "MBROLA-compatible", "MBROLA-format", or "native MBROLA-style renderer" in docs/UI for this independent implementation.
- Avoid naming that could imply shipping upstream MBROLA binaries/code unless that is explicitly true and licensed accordingly.

## Follow-up checklist

- [ ] Legal counsel review for patent status by target jurisdiction.
- [ ] Confirm provenance metadata for every locally used voice database.
- [ ] Confirm redistribution rights before publishing any derived cache artifacts.
