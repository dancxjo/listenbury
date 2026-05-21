# SpeechHypothesisEngine

`SpeechHypothesisEngine` (in `src/audio/lattice.rs`) is the top-level fusion
pipeline for speech hypotheses:

1. Collect active hypotheses from a `HypothesisLattice`.
2. Ask each `SpeechEvidenceSource` for `FusionInput` signals.
3. Merge/normalize confidence signals per hypothesis id.
4. Run `fuse_hypotheses` for final rescoring.
5. Classify spans into `stable_span_ids` and `revisable_span_ids`.

The default engine composes these first-class evidence sources:

- `acoustic`
- `phonetic`
- `transcript_stability` (ASR confidence + stable-prefix signals)
- `visual_speech`

## Adding a new evidence source

Implement `SpeechEvidenceSource`:

```rust
struct MyEvidenceSource;

impl SpeechEvidenceSource for MyEvidenceSource {
    fn name(&self) -> &'static str { "my_source" }

    fn collect(&self, lattice: &HypothesisLattice) -> Vec<(SpanHypothesisId, FusionInput)> {
        // map hypotheses to FusionInput fields
        Vec::new()
    }
}
```

Then register it:

```rust
let mut engine = SpeechHypothesisEngine::with_default_sources();
engine.add_source(MyEvidenceSource);
```

`SpeechHypothesisFusion` is serializable and includes `evidence_trace` to aid
debugging in logs, snapshots, and viewer tooling.
