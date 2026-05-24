# Synthetic IR spine

Listenbury treats `SyntheticPlan` (`src/speech/synthetic_plan.rs`) as the
central synthesis IR between linguistic/prosodic analysis and backend-specific
synthesis payloads. Runtime synthesis is the incremental version of the same
idea: not one utterance object, but a train of timestamped chunks moving through
draft, planned, committed, and spoken horizons.

## Layering

| Representation | Layer |
| --- | --- |
| `ForcedAlignment`, `PraatProsodyAnalysis` | Source input |
| `ProsodyTimingPlan` | Source analysis output |
| `SyntheticPlan` / `SyntheticPhone` | Central synthesis IR spine |
| `PiperPhonemeSequence`, `PiperIdSequence`, `PiperTimingPlan` | Piper lowerings |
| `PhoneTimedPlan`, `MbrolaPhone`, `.pho` text | MBROLA lowerings |
| `PhoneAcousticTarget`, Klatt trajectory/params | Acoustic rendering target |
| Prosody audit / debug payloads | Trace/debug view |

## Boundary

```
text/alignment/g2p/singing intent
        ↓
SyntheticPlan
        ↓
backend lowerings (Piper, MBROLA, Klatt, diphone, debug)
```

## Streaming Synthetic Work Graph

`src/speech/work.rs` defines the runtime-facing stream contract around this IR:

```
TextStream
  -> LinguisticPlanStream
  -> AcousticPlanStream
  -> SpectralFrameStream
  -> RenderFrameStream
  -> WaveformStream
  -> AudioSink
```

The graph treats mel as one `SyntheticRepresentation` variant, not as the universal
middle of the pipeline. Renderers can accept `Mel`, `MelF0`, phone-timed
articulatory targets, source/filter tracks, coarse text, or already-rendered
wave chunks depending on their declared capabilities.

The generalized flow is:

```
text
  -> ling
  -> prosody/acoustics
  -> representation
  -> renderer
  -> waveform
  -> playback
```

`Mel` is one representation, not the throne. HiFi-GAN, BigVGAN, Griffin-Lim,
WORLD, LPCNet, Klatt, MBROLA, diphone, source/filter, post-filter, and
already-rendered wave paths should all fit behind the same renderer contract.

## Runtime Types

The synthetic work layer uses these key types:

| Type | Purpose |
| --- | --- |
| `TextChunk` | Streaming text plus `BoundaryHint` and `TextSource`. |
| `LingChunk` | Words, phones, syllables, phrase shape, and commitment. |
| `AcousticChunk` | Phone timing, F0, energy, voicing, breath, voice, and optional articulatory plan. |
| `SyntheticRepresentation` | Mel, MelF0, WORLD, LPCNet, articulatory, phone-timed, partial-prosody, coarse-text, source/filter, or wave payload. |
| `WaveChunk` | Timestamped samples after rendering. |
| `SyntheticEvent` | Playback-side say, pause, fade, or repair events. |
| `SyntheticWorkGraph` | Ticks registered stages under a shared `PipelineTime` and `WorkBudget`. |

Every chunk that can cross a revision boundary carries `Commitment`:

```
Draft -> Planned -> Committed -> Spoken
```

Draft and planned material may still be revised. Committed material is in the
mouth queue; spoken material is history. Corrections after that point are
represented as `RepairPlan`, not mutation.

## Clocks And Watermarks

Runtime synthesis has three clocks:

| Clock | Contract |
| --- | --- |
| `SyntheticClock::Audio` | Sample-clock work for the sink; it must not underrun. |
| `SyntheticClock::Frame` | Mel/acoustic frame work, normally around 5 to 20 frames per tick. |
| `SyntheticClock::Linguistic` | Irregular word, phrase, clause, sentence, or turn work. |

`SyntheticPipelineWatermarks` expresses the current buffer strategy:

| Buffer | Default target | Low-latency target |
| --- | ---: | ---: |
| Acoustic | 500 ms | 300 ms |
| Representation | 350 ms | 180 ms |
| Wave | 200 ms | 120 ms |
| Audio sink | 80 ms | 45 ms |

`SyntheticStageRuntimePolicy` binds a stage to its clock, lookahead target,
minimum commit boundary, and maximum latency. The current graph still uses simple
manual stage registration; policy objects are the vocabulary for deciding when a
stage has enough input, needs lookahead, is late, or should degrade.

## Repairs

If the system already spoke the wrong thing, history should remain immutable.
The renderer/playback side emits a repair event:

```
SyntheticEvent::Repair(RepairPlan)
```

Repair strategies include:

- `ContinueAsIfCorrect`
- `MicroPauseReplacement`
- `IMeanResume`
- `FullRestatement`

This keeps streaming LLM output usable: a partial phrase can be spoken with
suspensive prosody, and a later correction can be handled socially rather than
by rewriting audio that already left the speaker.

## Current vertical proof

- `ProsodyTimingPlan -> SyntheticPlan` is explicit via `synthetic_plan_from_prosody_timing`.
- MBROLA lowering now flows through `SyntheticPlan` in `prosody_timing_plan_to_phone_timed_plan`.
- Piper timing lowering now flows through `SyntheticPlan` in `prosody_plan_to_piper_timing`.
- `listenbury prosody-plan` builds a `SyntheticPlan` internally before reporting summary counts.
- `speech::work::BlockingVocoderRenderer` adapts existing `VocoderBackend`
  implementations into the representation-to-wave layer.
- `SyntheticClock`, `SyntheticPipelineWatermarks`, and
  `SyntheticStageRuntimePolicy` make stage timing explicit without forcing mel to
  be the universal middle.
