# Canonical speech IR spine

Listenbury now treats `CanonicalSpeechPlan` (`src/speech/canonical_plan.rs`) as the canonical speech IR between linguistic/prosodic analysis and backend-specific synthesis payloads.

## Layering

| Representation | Layer |
| --- | --- |
| `ForcedAlignment`, `PraatProsodyAnalysis` | Source input |
| `ProsodyTimingPlan` | Source analysis output |
| `CanonicalSpeechPlan` / `CanonicalSpeechPhone` | Canonical speech IR spine |
| `PiperPhonemeSequence`, `PiperIdSequence`, `PiperTimingPlan` | Piper lowerings |
| `PhoneTimedPlan`, `MbrolaPhone`, `.pho` text | MBROLA lowerings |
| `PhoneAcousticTarget`, Klatt trajectory/params | Acoustic rendering target |
| Prosody audit / debug payloads | Trace/debug view |

## Boundary

```
text/alignment/g2p/singing intent
        ↓
CanonicalSpeechPlan
        ↓
backend lowerings (Piper, MBROLA, Klatt, diphone, debug)
```

## Current vertical proof

- `ProsodyTimingPlan -> CanonicalSpeechPlan` is explicit via `canonical_speech_plan_from_prosody_timing`.
- MBROLA lowering now flows through the canonical IR in `prosody_timing_plan_to_phone_timed_plan`.
- Piper timing lowering now flows through the canonical IR in `prosody_plan_to_piper_timing`.
- `listenbury prosody-plan` builds a canonical plan internally before reporting summary counts.
