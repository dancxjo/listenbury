# Speech Pipeline

Listenbury speech synthesis is organized as an explicit staged pipeline. The
`say` command selects components and output policy, then runs a
`SpeechPipeline` instead of branching directly across backend-specific
synthesis worlds.

```text
text
  -> linguistic analysis
  -> phone/prosody plan
  -> acoustic plan
  -> vocoder/renderer
  -> mouth sink
```

The shared public stage contracts live in `src/speech/pipeline.rs`:

- `LinguisticPlan` is the analyzed text plus a shared `PhonePlan`.
- `PhonePlan` is the cross-backend phone/timing representation.
- `AcousticPlan` names the selected route and carries the phone plan into the
  acoustic or compatibility layer.
- `AudioRender` is rendered audio frames plus a playback/source label.

Current `say` backend flags are compatibility selectors for stage
implementations:

- default / `--riper`: Piper-compatible ONNX route.
- `--piper`: external Piper process route.
- `--klatt`: Klatt formant renderer route.
- `--diphone`: MBROLA-compatible diphone route.
- `--hifigan`: SpeechT5 acoustic route with SpeechT5 HiFi-GAN vocoding.
- `--hifigan --skip-gan`: source-filter acoustic route with the mel debug
  renderer.

Future work should land by implementing or replacing a stage, not by adding a
new command-wide branch:

- MBROLA belongs under acoustic planning and rendering for diphone plans.
- Neural diphone generation belongs behind the diphone acoustic/renderer stage.
- SpeechT5 belongs in the acoustic stage, with HiFi-GAN as a vocoder stage.
- Piper parity belongs behind the Piper-compatible renderer and tensor/phone ID
  bridge.

Debug output should prefer stage names (`phone-plan`, `acoustic-plan`, `mel`,
`audio`, `pipeline`) over backend-specific trapdoors.
