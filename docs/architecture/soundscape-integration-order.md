# Soundscape pipeline integration order

This adapter-first path wires current pipeline events into soundscape types without changing default runtime behavior.

## Current event seams wired by `SoundscapePipelineAdapter`

- Audio input: `audio::AudioFrame`
- Playback output: `mouth::player::PlaybackEvent::SpeechStarted`
- VAD decision: `hearing::VadResult`
- ASR hypothesis: `speech::transcript::TranscriptChunk`

These convert into:

- `ObservedSound`
- `ExpectedSound`
- `SoundscapeFrame`
- `SourceAttributedTranscript`

## Ticket dependency order

1. `#292` core soundscape types
2. `#293` source attribution hypotheses
3. `#298` expected playback traces
4. `#294` criteria-based isolation
5. `#296` voice counting
6. `#297` overlap representation
7. `#299` transcript/timeline output
8. `#300` debug view
9. `#301` source-separation adapters

## Known-voice registry privacy boundary

Known voice identities, enrollment samples, and embedding references are local-only
soundscape metadata. The first registry slice is intentionally explicit and does
not upload voice audio, embeddings, or identity labels to remote services.
