# Soundscape pipeline integration order

This adapter-first path wires current pipeline events into soundscape types without changing default runtime behavior.

## Current event seams wired by `SoundscapePipelineAdapter`

- Audio input: `audio::AudioFrame`
- Playback output: `mouth::player::PlaybackEvent::SyntheticStarted`
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
soundscape metadata. The registry persists to local memory state
(`listenbury_data/memory/known_voices.json`) and enrollment vectors are stored in
the local Qdrant collection `listenbury_known_voice_enrollments` with payload
metadata for provenance (`voice_id`, `voice_label`, `voice_kind`,
`enrollment_sample_id`, `audio_span_id`, `source`, `quality`, timestamps, and
`memory_scope=local_only`).

This voice-identity memory is intentionally separate from transcript/text memory:
transcript traces are conversational content for recall/summarization, while
voice identity memory is probabilistic speaker evidence used for enrollment,
matching, and attribution.
