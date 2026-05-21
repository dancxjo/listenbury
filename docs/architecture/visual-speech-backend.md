# Backend Visual Speech Evidence

Listenbury treats camera input as local audiovisual evidence for speech analysis,
not as identity recognition and not as a lip-reading replacement for ASR.

## Capture Path

WaveDeck can request browser camera access and transport transient `rgba8` frames
to the backend at `/api/browser-video-frame`. The browser does not compute
mouth-motion features. Rust receives each frame, extracts a `VisualSpeechFrame`,
and stores only the derived feature trace in `LiveSessionVisualSpeechStore`.

Raw frame bytes are ordinary HTTP request bodies only. They are not persisted by
default and are not exposed as a debug artifact.

## Timing

Each transported frame carries:

- `X-Capture-Elapsed-Ms`
- `X-Frame-Duration-Ms`
- `X-Frame-Index`
- `X-Capture-Time-Ms`
- `X-Audio-Offset-Ms`

The backend records these in `VisualProvenance` and maps each feature frame onto
the audio timeline as `timeRangeMs`. `AvSyncConfig` controls manual video offset,
maximum usable desync, and confidence decay. If sync or mouth visibility is poor,
visual evidence is degraded to weak, advisory-only, or unusable.

## Evidence Fusion

The backend maps derived mouth features into `VisualSpeechClaim` values and can
convert those claims into `SpanHypothesis` and `FusionInput` values. Bilabial
closure evidence supports or conflicts with /p, b, m/ candidates without
bypassing the existing hypothesis lattice and fusion scorer.
