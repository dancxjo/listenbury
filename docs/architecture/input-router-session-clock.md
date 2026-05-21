# InputRouter and SessionClock timing model

## SessionClock

`SessionClock` is the runtime timing authority for live session work. It provides:

- monotonic runtime timestamps (`now()`)
- normalized session-relative time (`elapsed_ms`)
- deterministic conversion from elapsed offsets to wall-clock timestamps (`at_elapsed_ms`)

All normalized trace timing is derived from the same session start anchor to avoid per-command timestamp drift.

## InputRouter

`InputRouter` is the single routing surface for live web/native capture controls:

- native microphone enable/disable
- browser microphone enable/disable
- browser audio frame routing
- browser video frame routing

Arbitration guarantee:

- enabling browser mic disables native mic capture
- enabling native mic disables browser mic capture

This keeps one active microphone source authoritative at a time.

## Trace provenance and normalized timing

Live trace events now carry stable provenance (`source`) plus normalized timing fields:

- `normalized_unix_ns`
- `normalized_elapsed_ms`

Ordering guarantee:

1. Event wall-clock (`t_unix_ns`) and normalized wall-clock (`normalized_unix_ns`) are emitted from the session clock normalization path.
2. Session-relative ordering uses `normalized_elapsed_ms`.
3. Routed browser camera events use explicit provenance (`browser.camera`), while trace-runtime events use `runtime.trace`.
