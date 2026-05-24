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

## Canonical runtime envelope and causality

Live traces and memory journal entries now both emit a canonical `RuntimeEvent`
envelope (`runtime_event`) with:

- stable `id`
- `timestamp` (wall-clock)
- `monotonic_ms` (session-relative ordering anchor)
- typed `kind` (`domain` + typed event payload)
- source/provenance tag (`source`)
- `causality` references and coarse `correlation` tags

Event ordering semantics:

1. Within one session, `monotonic_ms` is the primary ordering field.
2. `timestamp` is used for wall-clock comparison across sessions/subsystems.
3. When `monotonic_ms` is equal, consumers should retain source order or break
   ties with stable `id`.

Causality semantics:

- `causality` entries point to upstream IDs or correlation anchors
  (`turn:*`, `utterance:*`, `synthetic_unit:*`, etc.).
- `correlation` is a looser grouping key for cross-subsystem joins and replay
  slices.
- Consumers should treat missing `causality` as unknown ancestry (not an error).
