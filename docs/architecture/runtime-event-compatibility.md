# Runtime Event Classification Compatibility

`RuntimeEventKind` should be produced from typed runtime domain decisions at event producers.

## Preferred path (current)

- Producers set a typed `RuntimeEventKind` on `LiveTraceEvent` (for example via `LiveTraceEvent::set_runtime_kind`).
- `RuntimeEvent::from_live_trace_event` preserves that typed kind.

## Legacy compatibility path

- If a `LiveTraceEvent` has no typed runtime kind, `RuntimeEvent::from_live_trace_event` falls back to
  `legacy_classify_runtime_kind_from_string`.
- This keeps historical trace replay compatible with old `kind` string naming conventions.

## Current status — all live producers still use the legacy path

As of this writing **every** live runtime event producer emits events via `LiveTraceEvent::new`
followed by field mutations but **never** calls `LiveTraceEvent::set_runtime_kind`.  The
legacy string fallback therefore classifies every live event.  The following producers are
the primary sources of events that must be migrated before the legacy adapter can be deleted:

| Producer file | Representative kind strings | Migration status |
|---|---|---|
| `src/cli/commands/live_half_duplex.rs` | `capture_started`, `speech_started`, `asr_started`, `transcript`, `first_llm_token`, `playback_finished` (and ~10 others) | ⬜ Not started |
| `src/web/server.rs` (test/diagnostic) | `transcript` | ⬜ Not started |

Historical `.jsonl` trace files and golden-trace fixtures produced before
`TypedRuntimeEvent` existed will **always** need the legacy fallback (or a dedicated
replay adapter) — that is the long-term home of this function.

## Deletion criteria for the legacy adapter

Delete `legacy_classify_runtime_kind_from_string` after **all** of the following are true:

- [ ] `src/cli/commands/live_half_duplex.rs` — every call to `trace.emit_now`,
      `trace.buffer_now`, `trace.emit`, and `trace.buffer` invokes `set_runtime_kind`
      on the event before emission.
- [ ] `src/web/server.rs` — the diagnostic `LiveTraceEvent::new("transcript", …)` call
      uses a typed kind.
- [ ] Any new live producer added to `src/` calls `set_runtime_kind`.
- [ ] Trace-replay code either (a) no longer relies on string-prefix inference, or
      (b) uses a clearly-named replay-only adapter instead of this function.

## Mechanical detection

`src/runtime_event.rs` exposes `pub(crate) fn event_used_legacy_classification(event: &LiveTraceEvent) -> bool`
for use in tests.  The test `all_known_live_producers_use_typed_runtime_kind` documents the
current (unfinished) migration state.  When a producer is migrated, update that test to
assert `!event_used_legacy_classification` for the corresponding kind strings.
