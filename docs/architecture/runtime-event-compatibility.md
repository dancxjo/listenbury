# Runtime Event Classification Compatibility

`RuntimeEventKind` should be produced from typed runtime domain decisions at event producers.

## Preferred path (current)

- Producers set a typed `RuntimeEventKind` on `LiveTraceEvent` (for example via `LiveTraceEvent::set_runtime_kind`).
- `RuntimeEvent::from_live_trace_event` preserves that typed kind.

## Legacy compatibility path

- If a `LiveTraceEvent` has no typed runtime kind, `RuntimeEvent::from_live_trace_event` falls back to
  `legacy_classify_runtime_kind_from_string`.
- This keeps historical trace replay compatible with old `kind` string naming conventions.

## Deletion criteria for the legacy adapter

Delete `legacy_classify_runtime_kind_from_string` after:

1. all live runtime event producers set typed `RuntimeEventKind`, and
2. trace import/replay no longer depends on inferring domains from legacy string prefixes.
