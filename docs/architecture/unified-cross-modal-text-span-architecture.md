# Unified cross-modal text and span architecture

## Summary

Listenbury models speech, transcription, prosody, semantics, and memory as aligned modalities over one evolving text timeline (`TextId`), rather than isolated pipelines.

```text
TextId
 ├─ audio spans
 ├─ phoneme spans
 ├─ word spans
 ├─ clause / breath-group spans
 ├─ prosody spans
 ├─ semantic spans
 ├─ topic / episode spans
 └─ memory spans
```

Core implementation lives in `src/span.rs`.

## Core substrate

- `TextId`, `SpanId`, and `Cursor` identify a shared timeline.
- `Span<T>` carries:
  - owning `text_id`
  - `modality`
  - lifecycle `state`
  - `start`/`end` cursors
  - content payload and confidence/stability
  - append-only `revisions` history
- `AlignmentGraph` stores many-to-many cross-modal edges (`Alignment`) between spans.

## Span lifecycle and non-destructive revision

Lifecycle states are represented by `SpanState`:

```text
Hypothesis -> Stable -> Committed -> Revised
                                  \-> Deprecated
```

Revisions do not destructively replace prior values: `Span::revise` pushes a `SpanRevision<T>` snapshot before applying new contents/state.

## Cross-modal alignment under one `TextId`

Multiple modalities are expected to share one timeline ID and align through graph edges, e.g.:

- `Audio` contains `Word`
- `Phoneme` is equivalent to `Word`
- `Word` contained by `Clause`
- `Prosody` overlaps `Clause`
- `Semantic` derived from `Clause`
- `Memory` derived from `Topic`/`Episode`

## Demo coverage in tests

The acceptance flow is covered in `src/span.rs` tests:

- `demo_shows_provisional_to_committed_to_revised_span`
- `revisions_preserve_history`
- `aligns_multiple_modalities_under_one_text_id`

These show provisional/hypothesis spans becoming committed, then revised with preserved history, and aligned across modalities on the same `TextId`.

## Runtime trace/timeline compatibility adapter

`src/trace/viewer_payload.rs` now emits a `span_graph` alongside legacy `streams`, `events`, and `markers`.

- Legacy fields remain for existing WaveDeck/browser payload consumers.
- `span_graph.spans` materializes timed words and runtime event regions as canonical `Span<String>` values.
- `span_graph.alignments` carries cross-modal edges (for example, `playback` audio regions containing overlapping Pete word spans).

This keeps old UI payload contracts working while moving runtime timeline interpretation onto the shared span/alignment substrate.
