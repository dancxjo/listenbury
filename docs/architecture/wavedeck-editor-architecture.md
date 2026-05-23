# WaveDeck editor-grade timeline architecture

## Goal

WaveDeck is evolving from an observability timeline into Listenbury’s speech-aware editor workbench:

```text
inspect -> correct -> align -> annotate -> edit -> replay -> export
```

The editor layer sits on top of existing timeline/span/alignment/replay primitives and keeps the observability workflow intact.

## Editing substrate

`web/browser-transcript-player/wavedeck-editor-model.mjs` defines a non-destructive operation model with provenance:

- `WaveDeckEditKind.Audio*` operations for clip-region edits (`trim`, `split`, `fade`, `gain`)
- `WaveDeckEditKind.TranscriptReplaceWord` for transcript corrections
- `WaveDeckEditKind.AlignmentMoveBoundary` for timing boundary correction

Each operation carries provenance (`origin`, `actor`, `at_ms`, source IDs, optional reason), preserving automatic/manual/derived lineage.

## Original vs edited evidence

Edit sessions keep both immutable source evidence and mutable edited state:

- `spansById[spanId].original` (captured evidence)
- `spansById[spanId].edited` (current edited view)
- `spansById[spanId].revisionLog` (applied non-destructive revisions)

This maintains clear separation between observed artifacts and corrected artifacts.

## Replayable persistence

The edit layer serializes as an operation log:

- `serializeWaveDeckEditLog(session)`
- `deserializeWaveDeckEditLog(json)`
- `replayWaveDeckEditLog(baseSession, editLog)`

Edits are persisted/replayed by applying ops to baseline spans instead of rewriting source audio.

## Centralized selection/edit state

WaveDeck selection state is centralized through `reduceWaveDeckEditorState`, covering:

- item selection (`word`/`event`)
- timeline brush selection
- drag-selection lifecycle

This keeps selection/edit intent in one shared state path instead of per-lane ad hoc state.

## Current implementation slice

Initial concrete operation support is implemented for:

- `AlignmentMoveBoundary` application
- undo support for applied boundary moves
- JSON log replay of boundary edits

This is the Milestone A/B spine and is intentionally scoped for incremental extension.
