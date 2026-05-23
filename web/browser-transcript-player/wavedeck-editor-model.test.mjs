import test from "node:test";
import assert from "node:assert/strict";

import {
  WaveDeckEditKind,
  WaveDeckEditOrigin,
  createEditProvenance,
  createWaveDeckEditSession,
  createWaveDeckEditorState,
  reduceWaveDeckEditorState,
  applyWaveDeckEditOp,
  undoWaveDeckEdit,
  serializeWaveDeckEditLog,
  deserializeWaveDeckEditLog,
  replayWaveDeckEditLog,
} from "./wavedeck-editor-model.mjs";

test("editor state centralizes item and time-range selection", () => {
  let state = createWaveDeckEditorState();
  state = reduceWaveDeckEditorState(state, {
    type: "set_selected_item",
    selectedItem: { type: "word", laneIndex: 1, itemIndex: 3 },
  });
  assert.deepEqual(state.selectedItem, { type: "word", laneIndex: 1, itemIndex: 3 });
  assert.equal(state.brushSelection, null);

  state = reduceWaveDeckEditorState(state, {
    type: "set_brush_selection",
    brushSelection: { startMs: 250, endMs: 780 },
  });
  assert.equal(state.brushSelection.startMs, 250);
  assert.equal(state.brushSelection.endMs, 780);

  state = reduceWaveDeckEditorState(state, { type: "clear_selection" });
  assert.equal(state.selectedItem, null);
  assert.equal(state.brushSelection, null);
});

test("alignment boundary edit is non-destructive and keeps provenance", () => {
  const session = createWaveDeckEditSession({
    spans: [
      {
        id: "word:1",
        start_ms: 100,
        end_ms: 240,
        modality: "Word",
        metadata: { text: "hello" },
      },
    ],
  });
  const provenance = createEditProvenance({
    origin: WaveDeckEditOrigin.Manual,
    actor: "editor-user",
    at_ms: 5_000,
    source_span_ids: ["word:1"],
    reason: "nudge start to plosive burst",
  });
  const op = {
    kind: WaveDeckEditKind.AlignmentMoveBoundary,
    span_id: "word:1",
    boundary: "start",
    new_time_ms: 120,
    provenance,
  };
  const result = applyWaveDeckEditOp(session, op);
  assert.equal(result.applied, true);
  assert.equal(result.session.spansById["word:1"].original.start_ms, 100);
  assert.equal(result.session.spansById["word:1"].edited.start_ms, 120);
  assert.equal(result.session.editLog.length, 1);
  assert.equal(result.session.editLog[0].provenance.origin, WaveDeckEditOrigin.Manual);
});

test("edit logs serialize and replay into the same edited span state", () => {
  const base = createWaveDeckEditSession({
    spans: [{ id: "word:7", start_ms: 20, end_ms: 90, modality: "Word" }],
  });
  const provenance = createEditProvenance({
    origin: WaveDeckEditOrigin.Automatic,
    actor: "energy-snap",
    at_ms: 42,
    source_span_ids: ["word:7"],
    source_event_ids: ["event:2"],
  });
  const op = {
    kind: WaveDeckEditKind.AlignmentMoveBoundary,
    span_id: "word:7",
    boundary: "end",
    new_time_ms: 108,
    provenance,
  };
  const applied = applyWaveDeckEditOp(base, op).session;

  const serialized = serializeWaveDeckEditLog(applied);
  const restoredLog = deserializeWaveDeckEditLog(serialized);
  const replayed = replayWaveDeckEditLog(base, restoredLog);

  assert.equal(replayed.spansById["word:7"].edited.end_ms, 108);
  assert.deepEqual(replayed.editLog, applied.editLog);
});

test("undo restores the previous edited boundary", () => {
  const base = createWaveDeckEditSession({
    spans: [{ id: "word:2", start_ms: 10, end_ms: 40, modality: "Word" }],
  });
  const provenance = createEditProvenance({
    origin: WaveDeckEditOrigin.Manual,
    actor: "editor-user",
    at_ms: 11,
  });
  const changed = applyWaveDeckEditOp(base, {
    kind: WaveDeckEditKind.AlignmentMoveBoundary,
    span_id: "word:2",
    boundary: "end",
    new_time_ms: 55,
    provenance,
  }).session;

  const undone = undoWaveDeckEdit(changed);
  assert.equal(undone.applied, true);
  assert.equal(undone.session.spansById["word:2"].edited.end_ms, 40);
});
