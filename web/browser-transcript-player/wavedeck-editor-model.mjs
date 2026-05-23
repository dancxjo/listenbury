const MIN_SPAN_DURATION_MS = 1;

export const WaveDeckSelectionType = Object.freeze({
  Item: "item",
  TimeRange: "time_range",
});

export const WaveDeckEditOrigin = Object.freeze({
  Automatic: "automatic",
  Manual: "manual",
  Derived: "derived",
});

export const WaveDeckEditKind = Object.freeze({
  AudioTrim: "audio.trim",
  AudioSplit: "audio.split",
  AudioFade: "audio.fade",
  AudioGain: "audio.gain",
  TranscriptReplaceWord: "transcript.replace_word",
  AlignmentMoveBoundary: "alignment.move_boundary",
});

export function createWaveDeckEditorState() {
  return {
    selectedItem: null,
    brushSelection: null,
    dragSelection: null,
  };
}

export function reduceWaveDeckEditorState(state, action) {
  const previous = state ?? createWaveDeckEditorState();
  if (!action || typeof action !== "object") {
    return previous;
  }

  switch (action.type) {
    case "clear_selection":
      return { ...previous, selectedItem: null, brushSelection: null };
    case "set_selected_item":
      return {
        ...previous,
        selectedItem: normalizeSelectedItem(action.selectedItem),
        brushSelection: null,
      };
    case "set_brush_selection":
      return {
        ...previous,
        brushSelection: normalizeTimeSelection(action.brushSelection),
      };
    case "set_drag_selection":
      return {
        ...previous,
        dragSelection: normalizeDragSelection(action.dragSelection),
      };
    case "set_drag_selection_end":
      if (!previous.dragSelection || !Number.isFinite(action.endMs)) {
        return previous;
      }
      return {
        ...previous,
        dragSelection: {
          ...previous.dragSelection,
          endMs: action.endMs,
        },
      };
    default:
      return previous;
  }
}

export function createWaveDeckEditSession({ spans = [] } = {}) {
  const bySpanId = {};
  for (const span of spans) {
    const normalizedSpan = normalizeSpan(span);
    if (!normalizedSpan) {
      continue;
    }
    bySpanId[normalizedSpan.id] = {
      original: { ...normalizedSpan },
      edited: { ...normalizedSpan },
      revisionLog: [],
    };
  }
  return {
    spansById: bySpanId,
    editLog: [],
    undoStack: [],
    redoStack: [],
  };
}

export function createEditProvenance({
  origin,
  actor,
  at_ms,
  source_span_ids = [],
  source_event_ids = [],
  reason = null,
} = {}) {
  const normalizedOrigin = normalizeOrigin(origin);
  if (!normalizedOrigin || !actor || !Number.isFinite(at_ms)) {
    return null;
  }
  return {
    origin: normalizedOrigin,
    actor: String(actor),
    at_ms,
    source_span_ids: normalizeStringArray(source_span_ids),
    source_event_ids: normalizeStringArray(source_event_ids),
    reason: normalizeOptionalString(reason),
  };
}

export function normalizeWaveDeckEditOp(op) {
  if (!op || typeof op !== "object") {
    return null;
  }
  const kind = String(op.kind ?? "");
  const provenance = createEditProvenance(op.provenance ?? {});
  if (!kind || !provenance) {
    return null;
  }
  switch (kind) {
    case WaveDeckEditKind.AlignmentMoveBoundary: {
      const span_id = normalizeOptionalString(op.span_id);
      const boundary = op.boundary === "start" || op.boundary === "end" ? op.boundary : null;
      const new_time_ms = Number(op.new_time_ms);
      if (!span_id || !boundary || !Number.isFinite(new_time_ms)) {
        return null;
      }
      return { kind, span_id, boundary, new_time_ms, provenance };
    }
    case WaveDeckEditKind.AudioTrim:
    case WaveDeckEditKind.AudioSplit:
    case WaveDeckEditKind.AudioFade:
    case WaveDeckEditKind.AudioGain:
    case WaveDeckEditKind.TranscriptReplaceWord:
      return {
        kind,
        ...op,
        provenance,
      };
    default:
      return null;
  }
}

export function applyWaveDeckEditOp(session, op) {
  const normalized = normalizeWaveDeckEditOp(op);
  if (!normalized) {
    return { session, applied: false };
  }

  const currentSession = session ?? createWaveDeckEditSession();
  if (normalized.kind !== WaveDeckEditKind.AlignmentMoveBoundary) {
    const next = {
      ...currentSession,
      editLog: [...currentSession.editLog, normalized],
      undoStack: [...currentSession.undoStack, { op: normalized, undo: null }],
      redoStack: [],
    };
    return { session: next, applied: true };
  }

  const entry = currentSession.spansById[normalized.span_id];
  if (!entry) {
    return { session: currentSession, applied: false };
  }
  const previousSpan = entry.edited;
  const nextSpan = moveBoundary(previousSpan, normalized.boundary, normalized.new_time_ms);
  if (!nextSpan) {
    return { session: currentSession, applied: false };
  }
  const revision = {
    op: normalized,
    previous_span: previousSpan,
    next_span: nextSpan,
  };
  const next = {
    ...currentSession,
    spansById: {
      ...currentSession.spansById,
      [normalized.span_id]: {
        ...entry,
        edited: nextSpan,
        revisionLog: [...entry.revisionLog, revision],
      },
    },
    editLog: [...currentSession.editLog, normalized],
    undoStack: [...currentSession.undoStack, { op: normalized, undo: previousSpan }],
    redoStack: [],
  };
  return { session: next, applied: true };
}

export function undoWaveDeckEdit(session) {
  const current = session ?? createWaveDeckEditSession();
  const undoEntry = current.undoStack.at(-1);
  if (!undoEntry) {
    return { session: current, applied: false };
  }
  const nextUndoStack = current.undoStack.slice(0, -1);
  if (!undoEntry.undo || undoEntry.op.kind !== WaveDeckEditKind.AlignmentMoveBoundary) {
    const next = {
      ...current,
      undoStack: nextUndoStack,
      redoStack: [...current.redoStack, undoEntry],
    };
    return { session: next, applied: true };
  }

  const spanId = undoEntry.op.span_id;
  const entry = current.spansById[spanId];
  if (!entry) {
    return { session: current, applied: false };
  }
  const revertedSpan = undoEntry.undo;
  const next = {
    ...current,
    spansById: {
      ...current.spansById,
      [spanId]: {
        ...entry,
        edited: revertedSpan,
      },
    },
    undoStack: nextUndoStack,
    redoStack: [...current.redoStack, undoEntry],
  };
  return { session: next, applied: true };
}

export function serializeWaveDeckEditLog(session) {
  return JSON.stringify(session?.editLog ?? []);
}

export function deserializeWaveDeckEditLog(serialized) {
  const parsed = JSON.parse(serialized);
  if (!Array.isArray(parsed)) {
    return [];
  }
  return parsed
    .map((entry) => normalizeWaveDeckEditOp(entry))
    .filter(Boolean);
}

export function replayWaveDeckEditLog(baseSession, editLog) {
  let current = baseSession ?? createWaveDeckEditSession();
  for (const op of editLog ?? []) {
    const result = applyWaveDeckEditOp(current, op);
    if (result.applied) {
      current = result.session;
    }
  }
  return current;
}

function normalizeSelectedItem(selectedItem) {
  if (!selectedItem || typeof selectedItem !== "object") {
    return null;
  }
  const type = normalizeOptionalString(selectedItem.type);
  const laneIndex = Number(selectedItem.laneIndex);
  const itemIndex = Number(selectedItem.itemIndex);
  if (!type || !Number.isFinite(laneIndex) || !Number.isFinite(itemIndex)) {
    return null;
  }
  return {
    type,
    laneIndex,
    itemIndex,
  };
}

function normalizeTimeSelection(selection) {
  if (!selection || typeof selection !== "object") {
    return null;
  }
  const startMs = Number(selection.startMs);
  const endMs = Number(selection.endMs);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
    return null;
  }
  return {
    type: WaveDeckSelectionType.TimeRange,
    startMs,
    endMs,
  };
}

function normalizeDragSelection(selection) {
  if (!selection || typeof selection !== "object") {
    return null;
  }
  const pointerId = Number(selection.pointerId);
  const startClientX = Number(selection.startClientX);
  const startMs = Number(selection.startMs);
  const endMs = Number(selection.endMs);
  if (
    !Number.isFinite(pointerId) ||
    !Number.isFinite(startClientX) ||
    !Number.isFinite(startMs) ||
    !Number.isFinite(endMs)
  ) {
    return null;
  }
  return {
    pointerId,
    surface: selection.surface ?? null,
    startClientX,
    startMs,
    endMs,
  };
}

function normalizeOrigin(origin) {
  switch (origin) {
    case WaveDeckEditOrigin.Automatic:
    case WaveDeckEditOrigin.Manual:
    case WaveDeckEditOrigin.Derived:
      return origin;
    default:
      return null;
  }
}

function normalizeStringArray(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  return values
    .map((value) => normalizeOptionalString(value))
    .filter(Boolean);
}

function normalizeOptionalString(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const normalized = String(value).trim();
  return normalized.length > 0 ? normalized : null;
}

function normalizeSpan(span) {
  if (!span || typeof span !== "object") {
    return null;
  }
  const id = normalizeOptionalString(span.id);
  const start_ms = Number(span.start_ms);
  const end_ms = Number(span.end_ms);
  if (!id || !Number.isFinite(start_ms) || !Number.isFinite(end_ms) || end_ms <= start_ms) {
    return null;
  }
  return {
    ...span,
    id,
    start_ms,
    end_ms,
  };
}

function moveBoundary(span, boundary, nextTimeMs) {
  const start = Number(span?.start_ms);
  const end = Number(span?.end_ms);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return null;
  }
  if (boundary === "start") {
    const start_ms = Math.max(0, Math.min(nextTimeMs, end - MIN_SPAN_DURATION_MS));
    return {
      ...span,
      start_ms,
    };
  }
  if (boundary === "end") {
    const end_ms = Math.max(start + MIN_SPAN_DURATION_MS, nextTimeMs);
    return {
      ...span,
      end_ms,
    };
  }
  return null;
}
