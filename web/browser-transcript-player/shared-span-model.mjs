export const SpanModality = Object.freeze({
  Audio: "Audio",
  Transcript: "Transcript",
  Word: "Word",
  Phoneme: "Phoneme",
  Morpheme: "Morpheme",
  Playback: "Playback",
  BreathGroup: "BreathGroup",
});

export const AlignmentKind = Object.freeze({
  Contains: "contains",
  AlignedTo: "aligned_to",
  DerivedFrom: "derived_from",
  RevisionOf: "revision_of",
  Overlaps: "overlaps",
  PlayedAs: "played_as",
});

export function createSpan({
  id,
  start_ms,
  end_ms,
  modality,
  metadata = null,
}) {
  if (!id || !Number.isFinite(start_ms) || !Number.isFinite(end_ms) || end_ms <= start_ms) {
    return null;
  }
  return { id, start_ms, end_ms, modality, metadata };
}

export function createAlignment({
  source,
  target,
  kind,
  confidence = null,
  metadata = null,
}) {
  if (!source || !target || !kind) {
    return null;
  }
  return { source, target, kind, confidence, metadata };
}

export function projectTimedWordsToSpans({
  words,
  idPrefix,
  modality = SpanModality.Word,
  stream = null,
  turn = null,
}) {
  const spans = [];
  for (let index = 0; index < (words ?? []).length; index++) {
    const word = words[index];
    const startMs = word?.timing?.start_ms;
    const endMs = word?.timing?.end_ms;
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
      continue;
    }
    const stableId = word?.span_id ?? word?.id ?? index;
    const span = createSpan({
      id: `${idPrefix}:${stableId}`,
      start_ms: startMs,
      end_ms: endMs,
      modality,
      metadata: {
        text: word?.text ?? null,
        word_id: word?.id ?? null,
        span_id: word?.span_id ?? null,
        stream,
        turn: turn ?? word?._turn ?? null,
      },
    });
    if (span) {
      spans.push(span);
    }
  }
  return spans;
}

export function alignWordSpansToParentSpan(wordSpans, parentSpan, kind, confidence = null) {
  const alignments = [];
  if (!parentSpan) {
    return alignments;
  }
  for (const wordSpan of wordSpans ?? []) {
    if (
      wordSpan.start_ms <= parentSpan.end_ms &&
      wordSpan.end_ms >= parentSpan.start_ms
    ) {
      const alignment = createAlignment({
        source: wordSpan.id,
        target: parentSpan.id,
        kind,
        confidence,
      });
      if (alignment) {
        alignments.push(alignment);
      }
    }
  }
  return alignments;
}

export function buildSharedSpanModel({ spans = [], alignments = [] }) {
  return { spans, alignments };
}
