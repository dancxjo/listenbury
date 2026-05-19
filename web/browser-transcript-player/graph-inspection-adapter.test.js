const test = require("node:test");
const assert = require("node:assert/strict");

function rangesOverlap(leftStart, leftEnd, rightStart, rightEnd) {
  return Math.max(leftStart, rightStart) <= Math.min(leftEnd, rightEnd);
}

function pushGraphNode(nodes, ids, node) {
  if (!ids.has(node.id)) {
    ids.add(node.id);
    nodes.push(node);
  }
}

function pushGraphEdge(edges, ids, edge) {
  if (!ids.has(edge.id)) {
    ids.add(edge.id);
    edges.push(edge);
  }
}

function addGraphWord(nodes, edges, nodeIds, edgeIds, item) {
  const wordNodeId = `word:${item.laneIndex}:${item.wordIndex}`;
  pushGraphNode(nodes, nodeIds, { id: wordNodeId, nodeType: "word", turn: item.turn });
  const revisions = item.word._revisions ?? [];
  let revisionSourceId = null;
  revisions.forEach((revision, index) => {
    const revisionNodeId = `revision:${item.laneIndex}:${item.wordIndex}:${index}`;
    pushGraphNode(nodes, nodeIds, { id: revisionNodeId, nodeType: "revision", label: revision.fromText });
    if (revisionSourceId) {
      pushGraphEdge(edges, edgeIds, {
        id: `edge:${revisionSourceId}:${revisionNodeId}:revision`,
        source: revisionSourceId,
        target: revisionNodeId,
        edgeType: "revision",
      });
    }
    revisionSourceId = revisionNodeId;
  });
  if (revisionSourceId) {
    pushGraphEdge(edges, edgeIds, {
      id: `edge:${revisionSourceId}:${wordNodeId}:revision`,
      source: revisionSourceId,
      target: wordNodeId,
      edgeType: "revision",
    });
  }
}

function addOverlapEdges(edges, edgeIds, words, events, nodeIds) {
  for (const word of words) {
    const wordNodeId = `word:${word.laneIndex}:${word.wordIndex}`;
    if (!nodeIds.has(wordNodeId)) continue;
    for (const event of events) {
      const eventNodeId = `event:${event.laneIndex}:${event.eventIndex}`;
      if (!nodeIds.has(eventNodeId)) continue;
      if (word.turn !== null && event.turn !== null && word.turn !== event.turn) continue;
      if (
        rangesOverlap(
          word.word.resolvedTiming.start_ms,
          word.word.resolvedTiming.end_ms,
          event.event.start_ms,
          event.event.end_ms,
        )
      ) {
        pushGraphEdge(edges, edgeIds, {
          id: `edge:${wordNodeId}:${eventNodeId}:overlap`,
          source: wordNodeId,
          target: eventNodeId,
          edgeType: "alignment",
        });
      }
    }
  }
}

test("revision lineage builds chain from prior text nodes to final word", () => {
  const nodes = [];
  const edges = [];
  const nodeIds = new Set();
  const edgeIds = new Set();
  addGraphWord(nodes, edges, nodeIds, edgeIds, {
    laneIndex: 0,
    wordIndex: 1,
    turn: 1,
    word: {
      text: "what",
      _revisions: [{ fromText: "that", at_ms: 1000 }, { fromText: "wat", at_ms: 1200 }],
    },
  });

  assert.equal(nodes.filter((entry) => entry.nodeType === "revision").length, 2);
  assert(edges.some((edge) => edge.id === "edge:revision:0:1:1:word:0:1:revision"));
});

test("revisions-only filter keeps only revised words", () => {
  const words = [
    { word: { _revisions: [{ fromText: "that" }] } },
    { word: { _revisions: [] } },
    { word: {} },
  ];
  const filtered = words.filter((item) => item.word._revisions?.length > 0);
  assert.equal(filtered.length, 1);
});

test("overlap edges only connect intersecting spans within the same turn", () => {
  const nodes = [];
  const edges = [];
  const nodeIds = new Set();
  const edgeIds = new Set();
  pushGraphNode(nodes, nodeIds, { id: "word:0:0", nodeType: "word" });
  pushGraphNode(nodes, nodeIds, { id: "event:1:0", nodeType: "event" });
  pushGraphNode(nodes, nodeIds, { id: "event:1:1", nodeType: "event" });

  addOverlapEdges(
    edges,
    edgeIds,
    [{
      laneIndex: 0,
      wordIndex: 0,
      turn: 1,
      word: { resolvedTiming: { start_ms: 100, end_ms: 260 } },
    }],
    [
      { laneIndex: 1, eventIndex: 0, turn: 1, event: { start_ms: 200, end_ms: 300 } },
      { laneIndex: 1, eventIndex: 1, turn: 2, event: { start_ms: 210, end_ms: 240 } },
    ],
    nodeIds,
  );

  assert.equal(edges.length, 1);
  assert.equal(edges[0].id, "edge:word:0:0:event:1:0:overlap");
});
