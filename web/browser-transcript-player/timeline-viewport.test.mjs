import test from "node:test";
import assert from "node:assert/strict";

import { createTimeScale, createTimelineViewport } from "./timeline-viewport.mjs";

test("word interval mapping preserves start and end through zoom", () => {
  const word = { startMs: 1250, endMs: 1725 };
  const scale = createTimeScale({ pxPerSecond: 160, durationMs: 5000 });

  const interval = scale.intervalToPx({
    startMs: word.startMs,
    endMs: word.endMs,
    minWidthPx: 2,
  });

  assert.equal(interval.leftPx, 200);
  assert.equal(interval.widthPx, 76);
  assert.equal(scale.pxToMs(interval.leftPx), word.startMs);
  assert.equal(scale.pxToMs(interval.leftPx + interval.widthPx), word.endMs);
});

test("timeline viewport keeps cursor-centered zoom anchor stable", () => {
  const viewport = createTimelineViewport({
    pxPerSecond: 200,
    durationMs: 10_000,
    viewportWidthPx: 500,
    scrollLeftPx: 300,
  });

  const anchorMs = viewport.scale.pxToMs(300 + 125);
  const zoomed = createTimelineViewport({
    pxPerSecond: 400,
    durationMs: 10_000,
    viewportWidthPx: 500,
  });

  const nextScroll = zoomed.scrollLeftForAnchor(anchorMs, 125);

  assert.equal(anchorMs, 2125);
  assert.equal(nextScroll, 725);
});
