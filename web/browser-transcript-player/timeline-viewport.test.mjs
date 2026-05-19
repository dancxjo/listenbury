import test from "node:test";
import assert from "node:assert/strict";

import {
  createTimeScale,
  createTimelineViewport,
  renderedCanvasXToTimelinePx,
} from "./timeline-viewport.mjs";

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

test("waveform peak lookup uses the same time scale as timeline intervals", () => {
  const scale = createTimeScale({ pxPerSecond: 100, durationMs: 10_000 });

  assert.equal(
    scale.waveformPeakIndexAtPx(scale.msToPx(2500), {
      audioDurationMs: 10_000,
      peakCount: 100,
    }),
    25,
  );
  assert.equal(
    scale.waveformPeakIndexAtPx(scale.msToPx(10_500), {
      audioDurationMs: 10_000,
      peakCount: 100,
    }),
    null,
  );
});

test("downsampled waveform canvas maps back to rendered timeline pixels", () => {
  assert.equal(
    renderedCanvasXToTimelinePx({
      canvasX: 6000,
      canvasWidthPx: 12_000,
      renderedWidthPx: 24_000,
    }),
    12_000,
  );
});
