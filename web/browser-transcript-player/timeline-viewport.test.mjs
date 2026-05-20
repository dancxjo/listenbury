import test from "node:test";
import assert from "node:assert/strict";

import {
  buildWaveformWordDeckRowLayout,
  buildWaveformResolutionLevels,
  createTimeScale,
  createTimelineViewport,
  renderedCanvasXToTimelinePx,
  selectWaveformResolutionLevel,
  waveformBucketToPx,
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

test("waveform resolution selection steps from overview to fine detail across zoom levels", () => {
  const levels = [
    { bucketDurationMs: 40, label: "40ms buckets", buckets: [] },
    { bucketDurationMs: 10, label: "10ms buckets", buckets: [] },
    { bucketDurationMs: 2, label: "2ms buckets", buckets: [] },
  ];

  assert.equal(selectWaveformResolutionLevel(levels, { pxPerSecond: 50 }), levels[0]);
  assert.equal(selectWaveformResolutionLevel(levels, { pxPerSecond: 200 }), levels[1]);
  assert.equal(selectWaveformResolutionLevel(levels, { pxPerSecond: 1000 }), levels[2]);
});

test("waveform bucket time ranges map to timeline pixels with the shared time scale", () => {
  const scale = createTimeScale({ pxPerSecond: 100, durationMs: 10_000 });
  const interval = waveformBucketToPx(scale, {
    start_ms: 250,
    end_ms: 500,
  });

  assert.equal(interval.leftPx, 25);
  assert.equal(interval.widthPx, 25);
});

test("waveform word deck row layout places phoneme rows beneath word rows", () => {
  const layout = buildWaveformWordDeckRowLayout({
    rowCount: 2,
    wordRowHeightPx: 20,
    phonemeRowHeightPx: 16,
    wordToPhonemeGapPx: 2,
    rowGapPx: 4,
    marginPx: 4,
  });

  assert.deepEqual(layout.rows, [
    { wordTopPx: 4, phonemeTopPx: 26 },
    { wordTopPx: 46, phonemeTopPx: 68 },
  ]);
  assert.equal(layout.heightPx, 88);
});

test("waveform multi-resolution levels preserve timing, peak, and energy fields", () => {
  const samples = new Float32Array([0, 0.5, -0.25, 0.75, -1, 0.25, 0.5, -0.5]);
  const audioBuffer = {
    sampleRate: 1000,
    length: samples.length,
    duration: samples.length / 1000,
    numberOfChannels: 1,
    getChannelData(channel) {
      assert.equal(channel, 0);
      return samples;
    },
  };

  const levels = buildWaveformResolutionLevels(audioBuffer, {
    targetBucketCounts: [2, 4, 8],
  });

  assert.equal(levels.length, 3);
  assert.equal(levels[0].bucketDurationMs, 4);
  assert.equal(levels[1].bucketDurationMs, 2);
  assert.equal(levels[2].bucketDurationMs, 1);
  assert.deepEqual(levels[1].buckets[0], {
    start_ms: 0,
    end_ms: 2,
    min_sample: 0,
    max_sample: 0.5,
    rms_energy: Math.sqrt((0 ** 2 + 0.5 ** 2) / 2),
    sample_count: 2,
  });
});
