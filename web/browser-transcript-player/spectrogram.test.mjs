import test from "node:test";
import assert from "node:assert/strict";

import { createTimeScale } from "./timeline-viewport.mjs";
import {
  analyzeSpectrogramSamples,
  appendSpectrogramSamples,
  selectSpectrogramLevel,
  spectrogramValueAt,
} from "./spectrogram.mjs";

function sineWave({ sampleRate, durationMs, frequencyHz }) {
  const sampleCount = Math.round((sampleRate * durationMs) / 1000);
  const samples = new Float32Array(sampleCount);
  for (let index = 0; index < sampleCount; index++) {
    samples[index] = Math.sin((2 * Math.PI * frequencyHz * index) / sampleRate);
  }
  return samples;
}

test("spectrogram hop bins align to the timeline time scale", () => {
  const sampleRate = 16_000;
  const spectrogram = analyzeSpectrogramSamples(sineWave({
    sampleRate,
    durationMs: 250,
    frequencyHz: 440,
  }), {
    sampleRate,
    levels: [{ id: "alignment", windowSize: 512, hopSize: 160, fftSize: 512 }],
  });
  const level = spectrogram.levels[0];
  const scale = createTimeScale({ pxPerSecond: 100, durationMs: 1000 });

  assert.equal(level.hopMs, 10);
  assert.equal(scale.msToPx(level.hopMs * 6), 6);
  assert.equal(scale.pxToMs(6), level.hopMs * 6);
  assert.equal(level.frameCount, 25);
});

test("incremental spectrogram append reuses existing frames and matches full recompute", () => {
  const sampleRate = 16_000;
  const initial = sineWave({ sampleRate, durationMs: 180, frequencyHz: 660 });
  const appended = sineWave({ sampleRate, durationMs: 360, frequencyHz: 660 });
  const levels = [{ id: "detail", windowSize: 512, hopSize: 80, fftSize: 512 }];

  const base = analyzeSpectrogramSamples(initial, { sampleRate, levels });
  const incrementallyAppended = appendSpectrogramSamples(base, appended, { sampleRate, levels });
  const full = analyzeSpectrogramSamples(appended, { sampleRate, levels });

  assert.equal(incrementallyAppended.analysisMode, "append");
  assert.ok(incrementallyAppended.levels[0].reusedFrameCount > 0);
  assert.ok(incrementallyAppended.levels[0].reusedFrameCount < base.levels[0].frameCount);
  assert.equal(incrementallyAppended.levels[0].frameCount, full.levels[0].frameCount);
  assert.deepEqual(
    incrementallyAppended.levels[0].frames.map((frame) => Array.from(frame.slice(0, 8))),
    full.levels[0].frames.map((frame) => Array.from(frame.slice(0, 8))),
  );
});

test("zoom-dependent level selection prefers the finer level as zoom increases", () => {
  const sampleRate = 16_000;
  const spectrogram = analyzeSpectrogramSamples(sineWave({
    sampleRate,
    durationMs: 250,
    frequencyHz: 330,
  }), { sampleRate });

  assert.equal(selectSpectrogramLevel(spectrogram, { pxPerSecond: 80 })?.id, "overview");
  assert.equal(selectSpectrogramLevel(spectrogram, { pxPerSecond: 400 })?.id, "detail");
  assert.equal(selectSpectrogramLevel(spectrogram, { pxPerSecond: 1600 })?.id, "fine");
  assert.ok(spectrogram.levels[0].hopMs > spectrogram.levels[1].hopMs);
  assert.ok(spectrogram.levels[1].hopMs > spectrogram.levels[2].hopMs);
});

test("hover inspection can resolve magnitude at time/frequency", () => {
  const sampleRate = 16_000;
  const spectrogram = analyzeSpectrogramSamples(sineWave({
    sampleRate,
    durationMs: 300,
    frequencyHz: 1000,
  }), {
    sampleRate,
    levels: [{ id: "inspect", windowSize: 512, hopSize: 160, fftSize: 512 }],
  });
  const level = spectrogram.levels[0];
  const magnitude = spectrogramValueAt(level, 100, 1000);

  assert.equal(typeof magnitude, "number");
  assert.ok(magnitude <= 0);
  assert.ok(magnitude >= -96);
});
