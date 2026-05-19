import test from "node:test";
import assert from "node:assert/strict";

import {
  buildEnergyEnvelopeFromAudioBuffer,
  calculateReverseWordBreaks,
  detectEnergyLandmarks,
  refineWordTimingsWithEnergy,
} from "./energy-timing.mjs";

function fakeAudioBuffer(samples, sampleRate = 1000) {
  const data = Float32Array.from(samples);
  return {
    numberOfChannels: 1,
    length: data.length,
    sampleRate,
    duration: data.length / sampleRate,
    getChannelData() {
      return data;
    },
  };
}

test("buildEnergyEnvelopeFromAudioBuffer generates expected frame structure and non-negative energy metrics", () => {
  const buffer = fakeAudioBuffer([0, 1, 1, 0, 0, 0, 0, 0], 1000);
  const envelope = buildEnergyEnvelopeFromAudioBuffer(buffer, { windowMs: 4, hopMs: 2 });
  assert.equal(envelope.window_ms, 4);
  assert.equal(envelope.hop_ms, 2);
  assert.ok(envelope.frames.length >= 3);
  assert.ok(envelope.frames[0].rms_energy >= 0);
  assert.ok(envelope.frames[0].peak_energy >= 0);
});

test("refinement snaps boundary into a clear silence gap", () => {
  const samples = new Array(1000).fill(0.9);
  for (let i = 430; i <= 560; i++) samples[i] = 0;
  const envelope = buildEnergyEnvelopeFromAudioBuffer(fakeAudioBuffer(samples), { windowMs: 20, hopMs: 10 });
  const landmarks = detectEnergyLandmarks(envelope);
  const refined = refineWordTimingsWithEnergy(
    [
      { text: "hello", timing: { start_ms: 100, end_ms: 490 } },
      { text: "world", timing: { start_ms: 490, end_ms: 840 } },
    ],
    envelope,
    landmarks,
  );
  assert.equal(refined.length, 2);
  assert.equal(refined[0].timingResolution, "energy.snapped");
  assert.ok(refined[1].resolvedTiming.start_ms >= refined[0].resolvedTiming.end_ms);
  assert.ok(refined[0].resolvedTiming.end_ms >= 430 && refined[0].resolvedTiming.end_ms <= 560);
});

test("reverse word-break layer calculates internal boundaries from the tail", () => {
  const breaks = calculateReverseWordBreaks([
    { text: "small", timing: { start_ms: 100, end_ms: 350 } },
    { text: "word", timing: { start_ms: 350, end_ms: 650 } },
    { text: "largest", timing: { start_ms: 650, end_ms: 1200 } },
  ]);

  assert.equal(breaks.length, 2);
  assert.deepEqual(
    breaks.map((boundary) => [boundary.leftIndex, boundary.rightIndex]),
    [
      [0, 1],
      [1, 2],
    ],
  );
  assert.equal(breaks[1].ms, 719);
  assert.equal(breaks[0].ms, 444);
  assert.equal(breaks[0].method, "reverse-text-breaks");
});

test("adjacent words with no clean gap do not snap aggressively", () => {
  const samples = new Array(900).fill(0.6);
  const envelope = buildEnergyEnvelopeFromAudioBuffer(fakeAudioBuffer(samples), { windowMs: 20, hopMs: 10 });
  const landmarks = detectEnergyLandmarks(envelope);
  const refined = refineWordTimingsWithEnergy(
    [
      { text: "a", timing: { start_ms: 100, end_ms: 210 } },
      { text: "b", timing: { start_ms: 210, end_ms: 320 } },
    ],
    envelope,
    landmarks,
  );
  assert.equal(refined[0].timingResolution, "word.timing");
  assert.equal(refined[1].timingResolution, "word.timing");
});

test("noisy low-energy region falls back to whisper timing", () => {
  const samples = new Array(1200).fill(0).map((_, idx) => ((idx % 11) / 200) + 0.04);
  const envelope = buildEnergyEnvelopeFromAudioBuffer(fakeAudioBuffer(samples), { windowMs: 20, hopMs: 10 });
  const landmarks = detectEnergyLandmarks(envelope);
  const refined = refineWordTimingsWithEnergy(
    [{ text: "noise", timing: { start_ms: 300, end_ms: 520 } }],
    envelope,
    landmarks,
  );
  assert.equal(refined[0].timingResolution, "word.timing");
  assert.deepEqual(refined[0].resolvedTiming, refined[0].whisperTiming);
});

test("very short words keep positive, ordered durations after refinement", () => {
  const samples = new Array(700).fill(0.75);
  for (let i = 210; i <= 250; i++) samples[i] = 0;
  const envelope = buildEnergyEnvelopeFromAudioBuffer(fakeAudioBuffer(samples), { windowMs: 10, hopMs: 5 });
  const landmarks = detectEnergyLandmarks(envelope);
  const refined = refineWordTimingsWithEnergy(
    [
      { text: "a", timing: { start_ms: 180, end_ms: 225 } },
      { text: "to", timing: { start_ms: 225, end_ms: 290 } },
    ],
    envelope,
    landmarks,
    { minWordDurationMs: 10 },
  );
  assert.ok(refined[0].resolvedTiming.end_ms > refined[0].resolvedTiming.start_ms);
  assert.ok(refined[1].resolvedTiming.end_ms > refined[1].resolvedTiming.start_ms);
  assert.ok(refined[0].resolvedTiming.end_ms <= refined[1].resolvedTiming.start_ms);
});
