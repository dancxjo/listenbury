const DEFAULT_ENVELOPE_CONFIG = {
  windowMs: 20,
  hopMs: 10,
  dbFloor: -120,
};

const DEFAULT_LANDMARK_CONFIG = {
  silenceFloorRatio: 0.08,
  silenceNoiseMultiplier: 1.35,
  onsetRiseRatio: 0.07,
  offsetFallRatio: 0.07,
};

const DEFAULT_REFINEMENT_CONFIG = {
  startToleranceMs: 120,
  endToleranceMs: 160,
  minWordDurationMs: 20,
  minSnapConfidence: 0.58,
};
const MIN_RESOLVED_DURATION_MS = 1;

export function buildEnergyEnvelopeFromAudioBuffer(audioBuffer, options = {}) {
  if (!audioBuffer || typeof audioBuffer.length !== "number" || typeof audioBuffer.sampleRate !== "number") {
    return { window_ms: DEFAULT_ENVELOPE_CONFIG.windowMs, hop_ms: DEFAULT_ENVELOPE_CONFIG.hopMs, frames: [] };
  }

  const config = { ...DEFAULT_ENVELOPE_CONFIG, ...options };
  const sampleRate = Math.max(1, Math.round(audioBuffer.sampleRate));
  const sampleCount = Math.max(0, Math.round(audioBuffer.length));
  const channelCount = Math.max(1, Math.round(audioBuffer.numberOfChannels ?? 1));
  const windowSamples = Math.max(1, Math.round((sampleRate * config.windowMs) / 1000));
  const hopSamples = Math.max(1, Math.round((sampleRate * config.hopMs) / 1000));
  const channels = [];
  for (let channel = 0; channel < channelCount; channel++) {
    channels.push(audioBuffer.getChannelData(channel));
  }

  const frames = [];
  for (let frameStart = 0; frameStart < sampleCount; frameStart += hopSamples) {
    const frameEnd = Math.min(sampleCount, frameStart + windowSamples);
    if (frameEnd <= frameStart) {
      continue;
    }
    let rmsSquaredSum = 0;
    let peak = 0;
    let count = 0;

    for (let index = frameStart; index < frameEnd; index++) {
      let mono = 0;
      for (let channel = 0; channel < channelCount; channel++) {
        mono += channels[channel]?.[index] ?? 0;
      }
      mono /= channelCount;
      const abs = Math.abs(mono);
      peak = Math.max(peak, abs);
      rmsSquaredSum += mono * mono;
      count += 1;
    }

    const rms = count > 0 ? Math.sqrt(rmsSquaredSum / count) : 0;
    const dbfs = rms > 0 ? 20 * Math.log10(rms) : config.dbFloor;
    frames.push({
      frame_start_ms: Math.round((frameStart * 1000) / sampleRate),
      frame_end_ms: Math.max(
        Math.round(((frameStart + 1) * 1000) / sampleRate),
        Math.round((frameEnd * 1000) / sampleRate),
      ),
      rms_energy: rms,
      peak_energy: peak,
      dbfs,
    });
  }

  return {
    window_ms: config.windowMs,
    hop_ms: config.hopMs,
    frames,
  };
}

export function detectEnergyLandmarks(envelope, options = {}) {
  const frames = Array.isArray(envelope?.frames) ? envelope.frames : [];
  if (!frames.length) {
    return { onsets: [], offsets: [], valleys: [], silences: [], peaks: [] };
  }

  const config = { ...DEFAULT_LANDMARK_CONFIG, ...options };
  const energies = frames.map((frame) => Number(frame.rms_energy) || 0);
  const maxEnergy = Math.max(...energies, 0);
  const sorted = [...energies].sort((left, right) => left - right);
  const noiseFloor = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
  const silenceThreshold = Math.min(
    maxEnergy * 0.45,
    Math.max(maxEnergy * config.silenceFloorRatio, noiseFloor * config.silenceNoiseMultiplier),
  );
  const onsetRiseThreshold = maxEnergy * config.onsetRiseRatio;
  const offsetFallThreshold = maxEnergy * config.offsetFallRatio;

  const onsets = [];
  const offsets = [];
  const valleys = [];
  const peaks = [];
  const silences = [];

  let silenceStart = null;
  for (let i = 0; i < frames.length; i++) {
    const current = energies[i];
    const previous = i > 0 ? energies[i - 1] : current;
    const next = i + 1 < frames.length ? energies[i + 1] : current;
    const centerMs = Math.round((frames[i].frame_start_ms + frames[i].frame_end_ms) / 2);

    if (current <= silenceThreshold) {
      if (silenceStart === null) {
        silenceStart = frames[i].frame_start_ms;
      }
    } else if (silenceStart !== null) {
      silences.push({
        start_ms: silenceStart,
        end_ms: frames[i].frame_start_ms,
      });
      silenceStart = null;
    }

    if (i === 0 || i === frames.length - 1) {
      continue;
    }

    if (current >= previous && current >= next && current >= Math.max(silenceThreshold * 1.5, noiseFloor * 1.8)) {
      peaks.push(centerMs);
    }
    if (current <= previous && current <= next && current <= (previous + next) * 0.55) {
      valleys.push(centerMs);
    }
    if (current - previous >= onsetRiseThreshold && current > silenceThreshold && previous <= silenceThreshold * 1.25) {
      onsets.push(frames[i].frame_start_ms);
    }
    if (previous - current >= offsetFallThreshold && previous > silenceThreshold && current <= silenceThreshold * 1.25) {
      offsets.push(frames[i].frame_start_ms);
    }
  }
  if (silenceStart !== null) {
    silences.push({
      start_ms: silenceStart,
      end_ms: frames[frames.length - 1].frame_end_ms,
    });
  }

  return { onsets, offsets, valleys, silences, peaks };
}

export function refineWordTimingsWithEnergy(words, envelope, landmarks, options = {}) {
  const config = { ...DEFAULT_REFINEMENT_CONFIG, ...options };
  const frames = Array.isArray(envelope?.frames) ? envelope.frames : [];
  if (!Array.isArray(words) || !words.length) {
    return [];
  }

  const timingWords = words.map((word) => {
    const start = Number(word?.timing?.start_ms);
    const end = Number(word?.timing?.end_ms);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
      return {
        whisperTiming: null,
        energyTiming: null,
        resolvedTiming: null,
        timingResolution: "fallback-layout",
      };
    }
    return {
      whisperTiming: {
        start_ms: Math.max(0, Math.round(start)),
        end_ms: Math.max(Math.round(start), Math.round(end)),
      },
      energyTiming: null,
      resolvedTiming: {
        start_ms: Math.max(0, Math.round(start)),
        end_ms: Math.max(Math.round(start), Math.round(end)),
      },
      timingResolution: "word.timing",
      energySnapConfidence: null,
    };
  });

  if (!frames.length) {
    return timingWords;
  }

  const lastFrameEnd = frames[frames.length - 1]?.frame_end_ms ?? 0;
  const candidatePoints = collectCandidatePoints(landmarks, lastFrameEnd);

  for (let i = 0; i < timingWords.length; i++) {
    const slot = timingWords[i];
    if (!slot.whisperTiming) {
      continue;
    }
    const previous = findPreviousTimedWord(timingWords, i);
    const next = findNextTimedWord(timingWords, i);
    const whisperStart = slot.whisperTiming.start_ms;
    const whisperEnd = slot.whisperTiming.end_ms;

    const minStart = previous ? previous.resolvedTiming.end_ms : 0;
    const maxStart = Math.max(minStart, whisperEnd - config.minWordDurationMs);
    const startCandidate = findBoundaryCandidate({
      centerMs: whisperStart,
      toleranceMs: config.startToleranceMs,
      minMs: minStart,
      maxMs: maxStart,
      points: candidatePoints,
      preferredTypes: ["onset", "valley", "silence-start", "silence-end"],
    });

    const minEndBase = startCandidate?.ms ?? whisperStart;
    const minEnd = minEndBase + config.minWordDurationMs;
    const maxEnd = next ? next.whisperTiming.start_ms : Math.max(whisperEnd, lastFrameEnd);
    const endCandidate = findBoundaryCandidate({
      centerMs: whisperEnd,
      toleranceMs: config.endToleranceMs,
      minMs: minEnd,
      maxMs: maxEnd,
      points: candidatePoints,
      preferredTypes: ["offset", "valley", "silence-end", "silence-start"],
    });

    const snappedStart = clampBoundary(startCandidate?.ms ?? whisperStart, minStart, maxStart);
    const snappedEnd = clampBoundary(
      endCandidate?.ms ?? whisperEnd,
      snappedStart + config.minWordDurationMs,
      Math.max(snappedStart + config.minWordDurationMs, maxEnd),
    );
    const confidence = combineConfidence(startCandidate, endCandidate);

    if (
      Number.isFinite(snappedStart) &&
      Number.isFinite(snappedEnd) &&
      snappedEnd > snappedStart &&
      confidence >= config.minSnapConfidence &&
      (snappedStart !== whisperStart || snappedEnd !== whisperEnd)
    ) {
      slot.energyTiming = {
        start_ms: snappedStart,
        end_ms: snappedEnd,
        method: "rms-valley-snap",
        confidence,
      };
      slot.resolvedTiming = {
        start_ms: snappedStart,
        end_ms: snappedEnd,
      };
      slot.timingResolution = "energy.snapped";
      slot.energySnapConfidence = confidence;
    } else {
      slot.energyTiming = {
        start_ms: snappedStart,
        end_ms: snappedEnd,
        method: "rms-valley-snap",
        confidence,
      };
      slot.energySnapConfidence = confidence;
    }
  }

  enforceMonotonicWordOrder(timingWords);
  return timingWords;
}

function collectCandidatePoints(landmarks, finalMs) {
  const points = [];
  for (const ms of landmarks?.onsets ?? []) points.push({ ms: Math.round(ms), type: "onset" });
  for (const ms of landmarks?.offsets ?? []) points.push({ ms: Math.round(ms), type: "offset" });
  for (const ms of landmarks?.valleys ?? []) points.push({ ms: Math.round(ms), type: "valley" });
  for (const silence of landmarks?.silences ?? []) {
    if (Number.isFinite(silence?.start_ms)) points.push({ ms: Math.round(silence.start_ms), type: "silence-start" });
    if (Number.isFinite(silence?.end_ms)) points.push({ ms: Math.round(silence.end_ms), type: "silence-end" });
  }
  for (const ms of landmarks?.peaks ?? []) points.push({ ms: Math.round(ms), type: "peak" });
  if (Number.isFinite(finalMs)) {
    points.push({ ms: Math.max(0, Math.round(finalMs)), type: "tail" });
  }
  return points;
}

function findBoundaryCandidate({ centerMs, toleranceMs, minMs, maxMs, points, preferredTypes }) {
  if (!Number.isFinite(centerMs) || !Number.isFinite(minMs) || !Number.isFinite(maxMs) || minMs > maxMs) {
    return null;
  }
  let best = null;
  const rank = new Map(preferredTypes.map((type, index) => [type, index]));
  for (const point of points) {
    const ms = Number(point?.ms);
    if (!Number.isFinite(ms) || ms < minMs || ms > maxMs) {
      continue;
    }
    const distance = Math.abs(ms - centerMs);
    if (distance > toleranceMs) {
      continue;
    }
    const typeRank = rank.has(point.type) ? rank.get(point.type) : preferredTypes.length + 1;
    const typeWeight = landmarkTypeWeight(point.type);
    const distanceScore = Math.max(0, 1 - distance / Math.max(1, toleranceMs));
    const confidence = Math.max(0, Math.min(1, typeWeight * distanceScore));
    const score = confidence * 10 - typeRank;
    if (!best || score > best.score) {
      best = { ms: Math.round(ms), confidence, type: point.type, score };
    }
  }
  return best;
}

function landmarkTypeWeight(type) {
  switch (type) {
    case "onset":
    case "offset":
      return 1.0;
    case "valley":
      return 0.72;
    case "silence-start":
    case "silence-end":
      return 0.6;
    case "peak":
      return 0.35;
    default:
      return 0.5;
  }
}

function combineConfidence(startCandidate, endCandidate) {
  const start = startCandidate?.confidence ?? 0;
  const end = endCandidate?.confidence ?? 0;
  if (start === 0 && end === 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, Number(((start + end) / 2).toFixed(3))));
}

function clampBoundary(value, min, max) {
  if (!Number.isFinite(value)) {
    return min;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) {
    return Math.round(value);
  }
  return Math.max(Math.round(min), Math.min(Math.round(max), Math.round(value)));
}

function findPreviousTimedWord(slots, index) {
  for (let i = index - 1; i >= 0; i--) {
    if (slots[i]?.resolvedTiming) {
      return slots[i];
    }
  }
  return null;
}

function findNextTimedWord(slots, index) {
  for (let i = index + 1; i < slots.length; i++) {
    if (slots[i]?.whisperTiming) {
      return slots[i];
    }
  }
  return null;
}

function enforceMonotonicWordOrder(words) {
  let previousEnd = 0;
  for (const word of words) {
    if (!word?.resolvedTiming) {
      continue;
    }
    const whisper = word.whisperTiming ?? word.resolvedTiming;
    const start = Math.max(previousEnd, word.resolvedTiming.start_ms);
    const end = Math.max(start + MIN_RESOLVED_DURATION_MS, word.resolvedTiming.end_ms);
    word.resolvedTiming = { start_ms: start, end_ms: end };
    if (word.energyTiming) {
      word.energyTiming = { ...word.energyTiming, start_ms: start, end_ms: end };
    }
    if (word.timingResolution === "energy.snapped" && whisper.start_ms === start && whisper.end_ms === end) {
      word.timingResolution = "word.timing";
    }
    previousEnd = end;
  }
}
