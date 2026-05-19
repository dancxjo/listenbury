const DEFAULT_MIN_DB = -96;
const DEFAULT_MAX_DB = 0;
const EPSILON = 1e-12;

const WINDOW_CACHE = new Map();
const FFT_PLAN_CACHE = new Map();

export function defaultSpectrogramLevels(sampleRate) {
  const coarseHop = Math.max(1, Math.round(sampleRate * 0.01));
  const detailHop = Math.max(1, Math.round(coarseHop / 2));
  return [
    {
      id: "overview",
      windowSize: 1024,
      hopSize: coarseHop,
      fftSize: 1024,
    },
    {
      id: "detail",
      windowSize: 512,
      hopSize: detailHop,
      fftSize: 512,
    },
  ];
}

export function analyzeSpectrogramSamples(samples, options = {}) {
  const pcm = toFloat32Array(samples);
  const sampleRate = Math.max(1, Math.round(options.sampleRate ?? 16_000));
  const levels = normalizeLevels(options.levels ?? defaultSpectrogramLevels(sampleRate), sampleRate, options);
  return buildSpectrogram(pcm, sampleRate, levels, options, null);
}

export function appendSpectrogramSamples(previous, samples, options = {}) {
  if (!previous?.levels?.length) {
    return analyzeSpectrogramSamples(samples, options);
  }

  const pcm = toFloat32Array(samples);
  const sampleRate = Math.max(1, Math.round(options.sampleRate ?? previous.sampleRate ?? 16_000));
  const levels = normalizeLevels(options.levels ?? previous.levels, sampleRate, {
    ...previous,
    ...options,
  });
  const reusable =
    previous.sampleRate === sampleRate &&
    pcm.length >= (previous.sampleCount ?? 0) &&
    sameLevelDefinitions(previous.levels, levels)
      ? previous
      : null;
  return buildSpectrogram(pcm, sampleRate, levels, { ...previous, ...options }, reusable);
}

export function selectSpectrogramLevel(spectrogram, { pxPerSecond } = {}) {
  const levels = spectrogram?.levels ?? [];
  if (!levels.length) {
    return null;
  }
  const msPerPixel = 1000 / Math.max(1, Number(pxPerSecond) || 1);
  const ordered = [...levels].sort((left, right) => right.hopMs - left.hopMs);
  return ordered.find((level) => level.hopMs <= msPerPixel * 2) ?? ordered[ordered.length - 1];
}

export function spectrogramValueAt(level, timeMs, frequencyHz) {
  if (!level?.frames?.length) {
    return null;
  }
  const frameIndex = Math.max(
    0,
    Math.min(level.frames.length - 1, Math.round(Math.max(0, Number(timeMs) || 0) / level.hopMs)),
  );
  const binIndex = Math.max(
    0,
    Math.min(level.binCount - 1, Math.round((Math.max(0, Number(frequencyHz) || 0) / level.nyquistHz) * (level.binCount - 1))),
  );
  return level.frames[frameIndex]?.[binIndex] ?? null;
}

function buildSpectrogram(samples, sampleRate, levels, options, previous) {
  const previousLevels = new Map((previous?.levels ?? []).map((level) => [level.id, level]));
  const builtLevels = levels.map((level) => {
    const prior = previousLevels.get(level.id);
    return buildSpectrogramLevel(samples, sampleRate, level, prior);
  });
  return {
    sampleRate,
    sampleCount: samples.length,
    durationMs: samples.length > 0 ? (samples.length / sampleRate) * 1000 : 0,
    dbScale: options.dbScale ?? true,
    minDb: options.minDb ?? DEFAULT_MIN_DB,
    maxDb: options.maxDb ?? DEFAULT_MAX_DB,
    levels: builtLevels,
    analysisMode: previous ? "append" : "full",
  };
}

function buildSpectrogramLevel(samples, sampleRate, level, previousLevel) {
  const frameCount = frameCountForSampleCount(samples.length, level.hopSize);
  const previousFrames = previousLevel?.frames ?? [];
  const fullyResolvedFrameCount =
    (previousLevel?.sampleCount ?? 0) >= level.windowSize
      ? Math.floor(((previousLevel.sampleCount ?? 0) - level.windowSize) / level.hopSize) + 1
      : 0;
  const reusableFrameCount =
    previousLevel &&
    previousLevel.windowSize === level.windowSize &&
    previousLevel.hopSize === level.hopSize &&
    previousLevel.fftSize === level.fftSize &&
    previousLevel.sampleRate === sampleRate &&
    previousFrames.length <= frameCount
      ? Math.max(0, Math.min(previousFrames.length, fullyResolvedFrameCount))
      : 0;

  const frames = previousFrames.slice(0, reusableFrameCount);
  const window = hannWindow(level.windowSize);
  for (let frameIndex = reusableFrameCount; frameIndex < frameCount; frameIndex++) {
    frames.push(analyzeFrame(samples, frameIndex * level.hopSize, level, window));
  }

  return {
    id: level.id,
    sampleRate,
    windowName: "hann",
    windowSize: level.windowSize,
    hopSize: level.hopSize,
    hopMs: (level.hopSize / sampleRate) * 1000,
    fftSize: level.fftSize,
    binCount: level.binCount,
    binHz: sampleRate / level.fftSize,
    nyquistHz: sampleRate / 2,
    dbScale: true,
    minValue: level.minValue,
    maxValue: level.maxValue,
    frameDurationMs: (level.hopSize / sampleRate) * 1000,
    frameCount,
    sampleCount: samples.length,
    reusedFrameCount: reusableFrameCount,
    frames,
  };
}

function analyzeFrame(samples, startSample, level, window) {
  const fftReal = new Float64Array(level.fftSize);
  const fftImag = new Float64Array(level.fftSize);
  for (let index = 0; index < level.windowSize; index++) {
    fftReal[index] = (samples[startSample + index] ?? 0) * window[index];
  }

  fftInPlace(fftReal, fftImag);

  const bins = new Float32Array(level.binCount);
  for (let index = 0; index < level.binCount; index++) {
    const magnitude = Math.hypot(fftReal[index], fftImag[index]) / Math.max(1, level.windowSize);
    const db = 20 * Math.log10(magnitude + EPSILON);
    bins[index] = Math.max(level.minValue, Math.min(level.maxValue, db));
  }
  return bins;
}

function frameCountForSampleCount(sampleCount, hopSize) {
  if (sampleCount <= 0) {
    return 0;
  }
  return Math.floor((sampleCount - 1) / Math.max(1, hopSize)) + 1;
}

function normalizeLevels(levels, sampleRate, options) {
  return levels.map((level, index) => {
    const windowSize = nearestPowerOfTwo(Math.max(32, Math.round(level.windowSize ?? level.fftSize ?? 512)));
    const fftSize = nearestPowerOfTwo(Math.max(windowSize, Math.round(level.fftSize ?? windowSize)));
    const hopSize = Math.max(1, Math.round(level.hopSize ?? sampleRate * 0.01));
    return {
      id: level.id ?? `level-${index + 1}`,
      windowSize,
      hopSize,
      fftSize,
      binCount: fftSize / 2 + 1,
      minValue: Number.isFinite(level.minValue) ? level.minValue : options.minDb ?? DEFAULT_MIN_DB,
      maxValue: Number.isFinite(level.maxValue) ? level.maxValue : options.maxDb ?? DEFAULT_MAX_DB,
    };
  });
}

function sameLevelDefinitions(previousLevels, nextLevels) {
  if ((previousLevels?.length ?? 0) !== nextLevels.length) {
    return false;
  }
  return previousLevels.every((level, index) => {
    const next = nextLevels[index];
    return (
      level.id === next.id &&
      level.windowSize === next.windowSize &&
      level.hopSize === next.hopSize &&
      level.fftSize === next.fftSize
    );
  });
}

function toFloat32Array(samples) {
  if (samples instanceof Float32Array) {
    return samples;
  }
  if (ArrayBuffer.isView(samples)) {
    return new Float32Array(samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength));
  }
  return Float32Array.from(samples ?? []);
}

function hannWindow(size) {
  let cached = WINDOW_CACHE.get(size);
  if (cached) {
    return cached;
  }
  cached = new Float64Array(size);
  for (let index = 0; index < size; index++) {
    cached[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / Math.max(1, size - 1)));
  }
  WINDOW_CACHE.set(size, cached);
  return cached;
}

function fftInPlace(real, imag) {
  const size = real.length;
  const plan = fftPlan(size);

  for (let index = 0; index < size; index++) {
    const reversedIndex = plan.bitReverse[index];
    if (reversedIndex > index) {
      const tmpReal = real[index];
      real[index] = real[reversedIndex];
      real[reversedIndex] = tmpReal;

      const tmpImag = imag[index];
      imag[index] = imag[reversedIndex];
      imag[reversedIndex] = tmpImag;
    }
  }

  for (let step = 2; step <= size; step <<= 1) {
    const halfStep = step >> 1;
    const tableStep = size / step;
    for (let offset = 0; offset < size; offset += step) {
      for (let index = 0; index < halfStep; index++) {
        const twiddleIndex = index * tableStep;
        const cos = plan.cos[twiddleIndex];
        const sin = plan.sin[twiddleIndex];

        const match = offset + index + halfStep;
        const target = offset + index;

        const tReal = cos * real[match] - sin * imag[match];
        const tImag = sin * real[match] + cos * imag[match];

        real[match] = real[target] - tReal;
        imag[match] = imag[target] - tImag;
        real[target] += tReal;
        imag[target] += tImag;
      }
    }
  }
}

function fftPlan(size) {
  let cached = FFT_PLAN_CACHE.get(size);
  if (cached) {
    return cached;
  }
  const bits = Math.round(Math.log2(size));
  const bitReverse = new Uint32Array(size);
  for (let index = 0; index < size; index++) {
    bitReverse[index] = reverseBits(index, bits);
  }
  const cos = new Float64Array(size / 2);
  const sin = new Float64Array(size / 2);
  for (let index = 0; index < size / 2; index++) {
    const angle = (-2 * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }
  cached = { bitReverse, cos, sin };
  FFT_PLAN_CACHE.set(size, cached);
  return cached;
}

function reverseBits(value, width) {
  let reversed = 0;
  for (let index = 0; index < width; index++) {
    reversed = (reversed << 1) | (value & 1);
    value >>= 1;
  }
  return reversed;
}

function nearestPowerOfTwo(value) {
  return 2 ** Math.ceil(Math.log2(Math.max(1, value)));
}
