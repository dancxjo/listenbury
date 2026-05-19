const MIN_BUCKET_DURATION_MS = 0.0001;
const FLOAT_TOLERANCE = 0.000001;

export class TimeScale {
  constructor({ pxPerSecond, durationMs }) {
    this.pxPerSecond = clampFinite(pxPerSecond, 1);
    this.durationMs = Math.max(0, clampFinite(durationMs, 0));
  }

  get pxPerMs() {
    return this.pxPerSecond / 1000;
  }

  msToPx(ms) {
    return clampFinite(ms, 0) * this.pxPerMs;
  }

  pxToMs(px) {
    return clampFinite(px, 0) / this.pxPerMs;
  }

  durationToPx(durationMs) {
    return Math.max(0, clampFinite(durationMs, 0) * this.pxPerMs);
  }

  intervalToPx({ startMs, endMs, minWidthPx = 0 }) {
    const start = clampFinite(startMs, 0);
    const end = Math.max(start, clampFinite(endMs, start));
    return {
      leftPx: this.msToPx(start),
      widthPx: Math.max(minWidthPx, this.durationToPx(end - start)),
    };
  }

  waveformPeakIndexAtPx(px, { audioDurationMs, peakCount }) {
    const duration = Math.max(0, clampFinite(audioDurationMs, 0));
    const count = Math.max(0, Math.floor(clampFinite(peakCount, 0)));
    if (duration <= 0 || count <= 0) {
      return null;
    }
    const timeMs = clampFinite(px, 0) / this.pxPerMs;
    if (timeMs < 0 || timeMs > duration) {
      return null;
    }
    return Math.min(count - 1, Math.floor((timeMs / duration) * count));
  }
}

export class TimelineViewport {
  constructor({ pxPerSecond, durationMs, viewportWidthPx = 0, scrollLeftPx = 0 }) {
    this.scale = new TimeScale({ pxPerSecond, durationMs });
    this.viewportWidthPx = Math.max(0, clampFinite(viewportWidthPx, 0));
    this.scrollLeftPx = Math.max(0, clampFinite(scrollLeftPx, 0));
  }

  contentWidthPx() {
    return Math.max(this.viewportWidthPx, Math.ceil(this.scale.msToPx(this.scale.durationMs)));
  }

  visibleRangeMs({ minDurationMs = 0 } = {}) {
    const startMs = Math.max(0, this.scale.pxToMs(this.scrollLeftPx));
    const endMs = Math.min(
      this.scale.durationMs,
      this.scale.pxToMs(this.scrollLeftPx + this.viewportWidthPx),
    );
    return {
      startMs,
      endMs,
      durationMs: Math.max(minDurationMs, endMs - startMs),
    };
  }

  scrollLeftForAnchor(anchorMs, xInViewportPx, contentWidthPx = this.contentWidthPx()) {
    const maxScrollLeft = Math.max(0, contentWidthPx - this.viewportWidthPx);
    return Math.max(
      0,
      Math.min(maxScrollLeft, this.scale.msToPx(anchorMs) - clampFinite(xInViewportPx, 0)),
    );
  }
}

export function createTimeScale(options) {
  return new TimeScale(options);
}

export function createTimelineViewport(options) {
  return new TimelineViewport(options);
}

export function renderedCanvasXToTimelinePx({ canvasX, canvasWidthPx, renderedWidthPx }) {
  const width = Math.max(1, clampFinite(canvasWidthPx, 1));
  const rendered = Math.max(1, clampFinite(renderedWidthPx, width));
  return clampFinite(canvasX, 0) * (rendered / width);
}

export function buildWaveformResolutionLevels(audioBuffer, options = {}) {
  const sampleRate = Math.max(1, clampFinite(audioBuffer?.sampleRate, 1));
  const sampleCount = Math.max(0, Math.floor(clampFinite(audioBuffer?.length, 0)));
  const channelCount = Math.max(1, Math.floor(clampFinite(audioBuffer?.numberOfChannels, 1)));
  if (sampleCount <= 0) {
    return [];
  }

  const targetBucketCounts = Array.isArray(options.targetBucketCounts) && options.targetBucketCounts.length
    ? options.targetBucketCounts
    : [600, 2400, 9600, 19_200];
  const bucketSizes = [...new Set(targetBucketCounts
    .map((target) => Math.max(1, Math.ceil(sampleCount / Math.max(1, Math.floor(clampFinite(target, 1))))))
    .sort((left, right) => right - left))];

  const channels = [];
  for (let channel = 0; channel < channelCount; channel++) {
    channels.push(audioBuffer.getChannelData(channel));
  }

  return bucketSizes.map((bucketSize) => {
    const buckets = [];
    for (let startSample = 0; startSample < sampleCount; startSample += bucketSize) {
      const endSample = Math.min(sampleCount, startSample + bucketSize);
      let minSample = Infinity;
      let maxSample = -Infinity;
      let sumSquares = 0;
      for (let sampleIndex = startSample; sampleIndex < endSample; sampleIndex++) {
        let frameMin = Infinity;
        let frameMax = -Infinity;
        let frameSquareSum = 0;
        for (let channel = 0; channel < channelCount; channel++) {
          const sample = channels[channel]?.[sampleIndex] ?? 0;
          frameMin = Math.min(frameMin, sample);
          frameMax = Math.max(frameMax, sample);
          frameSquareSum += sample * sample;
        }
        minSample = frameMin === Infinity ? minSample : Math.min(minSample, frameMin);
        maxSample = frameMax === -Infinity ? maxSample : Math.max(maxSample, frameMax);
        sumSquares += frameSquareSum / channelCount;
      }
      const frameCount = Math.max(1, endSample - startSample);
      buckets.push({
        start_ms: (startSample / sampleRate) * 1000,
        end_ms: (endSample / sampleRate) * 1000,
        min_sample: minSample === Infinity ? 0 : minSample,
        max_sample: maxSample === -Infinity ? 0 : maxSample,
        rms_energy: Math.sqrt(sumSquares / frameCount),
        sample_count: frameCount,
      });
    }

    const bucketDurationMs = (bucketSize / sampleRate) * 1000;
    return {
      bucketDurationMs,
      bucketCount: buckets.length,
      label: formatWaveformBucketDuration(bucketDurationMs),
      buckets,
    };
  });
}

export function selectWaveformResolutionLevel(levels, options = {}) {
  if (!Array.isArray(levels) || levels.length === 0) {
    return null;
  }
  const pxPerSecond = Math.max(1, clampFinite(options.pxPerSecond, 1));
  const bucketsPerPixel = Math.max(0.25, clampFinite(options.bucketsPerPixel, 2));
  const desiredBucketDurationMs = (1000 / pxPerSecond) * bucketsPerPixel;

  let bestLevel = levels[0];
  let bestError = Number.POSITIVE_INFINITY;
  for (const level of levels) {
    const durationMs = Math.max(MIN_BUCKET_DURATION_MS, clampFinite(level?.bucketDurationMs, MIN_BUCKET_DURATION_MS));
    const error = Math.abs(Math.log(durationMs / desiredBucketDurationMs));
    if (
      error < bestError - FLOAT_TOLERANCE ||
      (Math.abs(error - bestError) <= FLOAT_TOLERANCE &&
        durationMs < Math.max(MIN_BUCKET_DURATION_MS, clampFinite(bestLevel?.bucketDurationMs, MIN_BUCKET_DURATION_MS)))
    ) {
      bestLevel = level;
      bestError = error;
    }
  }
  return bestLevel;
}

export function waveformBucketToPx(scale, bucket, minWidthPx = 0) {
  return scale.intervalToPx({
    startMs: bucket?.start_ms ?? 0,
    endMs: bucket?.end_ms ?? bucket?.start_ms ?? 0,
    minWidthPx,
  });
}

export function formatWaveformBucketDuration(bucketDurationMs) {
  const durationMs = Math.max(MIN_BUCKET_DURATION_MS, clampFinite(bucketDurationMs, MIN_BUCKET_DURATION_MS));
  if (durationMs >= 1000) {
    return `${stripTrailingZeros((durationMs / 1000).toFixed(2))}s buckets`;
  }
  if (durationMs >= 1) {
    return `${stripTrailingZeros(durationMs.toFixed(durationMs >= 10 ? 0 : 1))}ms buckets`;
  }
  return `${stripTrailingZeros(durationMs.toFixed(3))}ms buckets`;
}

function clampFinite(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function stripTrailingZeros(value) {
  return String(value).replace(/(?:\.0+|(\.\d*?)0+)$/, "$1");
}
