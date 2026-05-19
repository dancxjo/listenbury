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

function clampFinite(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
