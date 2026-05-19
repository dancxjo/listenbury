/**
 * scene-heading.mjs
 *
 * Utilities for generating realistic screenplay sluglines from location context.
 *
 * Slugline policy (in priority order):
 *   1. Explicit user/session location metadata
 *   2. Geolocation-derived context (if available and appropriate)
 *   3. Webcam/vision-derived setting classification
 *   4. Stable user-provided place labels
 *   5. Explicit user utterances naming the place
 *   6. Prior stable session context
 *   7. Restrained fallback: UNKNOWN ROOM, UNKNOWN OUTDOOR AREA, etc.
 *
 * Scene headings must describe determinable physical locations and times of day,
 * NOT internal runtime labels, topic names, or mood/tone labels.
 */

/**
 * Rules for classifying vision observations into physical locations.
 * Each rule specifies a set of patterns and the resulting place/interior-exterior pair.
 * Rules are checked in order; the first match wins.
 */
const VISION_RULES = [
  { patterns: [/steering wheel/i, /dashboard/i, /car seat/i, /\bcar\b/i], place: "CAR", ie: "INT." },
  { patterns: [/\bbed\b/i, /nightstand/i, /bedroom/i, /\bpillow\b/i, /mattress/i], place: "BEDROOM", ie: "INT." },
  { patterns: [/\bcouch\b/i, /\bsofa\b/i, /living room/i, /\blamp\b/i], place: "LIVING ROOM", ie: "INT." },
  { patterns: [/\bdesk\b/i, /home office/i, /office chair/i, /\bmonitor\b/i, /computer screen/i], place: "HOME OFFICE", ie: "INT." },
  { patterns: [/\bkitchen\b/i, /\bcounter\b/i, /\bstove\b/i, /refrigerator/i, /\bfridge\b/i, /\bsink\b/i], place: "KITCHEN", ie: "INT." },
  { patterns: [/\bclassroom\b/i, /whiteboard/i, /chalkboard/i, /\bdesk row/i], place: "CLASSROOM", ie: "INT." },
  { patterns: [/\bchurch\b/i, /\bpew\b/i, /\baltar\b/i], place: "CHURCH", ie: "INT." },
  { patterns: [/\bgrass\b/i, /\btrees?\b/i, /\bpark\b/i, /\bbench\b/i], place: "PARK", ie: "EXT." },
  { patterns: [/\bsidewalk\b/i, /\bstreet\b/i, /\broad\b/i, /\bcurb\b/i], place: "STREET", ie: "EXT." },
  { patterns: [/\byard\b/i, /\blawn\b/i, /\bgarden\b/i], place: "YARD", ie: "EXT." },
  { patterns: [/\bsky\b/i, /\boutdoor\b/i, /\boutside\b/i, /open air/i], place: "UNKNOWN OUTDOOR AREA", ie: "EXT." },
  { patterns: [/\bindoor\b/i, /\broom\b/i, /\bceiling\b/i, /\bwall\b/i, /\bfloor\b/i], place: "UNKNOWN ROOM", ie: "INT." },
];

/**
 * Classify an array of vision observation strings into a location descriptor.
 *
 * @param {string[]} observations - feature strings from webcam/vision (e.g. ["couch", "lamp", "indoor room"])
 * @returns {{ place: string, interiorExterior: string, confidence: string } | null}
 */
export function resolveLocationFromVision(observations) {
  if (!observations || observations.length === 0) {
    return null;
  }
  const combined = observations.join(" ").toLowerCase();
  for (const rule of VISION_RULES) {
    if (rule.patterns.some((pattern) => pattern.test(combined))) {
      return { place: rule.place, interiorExterior: rule.ie, confidence: "vision" };
    }
  }
  return null;
}

/**
 * Resolve a physical location from the available context.
 *
 * Context shape:
 *   place           - explicit string label (e.g. "LIVING ROOM")
 *   interiorExterior - "INT." | "EXT." | "INT./EXT."
 *   vision          - array of observed feature strings (from webcam/vision)
 *   timeOfDay       - explicit override for time of day label
 *   timestampMs     - epoch milliseconds used to infer time of day when not explicit
 *
 * @param {object} [context={}]
 * @returns {{ place: string, interiorExterior: string, confidence: string }}
 */
export function resolveLocation(context = {}) {
  // 1. Explicit user/session place metadata
  if (context.place) {
    return {
      place: String(context.place).toUpperCase(),
      interiorExterior: context.interiorExterior ?? "INT.",
      confidence: "explicit",
    };
  }

  // 3. Webcam/vision-derived classification
  if (context.vision && context.vision.length > 0) {
    const fromVision = resolveLocationFromVision(context.vision);
    if (fromVision) {
      return fromVision;
    }
  }

  // 7. Restrained fallback
  return { place: "UNKNOWN ROOM", interiorExterior: "INT.", confidence: "fallback" };
}

/**
 * Minimum millisecond value that is treated as a real wall-clock epoch timestamp.
 * Values below this threshold are assumed to be elapsed-time offsets (e.g. session
 * elapsed_ms counters starting near zero) rather than actual Date.now() timestamps.
 */
const MIN_EPOCH_MS = 86_400_000; // 24 hours since Unix epoch (1970-01-02)

/**
 * Infer a time-of-day label from an epoch millisecond timestamp.
 *
 * Falls back to "DAY" when no real wall-clock timestamp is available.
 * Values smaller than MIN_EPOCH_MS are treated as elapsed-time offsets rather
 * than real epoch timestamps and return "DAY".
 *
 * @param {number|null} [timestampMs=null]
 * @returns {"DAY"|"AFTERNOON"|"EVENING"|"NIGHT"}
 */
export function resolveTimeOfDay(timestampMs = null) {
  if (timestampMs == null || !Number.isFinite(timestampMs) || timestampMs < MIN_EPOCH_MS) {
    return "DAY";
  }
  const hour = new Date(timestampMs).getHours();
  if (hour >= 5 && hour < 12) return "DAY";
  if (hour >= 12 && hour < 17) return "AFTERNOON";
  if (hour >= 17 && hour < 20) return "EVENING";
  return "NIGHT";
}

/**
 * Format a proper screenplay slugline.
 *
 * @param {string} interiorExterior - "INT." | "EXT." | "INT./EXT."
 * @param {string} place - physical place name in uppercase (e.g. "LIVING ROOM")
 * @param {string} timeOfDay - time label (e.g. "DAY", "NIGHT", "EVENING")
 * @returns {string} e.g. "INT. LIVING ROOM - NIGHT"
 */
export function formatSlugline(interiorExterior, place, timeOfDay) {
  return `${interiorExterior} ${place} - ${timeOfDay}`;
}

/**
 * Convert an UPPER CASE or mixed string to Title Case.
 * e.g. "QUIET GRIEF" → "Quiet Grief", "PHONOLOGY WORKBENCH" → "Phonology Workbench"
 *
 * @param {string} text
 * @returns {string}
 */
export function toTitleCase(text) {
  return String(text ?? "")
    .toLowerCase()
    .replace(/(?:^|\s|-)\S/g, (char) => char.toUpperCase());
}

/**
 * Build a complete slugline from context, combining location and time-of-day resolution.
 *
 * @param {object} [context={}]
 * @returns {string} e.g. "INT. UNKNOWN ROOM - DAY"
 */
export function resolveSlugline(context = {}) {
  const location = resolveLocation(context);
  const timeOfDay = context.timeOfDay ?? resolveTimeOfDay(context.timestampMs ?? null);
  return formatSlugline(location.interiorExterior, location.place, timeOfDay);
}
