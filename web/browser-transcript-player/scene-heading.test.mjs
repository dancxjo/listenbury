import test from "node:test";
import assert from "node:assert/strict";

import {
  resolveLocation,
  resolveLocationFromVision,
  resolveTimeOfDay,
  formatSlugline,
  resolveSlugline,
} from "./scene-heading.mjs";

// ──────────────────────────────────────────────────────────────────────────────
// formatSlugline
// ──────────────────────────────────────────────────────────────────────────────

test("formatSlugline produces a proper INT slugline", () => {
  assert.equal(formatSlugline("INT.", "LIVING ROOM", "NIGHT"), "INT. LIVING ROOM - NIGHT");
});

test("formatSlugline produces a proper EXT slugline", () => {
  assert.equal(formatSlugline("EXT.", "PARK", "DAY"), "EXT. PARK - DAY");
});

test("formatSlugline produces an INT./EXT. slugline", () => {
  assert.equal(formatSlugline("INT./EXT.", "UNKNOWN LOCATION", "PRESENT"), "INT./EXT. UNKNOWN LOCATION - PRESENT");
});

// ──────────────────────────────────────────────────────────────────────────────
// resolveTimeOfDay
// ──────────────────────────────────────────────────────────────────────────────

test("resolveTimeOfDay returns DAY for null", () => {
  assert.equal(resolveTimeOfDay(null), "DAY");
});

test("resolveTimeOfDay returns DAY for undefined", () => {
  assert.equal(resolveTimeOfDay(undefined), "DAY");
});

test("resolveTimeOfDay returns DAY for small elapsed-time values", () => {
  // 500ms elapsed time — not a real wall-clock timestamp
  assert.equal(resolveTimeOfDay(500), "DAY");
});

test("resolveTimeOfDay returns DAY for a real morning timestamp", () => {
  // 2024-01-01T09:00:00Z
  const ms = new Date("2024-01-01T09:00:00Z").getTime();
  assert.equal(resolveTimeOfDay(ms), "DAY");
});

test("resolveTimeOfDay returns AFTERNOON for a real afternoon timestamp", () => {
  // 2024-01-01T14:00:00Z — local time depends on timezone, but UTC+0 gives 14h
  const ms = new Date("2024-01-01T14:00:00Z").getTime();
  const result = resolveTimeOfDay(ms);
  assert.ok(["AFTERNOON", "DAY", "EVENING"].includes(result), `unexpected: ${result}`);
});

test("resolveTimeOfDay returns NIGHT for a real night timestamp", () => {
  // Pick a time that is clearly late at night in UTC
  const ms = new Date("2024-01-01T23:00:00Z").getTime();
  const result = resolveTimeOfDay(ms);
  assert.ok(["NIGHT", "EVENING"].includes(result), `unexpected: ${result}`);
});

// ──────────────────────────────────────────────────────────────────────────────
// resolveLocationFromVision
// ──────────────────────────────────────────────────────────────────────────────

test("resolveLocationFromVision returns null for empty observations", () => {
  assert.equal(resolveLocationFromVision([]), null);
  assert.equal(resolveLocationFromVision(null), null);
});

test("resolveLocationFromVision classifies living room from couch and lamp", () => {
  const result = resolveLocationFromVision(["couch", "lamp", "indoor room"]);
  assert.equal(result.place, "LIVING ROOM");
  assert.equal(result.interiorExterior, "INT.");
  assert.equal(result.confidence, "vision");
});

test("resolveLocationFromVision classifies bedroom from bed and nightstand", () => {
  const result = resolveLocationFromVision(["bed", "nightstand", "indoor room"]);
  assert.equal(result.place, "BEDROOM");
  assert.equal(result.interiorExterior, "INT.");
});

test("resolveLocationFromVision classifies outdoor park from grass and trees", () => {
  const result = resolveLocationFromVision(["grass", "trees", "sky", "outdoor area"]);
  assert.equal(result.place, "PARK");
  assert.equal(result.interiorExterior, "EXT.");
});

test("resolveLocationFromVision classifies car from steering wheel and dashboard", () => {
  const result = resolveLocationFromVision(["steering wheel", "dashboard"]);
  assert.equal(result.place, "CAR");
  assert.equal(result.interiorExterior, "INT.");
});

test("resolveLocationFromVision classifies home office from desk and monitor", () => {
  const result = resolveLocationFromVision(["desk", "monitor", "computer screen"]);
  assert.equal(result.place, "HOME OFFICE");
  assert.equal(result.interiorExterior, "INT.");
});

test("resolveLocationFromVision falls back to UNKNOWN OUTDOOR AREA for generic outdoor cues", () => {
  const result = resolveLocationFromVision(["sky", "outside"]);
  assert.equal(result.interiorExterior, "EXT.");
  assert.ok(result.place.includes("OUTDOOR"), `expected outdoor place, got: ${result.place}`);
});

test("resolveLocationFromVision falls back to UNKNOWN ROOM for generic indoor cues", () => {
  const result = resolveLocationFromVision(["indoor", "room"]);
  assert.equal(result.place, "UNKNOWN ROOM");
  assert.equal(result.interiorExterior, "INT.");
});

test("resolveLocationFromVision returns null when observations do not match any rule", () => {
  const result = resolveLocationFromVision(["abstract concept", "runtime", "quiet grief"]);
  assert.equal(result, null);
});

// ──────────────────────────────────────────────────────────────────────────────
// resolveLocation
// ──────────────────────────────────────────────────────────────────────────────

test("resolveLocation uses explicit place when provided", () => {
  const result = resolveLocation({ place: "living room", interiorExterior: "INT." });
  assert.equal(result.place, "LIVING ROOM");
  assert.equal(result.confidence, "explicit");
});

test("resolveLocation defaults interiorExterior to INT. when explicit place given without it", () => {
  const result = resolveLocation({ place: "bedroom" });
  assert.equal(result.interiorExterior, "INT.");
});

test("resolveLocation uses vision observations when no explicit place", () => {
  const result = resolveLocation({ vision: ["couch", "lamp"] });
  assert.equal(result.place, "LIVING ROOM");
  assert.equal(result.confidence, "vision");
});

test("resolveLocation falls back to UNKNOWN ROOM when no context available", () => {
  const result = resolveLocation({});
  assert.equal(result.place, "UNKNOWN ROOM");
  assert.equal(result.interiorExterior, "INT.");
  assert.equal(result.confidence, "fallback");
});

test("resolveLocation falls back when vision observations do not match any rule", () => {
  const result = resolveLocation({ vision: ["runtime", "phonology workbench", "quiet grief"] });
  assert.equal(result.place, "UNKNOWN ROOM");
  assert.equal(result.confidence, "fallback");
});

// ──────────────────────────────────────────────────────────────────────────────
// resolveSlugline
// ──────────────────────────────────────────────────────────────────────────────

test("resolveSlugline produces INT. UNKNOWN ROOM - DAY when no context", () => {
  assert.equal(resolveSlugline({}), "INT. UNKNOWN ROOM - DAY");
});

test("resolveSlugline uses explicit place and time of day", () => {
  assert.equal(
    resolveSlugline({ place: "living room", interiorExterior: "INT.", timeOfDay: "NIGHT" }),
    "INT. LIVING ROOM - NIGHT",
  );
});

test("resolveSlugline resolves vision to EXT. PARK - DAY", () => {
  assert.equal(
    resolveSlugline({ vision: ["grass", "trees", "bench"], timeOfDay: "DAY" }),
    "EXT. PARK - DAY",
  );
});

test("resolveSlugline resolves vision to INT. BEDROOM - NIGHT", () => {
  assert.equal(
    resolveSlugline({ vision: ["bed", "nightstand"], timeOfDay: "NIGHT" }),
    "INT. BEDROOM - NIGHT",
  );
});

test("resolveSlugline does not use mood or topic labels as location", () => {
  // Topic and mood labels should not appear in sluglines
  const result = resolveSlugline({ vision: ["quiet grief", "phonology workbench"] });
  assert.ok(!result.includes("GRIEF"), `mood label leaked into slugline: ${result}`);
  assert.ok(!result.includes("PHONOLOGY"), `topic label leaked into slugline: ${result}`);
  assert.ok(!result.includes("RUNTIME"), `runtime label leaked into slugline: ${result}`);
  assert.match(result, /^INT\.\s+UNKNOWN ROOM - DAY$/);
});

test("resolveSlugline uses explicit timeOfDay override when provided", () => {
  const result = resolveSlugline({ timeOfDay: "EVENING" });
  assert.equal(result, "INT. UNKNOWN ROOM - EVENING");
});
