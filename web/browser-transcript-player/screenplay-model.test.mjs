import test from "node:test";
import assert from "node:assert/strict";

import {
  assembleNarrativeManuscript,
  buildNarrativeEpisode,
  createNarrativeSession,
  reduceNarrativeEvent,
} from "./screenplay-model.mjs";

function mkEvent(kind, turn, elapsed_ms, extra = {}) {
  return { kind, turn, elapsed_ms, ...extra };
}

function buildWaveDeckFixtureSession() {
  const session = createNarrativeSession();
  const events = [
    mkEvent("speech_started", 1, 100),
    mkEvent("transcript_candidate", 1, 140, {
      artifact: { stable_text: "Can you explain", unstable_text: "that overlap routing means" },
    }),
    mkEvent("asr_timed_word_stream", 1, 150, {
      artifact: {
        words: [
          { id: 1, text: "Can", commitment: "StableText" },
          { id: 2, text: "you", commitment: "StableText" },
          { id: 3, text: "explain", commitment: "StableText" },
          { id: 4, text: "that", commitment: "Hypothetical" },
        ],
      },
    }),
    mkEvent("transcript_candidate", 1, 180, {
      artifact: { stable_text: "Can you explain what overlap routing means", unstable_text: "?" },
    }),
    mkEvent("transcript", 1, 210, { text: "Can you explain what overlap routing means?" }),
    mkEvent("first_safe_speech_unit_emitted", 1, 260, {
      text: "Sure. Overlap routing decides whether Pete yields.",
    }),
    mkEvent("speculative_speech_updated", 1, 300, {
      text: "Sure. Overlap routing decides whether Pete yields when interruption arrives.",
    }),
    mkEvent("interruption_detected", 1, 320),
    mkEvent("speech_unit_cancelled", 1, 330, {
      text: "Sure. Overlap routing decides whether Pete yields when interruption arrives.",
    }),
    mkEvent("tts_timed_word_stream_revision", 1, 331, {
      reason: "cancelled for interruption",
      artifact: {
        words: [
          { text: "Sure.", commitment: "Cancelled" },
          { text: "Overlap", commitment: "Cancelled" },
        ],
      },
    }),
    mkEvent("transcript", 2, 500, { text: "I miss how he used to sound." }),
    mkEvent("speech_unit_committed", 2, 560, { text: "I do too. We can stay with that for a minute." }),
  ];

  for (const event of events) {
    reduceNarrativeEvent(session, event);
  }
  return session;
}

function buildWaveDeckEpisode(number = 1) {
  return buildNarrativeEpisode(buildWaveDeckFixtureSession(), { episodeNumber: number });
}

test("narrative model segments beats and scenes with revisions, cancellation, and topic shift", () => {
  const session = buildWaveDeckFixtureSession();
  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });

  assert.equal(episode.scenes.length, 2, "topic shift should split into two scenes");

  // Headings are now proper screenplay sluglines, not runtime labels
  assert.match(episode.scenes[0].heading, /^INT\. UNKNOWN ROOM - DAY$/);
  assert.match(episode.scenes[1].heading, /^INT\. UNKNOWN ROOM - DAY$/);

  // Topic labels are preserved separately for metadata/soft notes
  assert.equal(episode.scenes[0].topicLabel, "WAVEDECK INSPECTION");
  assert.equal(episode.scenes[1].topicLabel, "QUIET GRIEF");

  assert.ok(episode.scenes[0].beats.some((beat) => beat.kind === "transcript_revision"));
  assert.ok(episode.scenes[0].beats.some((beat) => beat.kind === "interruption"));
  assert.ok(episode.scenes[0].beats.some((beat) => beat.kind === "cancellation"));
  assert.ok(episode.scenes[0].sourceEventIds.some((id) => id.startsWith("turn-turn:1:transcript_candidate:140")));
  assert.ok(episode.sceneList.every((scene) => scene.summary.length > 0), "scene summaries should be present");
  assert.match(episode.screenplayBody, /UNKNOWN VOICE #1\nCan you explain what overlap routing means\?/);
  assert.match(episode.screenplayBody, /PETE\nSure\. Overlap routing decides whether Pete yields/);
  assert.ok(!episode.screenplayBody.includes("\nUSER\n"), "legacy USER cue must never render");
  assert.ok(!episode.screenplayBody.includes("\nASSISTANT\n"), "legacy ASSISTANT cue must never render");
});

test("scene boundaries revise retroactively as more context arrives", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript", 1, 100, { text: "Can you explain overlap routing?" }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 1, 160, { text: "Yes. It decides when Pete yields." }));

  const earlyEpisode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  assert.equal(earlyEpisode.scenes.length, 1, "single topic should remain one scene");

  reduceNarrativeEvent(session, mkEvent("transcript", 2, 300, { text: "I miss how he used to sound." }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 2, 340, { text: "I do too." }));

  const revisedEpisode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  assert.equal(revisedEpisode.scenes.length, 2, "later topic shift should retroactively split the episode");
  assert.match(revisedEpisode.summary, /WAVEDECK INSPECTION → QUIET GRIEF/);
});

test("transcript propositions stay available without becoming screenplay beats", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript_proposition", 1, 100, { text: "hello worl" }));
  reduceNarrativeEvent(session, mkEvent("transcript_proposition", 1, 120, { text: "hello world" }));

  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  assert.equal(session.proposition.text, "hello world");
  assert.deepEqual(
    session.propositionDeleted.map((entry) => entry.text),
    ["worl"],
  );
  assert.equal(episode.scenes.length, 0);
});

test("episodes assemble into chapters and manuscript structure", () => {
  const episode1 = buildWaveDeckEpisode(1);
  const episode2 = buildWaveDeckEpisode(2);
  const manuscript = assembleNarrativeManuscript([episode1, episode2], { title: "Listenbury Manuscript" });

  assert.equal(manuscript.title, "Listenbury Manuscript");
  assert.equal(manuscript.episodes.length, 2);
  assert.equal(manuscript.chapters.length, 1, "matching episode arcs should share a chapter");
  assert.equal(manuscript.chapters[0].episodes.length, 2);
  assert.equal(manuscript.children[0].type, "chapter");
});

test("turn and speech-unit IDs are used when present", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, {
    kind: "transcript",
    turn: 1,
    turn_id: 55,
    elapsed_ms: 100,
    text: "Can you hear me?",
  });
  reduceNarrativeEvent(session, {
    kind: "speech_unit_committed",
    turn: 1,
    turn_id: 55,
    speech_unit_id: 4001,
    elapsed_ms: 140,
    text: "Yes, clearly.",
  });
  reduceNarrativeEvent(session, {
    kind: "speech_unit_cancelled",
    turn: 1,
    turn_id: 55,
    speech_unit_id: 4001,
    elapsed_ms: 150,
  });

  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  const beatKinds = episode.scenes.flatMap((scene) => scene.beats.map((beat) => beat.kind));
  assert.ok(
    beatKinds.includes("cancellation"),
    "speech-unit cancellation still keys into turn when text is omitted",
  );
  assert.ok(
    episode.sourceEventIds.some((id) => id.startsWith("turn-tid:55:speech_unit_committed")),
    "source ids are keyed by turn_id when available",
  );
});

test("dialogue segments optionally carry span metadata", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("asr_timed_word_stream", 1, 100, {
    artifact: {
      words: [
        { id: 1, span_id: 101, text: "hello", commitment: "StableText", timing: { start_ms: 100, end_ms: 170 } },
      ],
    },
  }));
  reduceNarrativeEvent(session, mkEvent("tts_timed_word_stream_revision", 1, 200, {
    artifact: {
      words: [
        { id: 2, span_id: 202, text: "hi", commitment: "Played", timing: { start_ms: 240, end_ms: 300 } },
      ],
    },
  }));

  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  const userBeat = episode.scenes[0].beats.find((beat) => beat.kind === "voice_dialogue");
  const llmBeat = episode.scenes[0].beats.find((beat) => beat.kind === "llm_dialogue");
  assert.ok(userBeat.segments[0].spanMetadata?.length, "user segment should include optional span metadata");
  assert.ok(llmBeat.segments[0].spanMetadata?.length, "llm segment should include optional span metadata");
});

// ──────────────────────────────────────────────────────────────────────────────
// Acceptance tests: realistic scene headings (issue requirements)
// ──────────────────────────────────────────────────────────────────────────────

test("scene heading is a proper screenplay slugline, not a runtime label", () => {
  const episode = buildWaveDeckEpisode(1);
  for (const scene of episode.scenes) {
    // Must start with INT., EXT., or INT./EXT.
    assert.match(scene.heading, /^(?:INT\.|EXT\.|INT\.\/EXT\.) /, `scene heading should start with INT./EXT.: ${scene.heading}`);
    // Must contain a time-of-day separator with a valid label
    assert.match(scene.heading, / - (?:DAY|NIGHT|AFTERNOON|EVENING|PRESENT)$/, `scene heading should end with a valid time-of-day label: ${scene.heading}`);
    // Must NOT contain runtime labels
    assert.ok(!scene.heading.includes("LISTENBURY RUNTIME"), `slugline must not include LISTENBURY RUNTIME: ${scene.heading}`);
    assert.ok(!scene.heading.includes("QUIET GRIEF"), `mood label must not appear in slugline: ${scene.heading}`);
    assert.ok(!scene.heading.includes("PHONOLOGY WORKBENCH"), `topic label must not appear in slugline: ${scene.heading}`);
    assert.ok(!scene.heading.includes("WAVEDECK INSPECTION"), `topic label must not appear in slugline: ${scene.heading}`);
    assert.ok(!scene.heading.includes("PRESENT"), `"PRESENT" must not appear in slugline: ${scene.heading}`);
  }
});

test("topic label is preserved as scene metadata, not used as location", () => {
  const episode = buildWaveDeckEpisode(1);
  const [sceneA, sceneB] = episode.scenes;
  // Topic labels preserved separately
  assert.equal(sceneA.topicLabel, "WAVEDECK INSPECTION");
  assert.equal(sceneB.topicLabel, "QUIET GRIEF");
  // Topic labels appear in action as soft note, not in the slugline
  assert.ok(sceneA.action.includes("Topic:"), "topic soft note should appear in action text");
  // Sluglines are proper physical locations
  assert.match(sceneA.heading, /^INT\. UNKNOWN ROOM - DAY$/);
  assert.match(sceneB.heading, /^INT\. UNKNOWN ROOM - DAY$/);
});

test("emotional/mood label is not used as slugline location", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript", 1, 100, { text: "I miss how he used to sound." }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 1, 140, { text: "I do too. We can stay with that." }));

  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  const [scene] = episode.scenes;

  // Heading must be a physical location slugline
  assert.match(scene.heading, /^INT\. UNKNOWN ROOM - DAY$/);
  // Mood label "QUIET GRIEF" should not appear in the heading
  assert.ok(!scene.heading.includes("GRIEF"), `mood label must not appear in slugline: ${scene.heading}`);
  // But the topic label is preserved in topicLabel
  assert.equal(scene.topicLabel, "QUIET GRIEF");
});

test("unknown room fallback is used when no location context is available", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript", 1, 100, { text: "Can you hear me?" }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 1, 140, { text: "Yes, clearly." }));

  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  assert.equal(episode.scenes[0].heading, "INT. UNKNOWN ROOM - DAY");
});

test("scene heading uses explicit place when locationContext is provided", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript", 1, 100, { text: "Can you hear me?" }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 1, 140, { text: "Yes, clearly." }));

  const episode = buildNarrativeEpisode(session, {
    episodeNumber: 1,
    locationContext: { place: "living room", interiorExterior: "INT.", timeOfDay: "NIGHT" },
  });
  assert.equal(episode.scenes[0].heading, "INT. LIVING ROOM - NIGHT");
});

test("scene heading uses vision-derived location when provided", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript", 1, 100, { text: "Can you hear me?" }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 1, 140, { text: "Yes, clearly." }));

  const episodeOutdoor = buildNarrativeEpisode(session, {
    episodeNumber: 1,
    locationContext: { vision: ["grass", "trees", "bench"], timeOfDay: "DAY" },
  });
  assert.equal(episodeOutdoor.scenes[0].heading, "EXT. PARK - DAY");

  const episodeBedroom = buildNarrativeEpisode(session, {
    episodeNumber: 1,
    locationContext: { vision: ["bed", "nightstand"], timeOfDay: "NIGHT" },
  });
  assert.equal(episodeBedroom.scenes[0].heading, "INT. BEDROOM - NIGHT");
});

test("scene-list headings match scene headings", () => {
  const episode = buildWaveDeckEpisode(1);
  for (let i = 0; i < episode.scenes.length; i++) {
    assert.equal(
      episode.sceneList[i].heading,
      episode.scenes[i].heading,
      `scene-list heading at index ${i} must match scene heading`,
    );
  }
});

test("rendered screenplay body includes topic as soft note, not in slugline", () => {
  const episode = buildWaveDeckEpisode(1);
  // The screenplay body should include the topic as a soft note
  assert.match(episode.screenplayBody, /Soft-note: Topic: Wavedeck Inspection\./);
  assert.match(episode.screenplayBody, /Soft-note: Topic: Quiet Grief\./);
  // The sluglines should be physical locations
  assert.match(episode.screenplayBody, /INT\. UNKNOWN ROOM - DAY/);
  // Runtime labels should NOT appear as sluglines
  assert.ok(!episode.screenplayBody.includes("INT. LISTENBURY RUNTIME"), "runtime label must not appear as slugline in body");
});

test("unknown voice ordinals are stable and speaker cues stay voice-oriented", () => {
  const session = createNarrativeSession();
  reduceNarrativeEvent(session, mkEvent("transcript", 1, 100, { text: "first", voice_id: "speaker-a", voice_label: "unknown" }));
  reduceNarrativeEvent(session, mkEvent("transcript", 2, 200, { text: "second", voice_id: "speaker-b", voice_label: "unknown" }));
  reduceNarrativeEvent(session, mkEvent("transcript", 3, 300, { text: "third", voice_id: "speaker-a", voice_label: "unknown" }));
  reduceNarrativeEvent(session, mkEvent("speech_unit_committed", 3, 340, { text: "acknowledged" }));

  const episode = buildNarrativeEpisode(session, { episodeNumber: 1 });
  const cues = episode.scenes.flatMap((scene) => scene.beats.map((beat) => beat.role).filter(Boolean));
  assert.deepEqual(cues.filter((cue) => cue.startsWith("UNKNOWN VOICE")), ["UNKNOWN VOICE #1", "UNKNOWN VOICE #2", "UNKNOWN VOICE #1"]);
  assert.ok(cues.includes("PETE"));
  assert.ok(!cues.includes("USER"));
  assert.ok(!cues.includes("ASSISTANT"));
});
