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
  assert.match(episode.scenes[0].heading, /WAVEDECK INSPECTION/);
  assert.match(episode.scenes[1].heading, /QUIET GRIEF/);
  assert.ok(episode.scenes[0].beats.some((beat) => beat.kind === "transcript_revision"));
  assert.ok(episode.scenes[0].beats.some((beat) => beat.kind === "interruption"));
  assert.ok(episode.scenes[0].beats.some((beat) => beat.kind === "cancellation"));
  assert.ok(episode.scenes[0].sourceEventIds.some((id) => id.startsWith("turn-1:transcript_candidate:140")));
  assert.ok(episode.sceneList.every((scene) => scene.summary.length > 0), "scene summaries should be present");
  assert.match(episode.screenplayBody, /USER\nCan you explain what overlap routing means\?/);
  assert.match(episode.screenplayBody, /PETE\nSure\. Overlap routing decides whether Pete yields/);
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
