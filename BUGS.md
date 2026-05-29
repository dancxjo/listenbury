# BUGS

This is Listenbury's living bug and feature-request ledger. Pete's `go` runtime can append to this file through `reportBug(...)`, `reportFeatureRequest(...)`, or `reportIssue(...)`; humans and coding agents should also add small, concrete entries here when they notice rough edges that are not yet worth a full GitHub issue.

Keep entries short, actionable, and dated when possible. Prefer preserving buglets over making them perfect.

## Open bugs

### Bug: `go` Harmony rendering/parsing is hand-rolled and fragile

- **Severity:** high
- **Area:** `src/cli/commands/go.rs`, Harmony prompt/render/filter path
- **Context:** `go` currently detects gpt-oss/Harmony mode and then parses generated channel/tool markers with local string matching. It recognizes many token fragments and fused marker patterns, but this is still a custom parser for a protocol with subtle boundary rules.
- **Risk:** A small model-template change, tokenizer rendering difference, or malformed partial token stream may cause analysis text, final text, or commentary tool calls to leak into the wrong path.
- **Suggested fix:** Isolate Harmony rendering/parsing into a dedicated module with golden tests, then replace as much as possible with the official `openai-harmony` crate or a thin adapter around it.

### Bug: `go` has no explicit prompt-format override

- **Severity:** medium
- **Area:** `GoConfig`, `go_prompt_format_for_model`, CLI flags
- **Context:** `GoConfig::from_command` initializes `prompt_format` as `PlainStream`, and `StreamOfConsciousness::start` switches it based on model path. That is convenient, but awkward when debugging template/model mismatches.
- **Risk:** A model with an unexpected filename may silently use the wrong prompt protocol. A gpt-oss model renamed without `gpt-oss` in its path could run in plain-stream mode.
- **Suggested fix:** Add `--harmony`, `--plain-stream`, and possibly `--auto-prompt-format` flags, with auto as the default.

### Bug: `go` command file is doing too much

- **Severity:** medium
- **Area:** `src/cli/commands/go.rs`
- **Context:** The file now contains CLI runtime orchestration, Harmony parsing, TypeScript action parsing, source-inspection workflow policy, mouth runtime, memory/RAG setup, work-board persistence, tests, and prompt text.
- **Risk:** Future changes will become harder to review, and unrelated fixes may collide in one large file.
- **Suggested fix:** Split into modules such as `go/config.rs`, `go/runtime.rs`, `go/harmony.rs`, `go/actions.rs`, `go/mouth.rs`, `go/work_board.rs`, `go/memory.rs`, and `go/source_gate.rs`.

### Bug: `shutup`, `pause`, and `resume` report success-ish messages but do not control playback yet

- **Severity:** medium
- **Area:** `TypeScriptAction::Shutup`, `Pause`, `Resume`; `MouthRuntime`
- **Context:** The action handlers append observations saying that queued-speech clearing and TTS pause control are not implemented.
- **Risk:** Pete may believe it can stop or pause speech when it cannot, which is especially awkward during interruption/barge-in tests.
- **Suggested fix:** Add explicit `MouthCommand::Shutup`, `Pause`, and `Resume` variants, clear queued speech where possible, and plumb cancellation/pause semantics into the active TTS/playback path.

### Bug: mock mouth self-hearing is unrealistically immediate

- **Severity:** low
- **Area:** `run_mock_mouth`
- **Context:** Mock mouth returns a self-heard event after a fixed 20 ms sleep regardless of utterance length.
- **Risk:** Pacing behavior can look healthy in mock runs while real speech blocks generation much longer.
- **Suggested fix:** Scale mock return latency by text length or estimated syllable/audio duration, while keeping a fast-test override.

### Bug: startup best-effort IP geolocation can add network latency and privacy surprise

- **Severity:** low
- **Area:** `gather_startup_context`, `best_effort_ip_location`
- **Context:** `go` calls `https://ipapi.co/json/` with a 900 ms timeout during startup.
- **Risk:** Startup can pause on network lookup, leak public-IP lookup intent, or produce noisy context when offline.
- **Suggested fix:** Gate this behind a flag or environment variable, cache it, and clearly mark it as optional external context.

### Bug: source-inspection workflow enforcement became advisory and may allow shallow browsing loops

- **Severity:** medium
- **Area:** source inspection reminders in `apply_actions`
- **Context:** Earlier behavior blocked source inspection until progress notes/synthesis were recorded. Current behavior appears to encourage notes and synthesis but continue anyway.
- **Risk:** Pete may slip back into source-page hopping without durable compression, especially during autonomous runs.
- **Suggested fix:** Decide whether this should be hard-gated or advisory by mode. Consider `--strict-source-gate` for autonomous code-reading sessions.

### Bug: `reportFeatureRequest` maps to `report_issue` but the payload type may still default to `bug` in some paths

- **Severity:** medium
- **Area:** Harmony tool-call mapping, TypeScript `reportIssue`/`reportFeatureRequest`, `append_issue_report`
- **Context:** Harmony mapping inserts `issue_type = feature_request` for `report_feature_request`, while TypeScript `report_feature_request` calls `report_issue_command(..., "feature_request")`. This is good, but the code has several alias paths and normalization steps.
- **Risk:** Feature requests may occasionally be logged as bugs if an alias bypasses the intended issue type field.
- **Suggested fix:** Add focused tests for `reportBug`, `reportFeatureRequest`, `reportIssue({ type })`, and Harmony `functions.report_feature_request`.

### Bug: graph-node update requires non-empty fields, making label-only merges impossible

- **Severity:** low
- **Area:** `TypeScriptActionPayload::UpdateGraphNodeFields` parsing
- **Context:** The parser drops graph-node updates when `fields` is empty, even if a useful `node_id` and `label` were provided.
- **Risk:** Pete cannot create or pin a named node with only an ID/label, which may be a natural first step during entity stabilization.
- **Suggested fix:** Permit label-only merges, or add a dedicated `createGraphNode`/`pinGraphNode` action.

### Bug: Harmony tool-call recipient parsing only accepts `to=functions.*`

- **Severity:** low
- **Area:** `harmony_tool_recipient`
- **Context:** Recipient extraction searches for `to=functions.` explicitly.
- **Risk:** If future Harmony rendering uses a slightly different namespace spelling, spacing, or recipient form, tool calls may be rejected even when semantically valid.
- **Suggested fix:** Keep `functions.*` as the supported namespace, but parse recipient syntax with the official Harmony structures or stricter tested grammar rather than substring matching.

## Feature requests

### Feature request: Add a first-class bug-ledger workflow around `BUGS.md`

- **Severity:** low
- **Area:** `reportBug`, `reportFeatureRequest`, docs/agent workflow
- **Context:** The new TypeScript issue-reporting commands are exactly the right little valve for capturing buglets without interrupting flow.
- **Suggested implementation:** Add tests ensuring reports append stable Markdown sections, preserve existing file content, and include title, type, severity, details, context, and timestamp.

### Feature request: Emit bug reports into both `BUGS.md` and durable memory

- **Severity:** low
- **Area:** `append_issue_report`, memory sink
- **Context:** `BUGS.md` is convenient for humans and coding agents, while Pete's memory/RAG system would benefit from remembering recurring bug patterns.
- **Suggested implementation:** When `ReportIssue` succeeds, submit a `MemoryTrace` or private note containing the same issue summary.
