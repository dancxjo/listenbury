# Follow-up tickets: unified cross-modal text/span architecture

This epic is intentionally sliced into focused follow-up tickets:

- [ ] **SPAN-101: Audio/phoneme/word live ingestor**
  - Stream provisional spans keyed by `TextId`.
  - Emit stabilization/commit events with trace IDs.
- [ ] **SPAN-102: Clause, breath-group, and prosody segmenters**
  - Build higher-order spans from lower-order alignments.
  - Attach confidence and overlap metadata.
- [ ] **SPAN-103: Semantic/topic/episode layering**
  - Derive semantic and discourse spans from committed clauses.
  - Support delayed rescans and semantic-guided repair.
- [ ] **SPAN-104: Memory découpage and long-horizon alignment**
  - Persist memory spans linked to topic/episode spans.
  - Add retrieval hooks by span lineage.
- [ ] **SPAN-105: Revision/event log integration**
  - Emit append-only span revision events for UI and replay.
  - Preserve revision chains for auditability.
- [ ] **SPAN-106: Cross-modal viewer lanes**
  - Render aligned span lanes in WaveDeck with path tracing.
  - Visualize provisional -> committed -> revised transitions.
