use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

use crate::memory::embed::EmbeddingProvider;
use crate::memory::neo4j::{trace_write_for, Neo4jStore, Neo4jWriteResult};
use crate::memory::qdrant::{vector_documents_for_trace, QdrantStore, DEFAULT_QDRANT_COLLECTION};
use crate::memory::sink::ChannelMemorySink;
use crate::memory::trace::MemoryTrace;

/// Cold-memory worker configuration.
#[derive(Clone)]
pub struct ColdMemoryWorkerConfig {
    pub neo4j: Option<Arc<dyn Neo4jStore>>,
    pub qdrant: Option<Arc<dyn QdrantStore>>,
    pub embeddings: Option<Arc<dyn EmbeddingProvider>>,
    pub qdrant_collection: String,
}

impl Default for ColdMemoryWorkerConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ColdMemoryWorkerConfig {
    pub fn new() -> Self {
        Self {
            neo4j: None,
            qdrant: None,
            embeddings: None,
            qdrant_collection: DEFAULT_QDRANT_COLLECTION.to_string(),
        }
    }
}

/// Summary of work performed by a background cold-memory worker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ColdMemoryWorkerReport {
    pub traces_processed: usize,
    pub graph_writes_ok: usize,
    pub graph_writes_failed: usize,
    pub vector_upserts_ok: usize,
    pub vector_upserts_failed: usize,
    pub embedding_failures: usize,
}

/// Background worker that drains [`MemoryTrace`] values from a channel.
pub struct ColdMemoryWorker {
    join: Option<JoinHandle<ColdMemoryWorkerReport>>,
}

impl ColdMemoryWorker {
    /// Spawn a worker over an existing receiver.
    pub fn spawn(receiver: mpsc::Receiver<MemoryTrace>, config: ColdMemoryWorkerConfig) -> Self {
        let join = thread::spawn(move || run_worker(receiver, config));
        Self { join: Some(join) }
    }

    /// Create a non-blocking [`ChannelMemorySink`] and a matching worker.
    pub fn spawn_channel(
        capacity: usize,
        config: ColdMemoryWorkerConfig,
    ) -> (ChannelMemorySink, Self) {
        let (sink, receiver) = ChannelMemorySink::new(capacity);
        (sink, Self::spawn(receiver, config))
    }

    /// Wait for the worker to finish after all senders are dropped.
    pub fn join(mut self) -> thread::Result<ColdMemoryWorkerReport> {
        self.join
            .take()
            .expect("cold-memory worker already joined")
            .join()
    }
}

fn run_worker(
    receiver: mpsc::Receiver<MemoryTrace>,
    config: ColdMemoryWorkerConfig,
) -> ColdMemoryWorkerReport {
    let mut report = ColdMemoryWorkerReport::default();

    for (sequence, trace) in receiver.into_iter().enumerate() {
        report.traces_processed += 1;
        process_trace(sequence as u64, &trace, &config, &mut report);
    }

    report
}

fn process_trace(
    sequence: u64,
    trace: &MemoryTrace,
    config: &ColdMemoryWorkerConfig,
    report: &mut ColdMemoryWorkerReport,
) {
    let mut graph_result = Neo4jWriteResult::default();

    if let Some(graph_store) = config.neo4j.as_ref() {
        let write = trace_write_for(trace, sequence);
        match graph_store.store_trace(write) {
            Ok(result) => {
                report.graph_writes_ok += 1;
                graph_result = result;
            }
            Err(error) => {
                report.graph_writes_failed += 1;
                tracing::warn!("cold-memory graph write failed: {error:#}");
            }
        }
    }

    let Some(vector_store) = config.qdrant.as_ref() else {
        return;
    };
    let Some(embedding_provider) = config.embeddings.as_ref() else {
        tracing::warn!("cold-memory vector write skipped: no embedding provider configured");
        return;
    };

    let mut points = Vec::new();
    for document in vector_documents_for_trace(trace, sequence, &graph_result) {
        match embedding_provider.embed(&document.text) {
            Ok(vector) => points.push(document.into_point(vector)),
            Err(error) => {
                report.embedding_failures += 1;
                tracing::warn!("cold-memory embedding failed: {error:#}");
            }
        }
    }

    if points.is_empty() {
        return;
    }

    match vector_store.upsert_points(&config.qdrant_collection, &points) {
        Ok(()) => report.vector_upserts_ok += points.len(),
        Err(error) => {
            report.vector_upserts_failed += points.len();
            tracing::warn!("cold-memory vector upsert failed: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::sync::Mutex;

    use anyhow::anyhow;

    use super::*;
    use crate::memory::neo4j::{Neo4jTraceWrite, Neo4jWriteResult};
    use crate::memory::qdrant::{QdrantPoint, QdrantSearchHit};
    use crate::memory::sink::MemorySink as _;
    use crate::memory::trace::SpeakerRole;
    use crate::time::ExactTimestamp;

    #[derive(Default)]
    struct RecordingGraphStore {
        writes: Mutex<Vec<Neo4jTraceWrite>>,
        fail: bool,
    }

    impl Neo4jStore for RecordingGraphStore {
        fn store_trace(&self, write: Neo4jTraceWrite) -> anyhow::Result<Neo4jWriteResult> {
            self.writes
                .lock()
                .expect("graph mutex poisoned")
                .push(write.clone());
            if self.fail {
                anyhow::bail!("neo4j unavailable");
            }
            Ok(Neo4jWriteResult {
                primary_node_id: Some(format!("neo4j::{}", write.primary_node.logical_id)),
                related_node_ids: write
                    .related_nodes
                    .iter()
                    .map(|node| format!("neo4j::{}", node.logical_id))
                    .collect(),
            })
        }
    }

    #[derive(Default)]
    struct RecordingQdrantStore {
        upserts: Mutex<Vec<(String, Vec<QdrantPoint>)>>,
        fail: bool,
    }

    impl QdrantStore for RecordingQdrantStore {
        fn upsert_points(&self, collection: &str, points: &[QdrantPoint]) -> anyhow::Result<()> {
            self.upserts
                .lock()
                .expect("qdrant mutex poisoned")
                .push((collection.to_string(), points.to_vec()));
            if self.fail {
                anyhow::bail!("qdrant unavailable");
            }
            Ok(())
        }

        fn search(
            &self,
            collection: &str,
            query_vector: &[f32],
            limit: usize,
        ) -> anyhow::Result<Vec<QdrantSearchHit>> {
            let upserts = self.upserts.lock().expect("qdrant mutex poisoned");
            let mut hits = upserts
                .iter()
                .filter(|(stored_collection, _)| stored_collection == collection)
                .flat_map(|(_, points)| {
                    points.iter().map(|point| QdrantSearchHit {
                        id: point.id.clone(),
                        score: cosine_similarity(&point.vector, query_vector),
                        payload: point.payload.clone(),
                    })
                })
                .collect::<Vec<_>>();
            hits.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(Ordering::Equal)
            });
            hits.truncate(limit);
            Ok(hits)
        }
    }

    struct StubEmbeddingProvider {
        fail: bool,
    }

    impl EmbeddingProvider for StubEmbeddingProvider {
        fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            if self.fail {
                return Err(anyhow!("embedding failed for {text}"));
            }
            Ok(vec![text.len() as f32, 1.0, 0.5])
        }
    }

    #[test]
    fn graph_only_success() {
        let graph = Arc::new(RecordingGraphStore::default());
        let mut config = ColdMemoryWorkerConfig::new();
        config.neo4j = Some(graph.clone());

        let (sink, worker) = ColdMemoryWorker::spawn_channel(8, config);
        sink.submit(sample_turn("Can you hear me?"));
        drop(sink);

        let report = worker.join().expect("worker should join");
        assert_eq!(report.traces_processed, 1);
        assert_eq!(report.graph_writes_ok, 1);
        assert_eq!(report.vector_upserts_ok, 0);

        let writes = graph.writes.lock().expect("graph mutex poisoned");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].primary_node.label, "ConversationTurn");
    }

    #[test]
    fn graph_and_vector_success() {
        let graph = Arc::new(RecordingGraphStore::default());
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let embeddings = Arc::new(StubEmbeddingProvider { fail: false });

        let mut config = ColdMemoryWorkerConfig::new();
        config.neo4j = Some(graph);
        config.qdrant = Some(qdrant.clone());
        config.embeddings = Some(embeddings);

        let (sink, worker) = ColdMemoryWorker::spawn_channel(8, config);
        sink.submit(sample_turn("Can you hear me?"));
        drop(sink);

        let report = worker.join().expect("worker should join");
        assert_eq!(report.graph_writes_ok, 1);
        assert_eq!(report.vector_upserts_ok, 1);
        assert_eq!(report.embedding_failures, 0);

        let hits = qdrant
            .search(DEFAULT_QDRANT_COLLECTION, &[13.0, 1.0, 0.5], 1)
            .expect("search should succeed");
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0]
                .payload
                .get("neo4j_node_id")
                .and_then(|value| value.as_str()),
            Some("neo4j::conversation_turn:0")
        );
    }

    #[test]
    fn embedding_failure_keeps_graph_write() {
        let graph = Arc::new(RecordingGraphStore::default());
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let embeddings = Arc::new(StubEmbeddingProvider { fail: true });

        let mut config = ColdMemoryWorkerConfig::new();
        config.neo4j = Some(graph.clone());
        config.qdrant = Some(qdrant.clone());
        config.embeddings = Some(embeddings);

        let (sink, worker) = ColdMemoryWorker::spawn_channel(8, config);
        sink.submit(sample_turn("still store graph"));
        drop(sink);

        let report = worker.join().expect("worker should join");
        assert_eq!(report.graph_writes_ok, 1);
        assert_eq!(report.embedding_failures, 1);
        assert_eq!(report.vector_upserts_ok, 0);

        assert_eq!(graph.writes.lock().expect("graph mutex poisoned").len(), 1);
        assert!(qdrant
            .upserts
            .lock()
            .expect("qdrant mutex poisoned")
            .is_empty());
    }

    #[test]
    fn database_unavailability_is_non_fatal() {
        let graph = Arc::new(RecordingGraphStore {
            writes: Mutex::default(),
            fail: true,
        });
        let qdrant = Arc::new(RecordingQdrantStore {
            upserts: Mutex::default(),
            fail: true,
        });
        let embeddings = Arc::new(StubEmbeddingProvider { fail: false });

        let mut config = ColdMemoryWorkerConfig::new();
        config.neo4j = Some(graph);
        config.qdrant = Some(qdrant);
        config.embeddings = Some(embeddings);

        let (sink, worker) = ColdMemoryWorker::spawn_channel(8, config);
        sink.submit(sample_turn("first"));
        sink.submit(sample_turn("second"));
        drop(sink);

        let report = worker.join().expect("worker should join");
        assert_eq!(report.traces_processed, 2);
        assert_eq!(report.graph_writes_failed, 2);
        assert_eq!(report.vector_upserts_failed, 2);
    }

    #[test]
    fn trace_ingestion_ordering_is_preserved() {
        let graph = Arc::new(RecordingGraphStore::default());
        let mut config = ColdMemoryWorkerConfig::new();
        config.neo4j = Some(graph.clone());

        let (sink, worker) = ColdMemoryWorker::spawn_channel(8, config);
        sink.submit(sample_turn("first"));
        sink.submit(sample_turn("second"));
        sink.submit(sample_turn("third"));
        drop(sink);

        let report = worker.join().expect("worker should join");
        assert_eq!(report.traces_processed, 3);

        let writes = graph.writes.lock().expect("graph mutex poisoned");
        let texts = writes
            .iter()
            .map(|write| {
                write
                    .primary_node
                    .properties
                    .get("text")
                    .and_then(|value| value.as_str())
                    .expect("conversation turn text")
                    .to_string()
            })
            .collect::<Vec<_>>();
        assert_eq!(texts, vec!["first", "second", "third"]);
    }

    fn sample_turn(text: &str) -> MemoryTrace {
        MemoryTrace::ConversationTurnFinalized {
            speaker: SpeakerRole::User,
            text: text.to_string(),
            occurred_at: ExactTimestamp::now(),
        }
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        let dot = left
            .iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>();
        let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
        if left_norm == 0.0 || right_norm == 0.0 {
            return 0.0;
        }
        dot / (left_norm * right_norm)
    }
}
