use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use anyhow::{Context, anyhow};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::memory::qdrant::{QdrantPoint, QdrantSearchHit, QdrantStore};
use crate::soundscape::{
    EmbeddingRef, EnrollmentSource, KnownVoice, KnownVoiceRegistry, VoiceAttribution,
    VoiceAttributionAlternative, VoiceAttributionSource, VoiceEnrollmentSample,
    VoiceEnrollmentSampleId, VoiceEntityAssociation, VoiceEntityAssociationSource, VoiceId,
    VoiceMatcher, VoiceSignatureId,
};
use crate::{span::SpanId, time::ExactTimestamp};

pub const DEFAULT_KNOWN_VOICE_REGISTRY_PATH: &str = "listenbury_data/memory/known_voices.json";
pub const KNOWN_VOICE_QDRANT_COLLECTION: &str = "listenbury_known_voice_enrollments";
pub const KNOWN_VOICE_EMBEDDING_BACKEND: &str = "qdrant";
pub const KNOWN_VOICE_LOCALITY: &str = "local_only";

pub trait KnownVoiceEmbeddingProvider: Send + Sync {
    fn embed_enrollment(
        &self,
        voice: &KnownVoice,
        sample: &VoiceEnrollmentSample,
    ) -> anyhow::Result<Vec<f32>>;
    fn embed_signature(&self, signature_id: VoiceSignatureId) -> anyhow::Result<Vec<f32>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicKnownVoiceEmbeddingProvider;

impl KnownVoiceEmbeddingProvider for DeterministicKnownVoiceEmbeddingProvider {
    fn embed_enrollment(
        &self,
        voice: &KnownVoice,
        sample: &VoiceEnrollmentSample,
    ) -> anyhow::Result<Vec<f32>> {
        Ok(seed_to_embedding(format!(
            "{}:{}:{}",
            voice.id.0, sample.id.0, sample.audio_span_id.0
        )))
    }

    fn embed_signature(&self, signature_id: VoiceSignatureId) -> anyhow::Result<Vec<f32>> {
        Ok(seed_to_embedding(signature_id.0.to_string()))
    }
}

#[derive(Clone)]
pub struct KnownVoiceMemoryStore {
    path: PathBuf,
    qdrant: Arc<dyn QdrantStore>,
    embeddings: Arc<dyn KnownVoiceEmbeddingProvider>,
    qdrant_collection: String,
}

impl KnownVoiceMemoryStore {
    pub fn new(
        path: impl AsRef<Path>,
        qdrant: Arc<dyn QdrantStore>,
        embeddings: Arc<dyn KnownVoiceEmbeddingProvider>,
    ) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            qdrant,
            embeddings,
            qdrant_collection: KNOWN_VOICE_QDRANT_COLLECTION.to_string(),
        }
    }

    pub fn with_collection(mut self, qdrant_collection: impl Into<String>) -> Self {
        self.qdrant_collection = qdrant_collection.into();
        self
    }

    pub fn save_registry(&self, registry: &KnownVoiceRegistry) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create known-voice directory {:?}", parent))?;
        }
        let payload =
            serde_json::to_vec_pretty(registry).context("serialize known-voice registry")?;
        fs::write(&self.path, payload)
            .with_context(|| format!("write known-voice registry {:?}", self.path))?;
        Ok(())
    }

    pub fn load_registry(&self) -> anyhow::Result<KnownVoiceRegistry> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(KnownVoiceRegistry::default());
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("read known-voice registry {:?}", self.path));
            }
        };
        serde_json::from_slice(&bytes).context("deserialize known-voice registry")
    }

    pub fn list_known_voices(&self) -> anyhow::Result<Vec<KnownVoice>> {
        Ok(self.load_registry()?.voices)
    }

    pub fn list_enrollment_samples(
        &self,
        voice_id: VoiceId,
    ) -> anyhow::Result<Vec<VoiceEnrollmentSample>> {
        let registry = self.load_registry()?;
        Ok(registry
            .enrollment_samples
            .into_iter()
            .filter(|sample| sample.voice_id == voice_id)
            .collect())
    }

    pub fn embed_and_store_enrollment_sample(
        &self,
        registry: &mut KnownVoiceRegistry,
        sample_id: VoiceEnrollmentSampleId,
        embedded_at: ExactTimestamp,
    ) -> anyhow::Result<EmbeddingRef> {
        let sample_index = registry
            .enrollment_samples
            .iter()
            .position(|sample| sample.id == sample_id)
            .ok_or_else(|| anyhow!("unknown enrollment sample {sample_id:?}"))?;
        let sample = registry.enrollment_samples[sample_index].clone();
        let voice = registry
            .voices
            .iter()
            .find(|voice| voice.id == sample.voice_id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown voice for enrollment sample {sample_id:?}"))?;

        let vector = self
            .embeddings
            .embed_enrollment(&voice, &sample)
            .context("embed enrollment sample")?;
        let key = known_voice_point_key(sample.voice_id, sample.id);

        let point = QdrantPoint {
            id: key.clone(),
            vector,
            payload: known_voice_payload(&voice, &sample, embedded_at),
        };
        self.qdrant
            .upsert_points(&self.qdrant_collection, &[point])
            .context("upsert known-voice enrollment point")?;

        let embedding_ref = EmbeddingRef {
            backend: KNOWN_VOICE_EMBEDDING_BACKEND.to_string(),
            key,
        };
        registry.enrollment_samples[sample_index].embedding_ref = Some(embedding_ref.clone());
        Ok(embedding_ref)
    }

    pub fn store_heard_voice_signature_vector(
        &self,
        signature_id: VoiceSignatureId,
        voice_node_id: &str,
        vector: Vec<f32>,
        observed_at: ExactTimestamp,
    ) -> anyhow::Result<EmbeddingRef> {
        let key = format!("heard_voice_signature:{}", signature_id.0);
        let point = QdrantPoint {
            id: key.clone(),
            vector,
            payload: heard_voice_signature_payload(signature_id, voice_node_id, observed_at),
        };
        self.qdrant
            .upsert_points(&self.qdrant_collection, &[point])
            .context("upsert heard voice signature point")?;
        Ok(EmbeddingRef {
            backend: KNOWN_VOICE_EMBEDDING_BACKEND.to_string(),
            key,
        })
    }

    pub fn associate_voice_with_entity(
        &self,
        registry: &mut KnownVoiceRegistry,
        voice_id: VoiceId,
        entity_node_id: impl Into<String>,
        entity_label: Option<String>,
        confidence: f32,
        source: VoiceEntityAssociationSource,
        associated_at: ExactTimestamp,
    ) -> anyhow::Result<VoiceEntityAssociation> {
        let association = registry.associate_voice_with_entity(
            voice_id,
            entity_node_id,
            entity_label,
            confidence,
            source,
            associated_at,
        );
        self.save_registry(registry)?;
        Ok(association)
    }
}

#[derive(Clone)]
pub struct QdrantKnownVoiceMatcher {
    qdrant: Arc<dyn QdrantStore>,
    embeddings: Arc<dyn KnownVoiceEmbeddingProvider>,
    qdrant_collection: String,
    min_confidence: f32,
    search_limit: usize,
    max_alternatives: usize,
}

impl QdrantKnownVoiceMatcher {
    pub fn new(
        qdrant: Arc<dyn QdrantStore>,
        embeddings: Arc<dyn KnownVoiceEmbeddingProvider>,
    ) -> Self {
        Self {
            qdrant,
            embeddings,
            qdrant_collection: KNOWN_VOICE_QDRANT_COLLECTION.to_string(),
            min_confidence: 0.5,
            search_limit: 8,
            max_alternatives: 3,
        }
    }

    pub fn with_collection(mut self, collection: impl Into<String>) -> Self {
        self.qdrant_collection = collection.into();
        self
    }

    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_search_limit(mut self, search_limit: usize) -> Self {
        self.search_limit = search_limit.max(1);
        self
    }

    pub fn with_max_alternatives(mut self, max_alternatives: usize) -> Self {
        self.max_alternatives = max_alternatives;
        self
    }
}

impl VoiceMatcher for QdrantKnownVoiceMatcher {
    fn attribute(
        &self,
        span_id: SpanId,
        signature_ids: &[VoiceSignatureId],
        registry: &KnownVoiceRegistry,
    ) -> Vec<VoiceAttribution> {
        if signature_ids.is_empty() || registry.voices.is_empty() {
            return Vec::new();
        }

        let mut best_by_voice = HashMap::<VoiceId, RankedVoiceCandidate>::new();
        for signature_id in signature_ids {
            let query = match self.embeddings.embed_signature(*signature_id) {
                Ok(query) => query,
                Err(error) => {
                    tracing::warn!("known-voice signature embedding failed: {error:#}");
                    continue;
                }
            };

            let hits = match self
                .qdrant
                .search(&self.qdrant_collection, &query, self.search_limit)
            {
                Ok(hits) => hits,
                Err(error) => {
                    tracing::warn!("known-voice search failed: {error:#}");
                    continue;
                }
            };

            for hit in hits {
                let Some(candidate) = ranked_candidate_from_hit(&hit, registry) else {
                    continue;
                };
                best_by_voice
                    .entry(candidate.voice_id)
                    .and_modify(|existing| {
                        if candidate.confidence > existing.confidence {
                            *existing = candidate;
                        }
                    })
                    .or_insert(candidate);
            }
        }

        if best_by_voice.is_empty() {
            return Vec::new();
        }

        let mut ranked = best_by_voice.into_values().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(Ordering::Equal)
        });

        let top = &ranked[0];
        if top.confidence < self.min_confidence {
            return Vec::new();
        }

        let alternatives = ranked
            .iter()
            .skip(1)
            .take(self.max_alternatives)
            .map(|candidate| VoiceAttributionAlternative {
                voice_id: candidate.voice_id,
                confidence: candidate.confidence,
                source: candidate.source,
            })
            .collect();

        vec![VoiceAttribution {
            voice_id: top.voice_id,
            span_id: Some(span_id),
            confidence: top.confidence,
            source: top.source,
            alternatives,
        }]
    }
}

#[derive(Debug, Clone, Copy)]
struct RankedVoiceCandidate {
    voice_id: VoiceId,
    confidence: f32,
    source: VoiceAttributionSource,
}

fn ranked_candidate_from_hit(
    hit: &QdrantSearchHit,
    registry: &KnownVoiceRegistry,
) -> Option<RankedVoiceCandidate> {
    let voice_id = hit
        .payload
        .get("voice_id")
        .and_then(Value::as_str)
        .and_then(|raw| Uuid::parse_str(raw).ok())
        .map(VoiceId)?;
    if !registry.voices.iter().any(|voice| voice.id == voice_id) {
        return None;
    }
    let source = hit
        .payload
        .get("source")
        .cloned()
        .and_then(|value| serde_json::from_value::<EnrollmentSource>(value).ok())
        .map(attribution_source_for_enrollment)
        .unwrap_or(VoiceAttributionSource::EnrollmentMatch);
    Some(RankedVoiceCandidate {
        voice_id,
        confidence: hit.score.clamp(0.0, 1.0),
        source,
    })
}

fn attribution_source_for_enrollment(source: EnrollmentSource) -> VoiceAttributionSource {
    match source {
        EnrollmentSource::GeneratedTts => VoiceAttributionSource::GeneratedTts,
        _ => VoiceAttributionSource::EnrollmentMatch,
    }
}

fn known_voice_payload(
    voice: &KnownVoice,
    sample: &VoiceEnrollmentSample,
    embedded_at: ExactTimestamp,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("kind".to_string(), json!("known_voice_enrollment"));
    payload.insert("voice_id".to_string(), json!(voice.id.0.to_string()));
    payload.insert("voice_label".to_string(), json!(voice.label));
    payload.insert("voice_kind".to_string(), json!(voice.kind));
    payload.insert(
        "enrollment_sample_id".to_string(),
        json!(sample.id.0.to_string()),
    );
    payload.insert("audio_span_id".to_string(), json!(sample.audio_span_id.0));
    payload.insert("source".to_string(), json!(sample.source));
    payload.insert("quality".to_string(), json!(sample.quality));
    payload.insert(
        "voice_created_at_unix_nanos".to_string(),
        json!(voice.created_at.unix_nanos),
    );
    payload.insert(
        "embedded_at_unix_nanos".to_string(),
        json!(embedded_at.unix_nanos),
    );
    payload.insert("memory_scope".to_string(), json!(KNOWN_VOICE_LOCALITY));
    payload
}

fn heard_voice_signature_payload(
    signature_id: VoiceSignatureId,
    voice_node_id: &str,
    observed_at: ExactTimestamp,
) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("kind".to_string(), json!("heard_voice_signature"));
    payload.insert(
        "voice_signature_id".to_string(),
        json!(signature_id.0.to_string()),
    );
    payload.insert("voice_node_id".to_string(), json!(voice_node_id));
    payload.insert(
        "observed_at_unix_nanos".to_string(),
        json!(observed_at.unix_nanos),
    );
    payload.insert("memory_scope".to_string(), json!(KNOWN_VOICE_LOCALITY));
    payload
}

fn known_voice_point_key(voice_id: VoiceId, sample_id: VoiceEnrollmentSampleId) -> String {
    format!("known_voice_enrollment:{}:{}", voice_id.0, sample_id.0)
}

fn seed_to_embedding(seed: String) -> Vec<f32> {
    let mut values = vec![0.0_f32; 16];
    let value_len = values.len();
    for (index, byte) in seed.as_bytes().iter().enumerate() {
        values[index % value_len] += *byte as f32;
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut values {
            *value /= norm;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundscape::{
        EnrollmentQuality, EnrollmentSource, KnownVoice, VoiceEntityAssociationSource, VoiceKind,
    };
    use tempfile::tempdir;

    #[derive(Default)]
    struct RecordingQdrantStore {
        upserts: Mutex<Vec<(String, Vec<QdrantPoint>)>>,
    }

    impl QdrantStore for RecordingQdrantStore {
        fn upsert_points(&self, collection: &str, points: &[QdrantPoint]) -> anyhow::Result<()> {
            self.upserts
                .lock()
                .expect("qdrant mutex poisoned")
                .push((collection.to_string(), points.to_vec()));
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
                        score: cosine_similarity(query_vector, &point.vector),
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

    #[derive(Default)]
    struct StubKnownVoiceEmbeddings {
        enrollment_vectors: Mutex<HashMap<VoiceEnrollmentSampleId, Vec<f32>>>,
        signature_vectors: Mutex<HashMap<VoiceSignatureId, Vec<f32>>>,
    }

    impl StubKnownVoiceEmbeddings {
        fn with_enrollment(self, sample_id: VoiceEnrollmentSampleId, vector: Vec<f32>) -> Self {
            self.enrollment_vectors
                .lock()
                .expect("enrollment vectors mutex poisoned")
                .insert(sample_id, vector);
            self
        }

        fn with_signature(self, signature_id: VoiceSignatureId, vector: Vec<f32>) -> Self {
            self.signature_vectors
                .lock()
                .expect("signature vectors mutex poisoned")
                .insert(signature_id, vector);
            self
        }
    }

    impl KnownVoiceEmbeddingProvider for StubKnownVoiceEmbeddings {
        fn embed_enrollment(
            &self,
            _voice: &KnownVoice,
            sample: &VoiceEnrollmentSample,
        ) -> anyhow::Result<Vec<f32>> {
            self.enrollment_vectors
                .lock()
                .expect("enrollment vectors mutex poisoned")
                .get(&sample.id)
                .cloned()
                .ok_or_else(|| anyhow!("missing enrollment vector for {}", sample.id.0))
        }

        fn embed_signature(&self, signature_id: VoiceSignatureId) -> anyhow::Result<Vec<f32>> {
            self.signature_vectors
                .lock()
                .expect("signature vectors mutex poisoned")
                .get(&signature_id)
                .cloned()
                .ok_or_else(|| anyhow!("missing signature vector for {}", signature_id.0))
        }
    }

    #[test]
    fn known_voice_registry_persists_and_reloads() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("known_voices.json");
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let embeddings = Arc::new(DeterministicKnownVoiceEmbeddingProvider);
        let store = KnownVoiceMemoryStore::new(path, qdrant, embeddings);

        let registry = sample_registry();
        store.save_registry(&registry).expect("save registry");
        let reloaded = store.load_registry().expect("load registry");

        assert_eq!(reloaded, registry);
        assert_eq!(
            store.list_known_voices().expect("list known voices").len(),
            2
        );
    }

    #[test]
    fn embedding_storage_sets_embedding_ref_and_qdrant_payload() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("known_voices.json");
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let mut registry = sample_registry();
        let sample = registry.enrollment_samples[0].clone();
        let embeddings = Arc::new(
            StubKnownVoiceEmbeddings::default().with_enrollment(sample.id, vec![0.9, 0.1, 0.0]),
        );
        let store = KnownVoiceMemoryStore::new(path, qdrant.clone(), embeddings);

        let embedding_ref = store
            .embed_and_store_enrollment_sample(
                &mut registry,
                sample.id,
                ExactTimestamp::from_unix_nanos(1_760_000_000_000_000_000),
            )
            .expect("embed and store enrollment sample");

        let stored_sample = registry
            .enrollment_samples
            .iter()
            .find(|item| item.id == sample.id)
            .expect("sample exists");
        assert_eq!(stored_sample.embedding_ref, Some(embedding_ref.clone()));
        assert_eq!(embedding_ref.backend, KNOWN_VOICE_EMBEDDING_BACKEND);
        assert!(embedding_ref.key.contains("known_voice_enrollment"));

        let upserts = qdrant.upserts.lock().expect("qdrant mutex poisoned");
        assert_eq!(upserts.len(), 1);
        assert_eq!(upserts[0].0, KNOWN_VOICE_QDRANT_COLLECTION);
        assert_eq!(upserts[0].1.len(), 1);
        let payload = &upserts[0].1[0].payload;
        assert_eq!(
            payload.get("voice_id"),
            Some(&json!(sample.voice_id.0.to_string()))
        );
        assert_eq!(
            payload.get("enrollment_sample_id"),
            Some(&json!(sample.id.0.to_string()))
        );
        assert_eq!(
            payload.get("source"),
            Some(&json!(EnrollmentSource::GeneratedTts))
        );
        assert_eq!(
            payload.get("memory_scope"),
            Some(&json!(KNOWN_VOICE_LOCALITY))
        );
    }

    #[test]
    fn store_heard_voice_signature_vector_upserts_direct_voice_point() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("known_voices.json");
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let embeddings = Arc::new(DeterministicKnownVoiceEmbeddingProvider);
        let store = KnownVoiceMemoryStore::new(path, qdrant.clone(), embeddings);
        let signature_id = VoiceSignatureId::new();

        let embedding_ref = store
            .store_heard_voice_signature_vector(
                signature_id,
                "voice:test",
                vec![0.1, 0.2, 0.3],
                ExactTimestamp::from_unix_nanos(20),
            )
            .expect("store heard signature");

        assert_eq!(embedding_ref.backend, KNOWN_VOICE_EMBEDDING_BACKEND);
        let upserts = qdrant.upserts.lock().expect("qdrant mutex poisoned");
        assert_eq!(upserts.len(), 1);
        assert_eq!(upserts[0].0, KNOWN_VOICE_QDRANT_COLLECTION);
        assert_eq!(
            upserts[0].1[0].payload.get("voice_node_id"),
            Some(&json!("voice:test"))
        );
    }

    #[test]
    fn associate_voice_with_entity_persists_registry_link() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("known_voices.json");
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let embeddings = Arc::new(DeterministicKnownVoiceEmbeddingProvider);
        let store = KnownVoiceMemoryStore::new(&path, qdrant, embeddings);
        let mut registry = sample_registry();
        let voice_id = registry.voices[0].id;

        store
            .associate_voice_with_entity(
                &mut registry,
                voice_id,
                "person:travis",
                Some("Travis".to_string()),
                0.88,
                VoiceEntityAssociationSource::ExplicitUserStatement,
                ExactTimestamp::from_unix_nanos(30),
            )
            .expect("associate voice with entity");

        let reloaded = store.load_registry().expect("reload registry");
        assert_eq!(reloaded.voices_for_entity("person:travis").len(), 1);
        assert_eq!(reloaded.entities_for_voice(voice_id).len(), 1);
    }

    #[test]
    fn matcher_returns_ranked_attribution_with_alternatives() {
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let mut registry = sample_registry();
        let first = registry.enrollment_samples[0].clone();
        let second = registry.enrollment_samples[1].clone();
        let signature_id = VoiceSignatureId::new();

        let embeddings = Arc::new(
            StubKnownVoiceEmbeddings::default()
                .with_enrollment(first.id, vec![1.0, 0.0, 0.0])
                .with_enrollment(second.id, vec![0.0, 1.0, 0.0])
                .with_signature(signature_id, vec![0.9, 0.1, 0.0]),
        );
        let store = KnownVoiceMemoryStore::new(
            PathBuf::from(DEFAULT_KNOWN_VOICE_REGISTRY_PATH),
            qdrant.clone(),
            embeddings.clone(),
        );
        store
            .embed_and_store_enrollment_sample(&mut registry, first.id, ExactTimestamp::now())
            .expect("store first");
        store
            .embed_and_store_enrollment_sample(&mut registry, second.id, ExactTimestamp::now())
            .expect("store second");

        let matcher = QdrantKnownVoiceMatcher::new(qdrant, embeddings).with_min_confidence(0.3);
        let attributions = matcher.attribute(SpanId(99), &[signature_id], &registry);

        assert_eq!(attributions.len(), 1);
        assert_eq!(attributions[0].voice_id, first.voice_id);
        assert!(attributions[0].confidence > 0.8);
        assert_eq!(attributions[0].source, VoiceAttributionSource::GeneratedTts);
        assert_eq!(attributions[0].alternatives.len(), 1);
        assert_eq!(attributions[0].alternatives[0].voice_id, second.voice_id);
    }

    #[test]
    fn matcher_returns_empty_when_confidence_below_threshold() {
        let qdrant = Arc::new(RecordingQdrantStore::default());
        let mut registry = sample_registry();
        let first = registry.enrollment_samples[0].clone();
        let signature_id = VoiceSignatureId::new();
        let embeddings = Arc::new(
            StubKnownVoiceEmbeddings::default()
                .with_enrollment(first.id, vec![1.0, 0.0, 0.0])
                .with_signature(signature_id, vec![0.0, 1.0, 0.0]),
        );
        let store = KnownVoiceMemoryStore::new(
            PathBuf::from(DEFAULT_KNOWN_VOICE_REGISTRY_PATH),
            qdrant.clone(),
            embeddings.clone(),
        );
        store
            .embed_and_store_enrollment_sample(&mut registry, first.id, ExactTimestamp::now())
            .expect("store sample");

        let matcher = QdrantKnownVoiceMatcher::new(qdrant, embeddings).with_min_confidence(0.95);
        let attributions = matcher.attribute(SpanId(7), &[signature_id], &registry);
        assert!(attributions.is_empty());
    }

    fn sample_registry() -> KnownVoiceRegistry {
        let created_at = ExactTimestamp::from_unix_nanos(1_750_000_000_000_000_000);

        let tts_voice_id = VoiceId::new();
        let human_voice_id = VoiceId::new();
        let tts_sample_id = VoiceEnrollmentSampleId::new();
        let human_sample_id = VoiceEnrollmentSampleId::new();

        KnownVoiceRegistry {
            voices: vec![
                KnownVoice {
                    id: tts_voice_id,
                    label: "PETE-TTS".to_string(),
                    kind: VoiceKind::Pete,
                    enrollment_samples: vec![tts_sample_id],
                    created_at,
                    notes: Some("Self voice generated via local TTS".to_string()),
                },
                KnownVoice {
                    id: human_voice_id,
                    label: "TRAVIS".to_string(),
                    kind: VoiceKind::Human,
                    enrollment_samples: vec![human_sample_id],
                    created_at,
                    notes: Some("Manual enrollment".to_string()),
                },
            ],
            enrollment_samples: vec![
                VoiceEnrollmentSample {
                    id: tts_sample_id,
                    voice_id: tts_voice_id,
                    audio_span_id: SpanId(100),
                    source: EnrollmentSource::GeneratedTts,
                    quality: EnrollmentQuality::High,
                    embedding_ref: None,
                },
                VoiceEnrollmentSample {
                    id: human_sample_id,
                    voice_id: human_voice_id,
                    audio_span_id: SpanId(200),
                    source: EnrollmentSource::ManualLabel,
                    quality: EnrollmentQuality::High,
                    embedding_ref: None,
                },
            ],
            voice_entity_associations: Vec::new(),
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
