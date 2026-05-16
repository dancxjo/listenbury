/// Pluggable embedding provider used by cold-memory vector storage.
///
/// Implementations may call a local model, a remote API, or a mock test
/// provider.  Failures must be surfaced as `anyhow::Error` values so the
/// background worker can log them and continue running.
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}
