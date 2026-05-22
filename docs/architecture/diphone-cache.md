## Local generated-diphone cache

Listenbury stores generated neural diphones in a local disk cache (default: `./diphone-cache/`).

- Cache files are paired as `<cache-key>.json` (provenance metadata) and `<cache-key>.pcm` (audio samples).
- Cache keys include model/config identity, diphone phones, forge/normalization versions, speaker ID, and sample rate.
- Metadata records provenance details (model/config fingerprints, generation timestamp, carrier sequence, extraction range, segmentation confidence, and license/provenance notes).

### Local-only expectations

- The cache is git-ignored and intended for local reuse of expensive generation work.
- Cached generated diphones are **not** assumed redistributable.
- If model license metadata is unavailable, metadata records it as `"unknown"`.
- Runtime local generation and local caching are different from distributing a generated voice database.
