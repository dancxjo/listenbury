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

## CLI workbench

Use `listenbury diphone` to forge, prewarm, inspect, and audit:

```bash
listenbury diphone forge --model path/to/model.onnx --config path/to/model.json --left h --right @
listenbury diphone cache-build --model path/to/model.onnx --config path/to/model.json --inventory en-us-basic
listenbury diphone cache-list --model path/to/model.onnx --config path/to/model.json
listenbury diphone audit-plan --model path/to/model.onnx --config path/to/model.json --plan out/example.pho
```

Single-unit forge optionally writes debug artifacts with `--debug-dir`:

```text
<debug-dir>/<left>-<right>-carrier.wav
<debug-dir>/<left>-<right>-extracted.wav
<debug-dir>/<left>-<right>-normalized.wav
<debug-dir>/<left>-<right>.json
```
