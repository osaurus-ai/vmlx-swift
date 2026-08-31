# Safetensors heal-on-load checkpoint — 2026-08-31

## Scope

Existing bundles with dtype-misaligned safetensors offsets are repaired before
the mmap-backed loader opens them. Tensor payload bytes and logical descriptors
are preserved; only container order and offsets change.

## Safety contract

- Repair is enabled by default for ordinary writable model directories.
- Each affected shard is streamed to an adjacent temporary file, flushed,
  fully byte-verified tensor-by-tensor, and installed by atomic rename.
- The original shard is re-stat'd before replacement. An interrupted write,
  disk-capacity failure, read-only directory, malformed header, or any other
  repair error leaves the original available to MLX's copy-on-load fallback.
- Hugging Face hash-addressed stores and symlinked blob shards are not mutated.
- Repairs happen before mmap and are a no-op on subsequent loads.

## Current proof

| Gate | Evidence | Status |
| --- | --- | --- |
| Focused Swift tests | `JangPressSafetensorsAlignmentTests`, 8 tests under the Xcode Swift toolchain | PASS |
| Byte preservation | Dense and routed fixtures plus full source/output payload comparison in the installer | PASS |
| Atomic fallback | Injected interrupted stream retains original and cleans the temporary file | PASS |
| Disk fallback | Injected insufficient capacity returns fallback without mutation | PASS |
| Hash-store protection | HF snapshot and symlinked blob fixtures remain unchanged | PASS |
| Production hook | `JangPressPrestacker.prepareBundleIfNeeded` repairs before mmap preparation | PASS |
| Real model families | Qwen, Gemma, Laguna, Nemo, and Raptor short multi-turn matrix | PENDING |

The checkpoint is not merge- or release-ready until the real-model matrix
records output, load/warmup behavior, token/s, physical footprint, and crash
status.
