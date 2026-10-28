# QSA raw-key capacity checkpoint — PARTIAL

NOW: Qualify stepped raw-key storage against the original per-update concatenation.
DO NOT: Claim a model speedup or merge readiness from isolated cache timings.
BATCH OWNER: `QSAKVCache` in `Libraries/MLXLMCommon/KVCache.swift`.
NEXT: Rebuild the isolated local development app; compare both paths on identical
2L/4S requests at short, 8K and 35K contexts and exercise visible continuation.

## Ownership correction before app qualification

The first candidate, `dcdf90dc896ac537ec29c928bbc0d2defc849f82`, built but was
not launched. An additional exact-capacity probe reproduced a regression missed
by the initial tests: after 256 raw rows, trim8 and append4, retained raw views,
serialized raw payload and source raw cache changed under capacity mode. The
concat control preserved all three (`SWIFTTEST_QSACapacityOwnershipRedC0913__012635.log`).

`MLXArray` is a mutable reference wrapper; a slice write replaces its context.
The public raw getter now returns a fresh view even at exact capacity, and state
adoption separates the external wrapper from the live backing. This does not
retain an extra permanent view or force an eager full-prefix copy each token.
New regression coverage includes both storage modes, BF16/F16/F32,256/512rows,
and lazy/evaluated storage. The corrected standalone probe preserved all three
raw-state observations in both modes:
`SWIFTTEST_QSACapacityOwnershipGreen0913__013407.log`, exit0, peak1.16GiB,
swap0.49GiB unchanged, cleanup0/0/0. This is generated-fixture execution, not
full-model continuation proof.

## Scope and cache contract

The base is `4c6a0c0ccef1362663ac4ba74fb5b473730ac48f`. Main K/V buffers already
have stepped storage, and QSA already retains completed pooled blocks. This
change only gives the RAW indexer-key lane its own backing capacity and logical
row count; it does not change pooling, normalization, attention or quantization.

New rows overwrite pending rows at the committed offset, bounded by the raw
prefix actually present. Missing rows are never synthesized. Trim retains spare
capacity but removes logical rows and drops derived pooled state. Public reads
and three-array serialized state never expose spare capacity. Copy preserves
owned state; restored state reserves new storage only when subsequently needed.
Dtype changes retain concatenate's promotion instead of silently casting writes.
No model-name, quant-name or context-length allowlist is used.

`VMLX_QSA_RAW_CAPACITY=0` selects the old concatenation control at cache creation.
`VMLX_QSA_RAW_STORAGE_TRACE=1` emits an opt-in first-update receipt, including
logical length, backing capacity and dtype. These are diagnostic switches, not
new application settings. No per-token environment lookup is added.

The storage principle was compared with reference runtime's `_QSAIndexerCache` at revision
`7cbb407168ae628bbe0d7fe385be70e0954af303`. Its geometric reservation and media
position schema were not copied. Swift's separate QSA media-position companion
gap remains outside this patch and must not be declared qualified by text rates.

## Executed focused tests

Local M5 Max, existing bounded supervisor, optimized test host only:
`SWIFTTEST_QSARawStorageOwnership0913__012815.log` (corrected, formatted source).

- 24 tests in two suites, including `QSAKVCacheStorageTests`,
  `Qwen4ExpQSATests` and existing QSA persistence/quantization exclusions.
- 180 exact sequence checks across BF16/F16/F32, batch1/2, strided rows,
  contexts1/255/2040/8339/34939, single/multiple pending rows and trim-to-zero.
- Mixed-dtype transitions, retained views and independent copies, empty updates,
  valid serialized lengths, damaged/missing lanes, and safetensors save/reopen
  followed by restored raw/KV continuation across the2048 budget.
- Exact-capacity retained/serialized/source raw ownership across24 variants.
- Exit0; tracked peak3.83GiB; swap0.49GiB unchanged; ownership cleanup0/0/0.
  Three existing optional QSA timing tests returned early because their distinct
  diagnostic flags were unset; their optional timing workloads were not executed.

Median of three alternating-order synthetic12-lane raw-cache measurements,
256 appended tokens per arm (milliseconds per12-lane update):

| Starting rows | Concatenation | Capacity |
|---|---:|---:|
| 1024 | 0.292976 | 0.230832 |
| 8339 | 0.527424 | 0.243830 |
| 34939 | 0.612483 | 0.209890 |

These synchronized raw-cache measurements do not include model compute or app
streaming. The internal growth counter counts explicit backing-store growth,
not physical Metal allocations; retained views can still prevent donation.
No full-model tok/s gain, universal45tok/s floor, VLM quality, RAM soak or MTP
performance claim is made. Dev-app source/pin/binary and live rows are pending.

Private detailed receipts: `vmlx-private-evidence/post-1653-qwen38-audit/`;
supervisor logs in sibling `mtp-swift-2026-09-04/logs/`.
