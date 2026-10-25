# DeepSeek-V4.1-Flash runtime in vmlx-swift (2026-10-25)

Status and contract for DeepSeek-V4.1-Flash (`model_type: deepseek_v41`, text config
`deepseek_v41_text`) on the DSV4 code path, as served by Osaurus. The Python runtime in jangq
(`jang_tools/dsv41/`) is the numerical reference: see its `RUNTIME_SERVING.md` and
`DECODE_DSPARK_METHODS.md`. The authoritative prompt format is the bundle's
`encoding/encoding.py`.

## Family isolation (DSV4-0731 must not change)

`DeepseekV4ChatFamily(modelType:)` selects the contract. Anything that isn't explicitly
`deepseek_v41*` stays on 0731:

| behavior | 0731 (`deepseek_v4`) | V4.1 (`deepseek_v41`) |
|---|---|---|
| reasoning effort | named preface (`low` = none) | numeric `Reasoning Effort: N` inside `<｜System｜>` (ladder: none 25, low 50, medium 62, high 75, xhigh 88, max 100) |
| system messages | 0731 rendering | System-led; mid-conversation system turns supported |
| tool tags | 0731 DSML names | V4.1 tags with a leading space (`<｜DSML｜ calls>`, ` invoke`, ` parameter`); `ToolCallFormat.dsmlV41` |
| indexer weights | loaded (`.indexer.*`) | V4.1 indexer (`self_attn.v41_indexer.*`) |
| compressed-KV pools | 8-bit | 4-bit, two per byte |
| resident safetensors load | unchanged | required |
| reasoning floor | 0731 markers | never arms on a numeric V4.1 prompt |

Hosts must render with the bundle's family. Osaurus does this in
`SwiftTransformersTokenizerLoader.applyDeepseekV4NativeTemplate`
(`DeepseekV4ChatEncoder(family:)`, plus `reasoningBudget(from:)` for V4.1 thinking turns). An
absent effort renders as low (50). The bundle reference default is high (75).

## Engram (n-gram embedding tables)

- Tables are located from `jang_config` `runtime.engram_dir` (default `<bundle>/engram`).
  No environment variable is needed. Load throws if a V4.1 bundle's tables are missing or
  don't match the config. It never runs without Engram.
- Rows are read in place from the safetensors shards with concurrent `pread`: 64 lanes,
  256-row tasks, rows and scales together. The OS page cache keeps hot rows; nothing is wired.
  The reads for layer 14 overlap the compute of layers 0–13. Cold random rows read about
  130k rows/s this way, against 8–11k rows/s serially.
- The n-gram history travels with the cache (first compressed layer): copy, trim, metaState
  and disk records. A V4.1 disk record without history is a cache miss, never padded. The
  runtime holds no per-sequence state.

## KV cache size

V4.1 compressed pools are 4-bit (about 0.625 B per element against 1.125 for 8-bit). The disk
format records the width and refuses to restore into a cache of the other width.
`VMLX_DSV41_KVBYTES=1` logs total cache bytes, bytes per token and pool bits after each
prepare. Target: ~0.9 GB for a 1M-token context, resident and stored.

## Long context: indexer

Port of the Python indexer. Layers 2/8/14/20 build the shared index keys from compressor
output. The other index layers choose their own top-512. Layer 20 sets the candidate blocks
that layers 24+ select from, and layers in between reuse the previous selection. With 512 or
fewer visible compressed rows every row is attended, as in the reference. Without the indexer,
output diverges past ~512 compressed rows and prefill is quadratic.

## Measured baseline (PR #657 head `661e9173`, AR, before these fixes)

M5 Ultra 256 GB, Osaurus Release build, `DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX`:

| metric | value |
|---|---|
| load + first reply | 35.5 s |
| phys_footprint after load (peak) | 196.7 GB (197.4) |
| cold prefill, 1.2k tokens | 488 pp/s |
| decode (thinking on) | 26.6–26.9 tok/s |
| TTFT | 0.79–0.91 s |
| SSD L2 repeat prompt | 23.7 s → 0.47 s; hits survive a restart |
| tool calls | broken: raw V4.1 DSML in content (fixed by `dsmlV41`) |

`usage.prompt_tokens_details.cached_tokens` reports 0 even on an L2 hit. That's a reporting
gap, not a cache miss.

## GPU verification checklist (for the fixes above)

1. 0731: prompts byte-identical to before (low preface, no `<｜System｜>`); indexer weights
   load; generation unchanged.
2. V4.1 loads with no `VMLX_DSV41_ENGRAM_DIR` and is coherent; renaming `engram/` makes
   loading throw a "DeepSeek-V4.1 engram" error.
3. V4.1 loads `self_attn.v41_indexer.*` (`wq_b` as 8-bit, group 64).
4. A V4.1 tool prompt returns populated `tool_calls` and no DSML in content.
5. `xhigh` renders `Reasoning Effort: 88`; "Say hello in exactly one word." does not loop.
6. A multi-turn prefix-cache hit continues exactly like a cold run; the "n-gram history
   missing" log never appears.
7. `VMLX_DSV41_KVBYTES=1` shows `pool bits 4` and about half the bytes per token of 8-bit.
8. Past 2K tokens output stays coherent and prefill speed does not degrade quadratically.
9. Cold prefill of a 1K-token unique prompt is clearly faster than the baseline.

## DSpark (separate branch `feat/dsv41-dspark-swift`)

Not default-on. Open issues before it can be:
- The native-MTP restore replaces cache objects, but the drafter lane is keyed on cache
  identity, so every reject wipes the draft window (acceptance ~0). Fix with an in-place
  restore or a position-keyed lane.
- On a prefix hit only the uncached tail seeds the window.
- Per cycle there is a 40-cache deep copy, ~7 host syncs, and the markov head is cast to f32
  five times.
- The controller needs the wall-clock governor from the Python reference (never slower than
  AR, no permanent latch).
- Gate to V4.1 bundles with a `dspark/` directory. The `deepseek_v4*` cold-start depth must
  not turn native MTP on for 0731, and an env force must not override an explicit Off.
