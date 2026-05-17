# Qwen3.6 Native MTP Optimization Audit - 2026-05-17

This is the working audit for `goalnew.md`. It records the current Swift MTP
speed/correctness state after the chunk verifier investigation. It is not a
global production claim, and it does not enable MTP automatically.

## Objective Mapping

| `goalnew.md` requirement | Current evidence |
|---|---|
| Map the Swift MTP loop end to end. | `NativeMTPTokenIterator` now reports prefill/generation outcome, verifier mode, accepted depth counts, target forward count, verifier input token count, repair count, and phase timings. See `docs/VMLX_QWEN36_MTP_SPEED_ROOT_CAUSE_2026_05_17.md`. |
| Instrument before optimizing. | Live logs include `acceptedByDepth`, `targetVerifySec`, `mtpDraftSec`, `samplingSec`, `cacheCommitSec`, `targetForwards`, `verifyInputTokens`, `repairForwards`, RSS, footprint, and cache stats where enabled. |
| Explain why D3 was slower. | The old default hybrid path used `sequential_repair`, producing almost one target forward per generated token. Explicit `chunk_commit` verifies `[primary, d1, d2, d3]` in one target forward. |
| Implement the real fast verifier path. | The chunk path commits accepted verifier prefixes through `MambaCache.recordPrefixCommitState` and `commitRecordedPrefix`; it remains a greedy-only hybrid speed path. Non-greedy exact-pq hybrid rows use sequential repair after the 35B MXFP4 residual-correction failure. |
| Compare against Python vMLX / MTPLX concepts. | Swift now has recursive D1/D2/D3 drafting, one chunk verifier forward, accepted-prefix commit, and per-phase telemetry. Swift still lacks GraphBank/compiled small-M verifier shapes, dedicated draft-only sidecar heads, and a production self-test that promotes chunk mode per artifact. |
| Review autodetect/startup. | The census rows use `MTPBundleInspector` and real tensor evidence. All six local bundles report `bundleHasMTP=true`, `complete=true`, `canAutoLaunch=false`, and `vision=true`. |
| Validate six artifacts. | Speed rows are under `docs/local/qwen36-mtp-opt/20260517T050311Z-six-artifact-chunk-speed/`. Cache rows are under `docs/local/qwen36-mtp-opt/20260517T050824Z-27b-mxfp4-d3-cache/` and `docs/local/qwen36-mtp-opt/20260517T050858Z-recommended-depth-cache-rows/`. |

## Live Speed Results

Prompt:

```text
Count from 1 to 50 in order, separated by commas.
```

All accepted rows below used `temperature=0.00`, `topP=1.00`, `topK=0`,
`rep=nil`, `stop=stop`, `unclosedReasoning=NO`, `loop=NO`, and
`leaks=none`. No forced repetition penalty, stop-tag insertion, or sampling
clamp was used.

| Artifact | AR tok/s | D1 chunk | D2 chunk | D3 chunk | Exact count | Recommendation |
|---|---:|---:|---:|---:|---|---|
| 27B JANG_4M | 27.4 | 41.0 | 47.4 | 43.2 | yes | D2 for speed; D3 not preferred on this prompt. |
| 27B MXFP4 | 31.8 | 44.8 | 47.3 | 42.9 sweep / 45.2 isolated / 50.5 cache row | yes | D3 is viable and clears the 44 tok/s target when isolated/cache-warm; D2 is lower-variance. |
| 27B MXFP8 | 17.3 | 26.6 | 29.4 | 26.5 | yes | D2. D3 adds draft/reject overhead. |
| 35B JANG_2K | 120.1 | 112.0 wrong | 113.8 wrong | 94.6 wrong | no for chunk | MTP chunk blocked; use AR by default. |
| 35B MXFP4 | 105.3 | 131.0 | 158.2 | 169.3 | yes | D3. |
| 35B MXFP8 | 79.1 | 100.0 | 118.9 | 129.1 | yes | D3. |

The critical 27B MXFP4 D3 target is proven by:

- `docs/local/qwen36-mtp-opt/20260517T050601Z-27b-mxfp4-d3-isolated/d3_chunk_runs3.log`
  - `runs=46.0,45.2,45.2`
  - exact `1..50`
- `docs/local/qwen36-mtp-opt/20260517T050824Z-27b-mxfp4-d3-cache/d3_chunk_cache_runs2.log`
  - `runs=50.5,50.5`
  - exact `1..50`
  - disk hit increments to `1` on run 1
  - SSM hit increments to `1` on run 1

## Cache Rows

Recommended-depth cache rows:

| Artifact | Depth | Median tok/s | Cache finding |
|---|---:|---:|---|
| 27B JANG_4M | D2 | 48.9 | run 1 disk hit `1`, SSM hit `1`; exact output both runs. |
| 27B MXFP4 | D3 | 50.5 | run 1 disk hit `1`, SSM hit `1`; exact output both runs. |
| 27B MXFP8 | D2 | 31.7 | run 1 disk hit `1`, SSM hit `1`; exact output both runs. |
| 35B MXFP4 | D3 | 171.4 | run 1 disk hit `1`, SSM hit `1`; exact output both runs. |
| 35B MXFP8 | D3 | 129.9 | run 1 disk hit `1`, SSM hit `1`; exact output both runs. |

The cache coordinator reports `pagedIncompatible=true` for these hybrid rows.
That is correct for the current path: the proof is disk L2 plus SSM companion
state, not a claim that generic paged KV blocks alone are safe for hybrid SSM.

## 35B JANG_2K Boundary

The 35B JANG_2K artifact is the important failure:

- chunk D1/D2/D3 duplicate count tokens, so they are correctness failures;
- default sequential D3 is exact but only `82.4 tok/s`, slower than AR;
- chunk D3 with `VMLINUX_NATIVE_MTP_FORCE_REJECT_ALL=1` is exact, because no
  draft state is accepted.

Evidence:

- `docs/local/qwen36-mtp-opt/20260517T050311Z-six-artifact-chunk-speed/35b-jang2k_d3_chunk.log`
- `docs/local/qwen36-mtp-opt/20260517T050643Z-35b-jang2k-correctness-isolation/d3_default.log`
- `docs/local/qwen36-mtp-opt/20260517T050643Z-35b-jang2k-correctness-isolation/d3_chunk_reject_all.log`

The likely root cause is not sampling. It is a chunk verifier equivalence
boundary for partial accepted prefixes on this low-bit mixed JANG artifact:
accept-0 and sequential repair are exact, while accepting chunk-verified draft
positions can leave the recurrent SSM state at a boundary that is not equivalent
to token-by-token decode.

Production policy: keep chunk verifier opt-in for `MambaCache` models until a
per-artifact equivalence gate proves chunk accepted-prefix commits are exact.
This is not a hidden guard. It is a scheduler/correctness capability gate. The
runtime must not patch the output stream or force sampling values.

## Exact-PQ Hybrid Verifier Boundary

Current artifacts:

```text
docs/local/qwen36-mtp-current/20260517T125723Z-mxfp-growing-chat-mtp-d3-stats/35b-mxfp4/growing_chat_mtp_d3_bundle_defaults.log
docs/local/qwen36-mtp-current/20260517T130655Z-35b-mxfp4-growing-chat-mtp-d1-exact/growing_chat_mtp_d1_bundle_defaults.log
docs/local/qwen36-mtp-current/20260517T130725Z-35b-mxfp4-growing-chat-mtp-d1-sequential-exact/growing_chat_mtp_d1_sequential_bundle_defaults.log
docs/local/qwen36-mtp-current/20260517T131024Z-35b-mxfp4-growing-chat-mtp-d3-exact-postfix/growing_chat_mtp_d3_bundle_defaults_postfix.log
docs/local/qwen36-mtp-current/20260517T131050Z-mxfp-growing-chat-mtp-d3-exact-postfix/
docs/local/qwen36-mtp-current/20260517T131117Z-35b-mxfp4-growing-chat-mtp-d3-greedy-postfix/growing_chat_mtp_d3_greedy_postfix.log
```

The 35B MXFP4 growing-chat row isolated a real non-greedy failure:

- AR with bundle defaults passed the two-turn cache row.
- Greedy native-MTP D3 with `chunk_commit` passed.
- D3 exact-pq with `chunk_commit` emitted repeated garbage and stopped by
  length.
- D1 exact-pq with `chunk_commit` reproduced the same stream, so the issue was
  not recursive D3 draft depth.
- D1 exact-pq with sequential repair passed.

The runtime now routes non-greedy native MTP over `MambaCache`/hybrid SSM to
`sequential_repair` regardless of the fast chunk env. The post-fix D3 exact-pq
rows for 27B MXFP4, 27B MXFP8, 35B MXFP4, and 35B MXFP8 all pass with bundle
defaults, coherent two-turn output, disk-prefix hits, and SSM hits. Greedy
still reports `verifierMode=chunk_commit` and passes, so the speed path remains
available where the sampler does not need target/draft probability ratios.

## MTPLX Comparison

Swift now has the core runtime shape needed for MTPLX-style speed:

- recursive D1/D2/D3 drafting from MTP hidden state;
- one target verifier pass over `[primary, d1, d2, d3]`;
- private MTP cache;
- accepted verifier prefix commit;
- phase timing and acceptance telemetry.

Swift still lacks several MTPLX-style speed pieces:

- compiled/GraphBank small-M verifier shapes;
- specialized small-M qmv/qmm tuning;
- artifact-specific chunk equivalence promotion;
- adaptive depth selection based on measured acceptance and draft overhead;
- dedicated draft-only low-bit sidecar/head optimization.

The live results show why adaptive depth matters: 27B JANG_4M and 27B MXFP8 are
faster at D2 than D3 on this prompt, while 35B MXFP4/MXFP8 are fastest at D3.

## Current Recommendation

- Keep AR as the global default.
- Do not infer MTP from model names.
- Keep `chunk_commit` explicit and greedy-only for hybrid/Mamba caches.
- Use `sequential_repair` for stochastic exact-pq hybrid MTP until a real
  chunk-probability equivalence gate proves otherwise.
- For proven local explicit rows:
  - 27B JANG_4M: D2;
  - 27B MXFP4: D3 is allowed for the D3 target; D2 remains a lower-variance fallback;
  - 27B MXFP8: D2;
  - 35B JANG_2K: block chunk MTP, prefer AR;
  - 35B MXFP4: D3;
  - 35B MXFP8: D3.
- Before Osaurus auto-launches any MTP mode, add a per-artifact live equivalence
  gate: AR exact text, MTP exact text, cache repeat, reasoning on/off, and VL
  media turn plus text-only continuation.
