# Qwen3.6 Native MTP Production Reverify - 2026-05-17

This note records the fresh production-style Qwen3.6 native-MTP rerun after the
short-budget reasoning failures in the first six-variant matrix. It covers the
MXFP variants only; the user explicitly narrowed the current MTP follow-up away
from JANG_4M/JANG_2K for this pass.

Artifact root:

```text
docs/local/qwen36-mtp-current/
```

`docs/local/` is gitignored; paths below are local evidence references.

## Gate Contract

All four rows used the same gate shape:

- `BENCH_PROD=1`
- `BENCH_PROD_NATIVE_MTP_DEPTH=3`
- `VMLINUX_NATIVE_MTP_HYBRID_VERIFY=chunk_commit`
- `BENCH_PROD_COORD=1`
- `BENCH_PROD_CACHE_HYBRID=1`
- `BENCH_MAX_TOKENS=384`

The sampling row came from each bundle's `generation_config.json`:
`temp=1.000 topP=0.950 topK=20 minP=0.000 rep=nil`. No forced repetition
penalty, forced greedy decode, forced reasoning close, or output-moving guard
was used.

`chunk_commit` remains explicit. These rows prove the current opt-in path; they
do not justify auto-launching native MTP for every Qwen3.6 bundle or enabling it
inside generic batched scheduling.

## Results

| Bundle | Artifact | Result | Reasoning-row tok/s | Cache proof | Memory |
|---|---|---:|---:|---|---|
| `Qwen3.6-27B-MXFP4-MTP` | `20260517T124139Z-27b-mxfp4-prod-budget384/prod_mtp_d3_chunk_budget384.log` | 7/7 PASS | 25.8-29.9 | `disk{hits=1,misses=18,stores=11}`; `ssm{hits=1,misses=0,reDerives=0}` | peak RSS 15,686 MiB; footprint 16,999,313,912 bytes |
| `Qwen3.6-27B-MXFP8-MTP` | `20260517T124237Z-27b-mxfp8-prod-budget384/prod_mtp_d3_chunk_budget384.log` | 7/7 PASS | 18.0-21.1 | `disk{hits=1,misses=20,stores=10}`; `ssm{hits=1,misses=0,reDerives=0}` | peak RSS 1,034 MiB; footprint 2,759,822,000 bytes |
| `Qwen3.6-35B-A3B-MXFP4-MTP` | `20260517T124323Z-35b-mxfp4-prod-budget384/prod_mtp_d3_chunk_budget384.log` | 7/7 PASS | 79.3-95.7 | `disk{hits=1,misses=18,stores=9}`; `ssm{hits=1,misses=0,reDerives=0}` | peak RSS 537 MiB; footprint 20,082,231,504 bytes |
| `Qwen3.6-35B-A3B-MXFP8-MTP` | `20260517T124351Z-35b-mxfp8-prod-budget384/prod_mtp_d3_chunk_budget384.log` | 7/7 PASS | 59.9-70.9 | `disk{hits=1,misses=18,stores=8}`; `ssm{hits=1,misses=0,reDerives=0}` | peak RSS 487 MiB; footprint 1,796,868,352 bytes |

The gate scenarios were:

- S1 reasoning ON math;
- S2 same prompt with cache reuse candidate;
- S3 reasoning OFF factual;
- S4 reasoning ON -> OFF -> ON alternation inside one engine;
- S5 UTF-8 verbatim output.

Every row produced visible answers, no loop/leak, normal stop, L2 disk writes,
and an SSM companion hit. The previous "answer only appeared in reasoning" rows
were caused by the short production-gate budget for these MXFP artifacts. With
384 tokens, the model naturally closes reasoning and emits visible content.

## VL Follow-Up

`Qwen3.6-35B-A3B-MXFP4-MTP` also passed the strict VL+MTP chat-cache rerun with
the same 384-token budget:

```text
docs/local/qwen36-mtp-current/20260517T124945Z-35b-mxfp4-vl-mtp-budget384/vl_chat_cache_mtp_d3_budget384.log
```

The row used `BENCH_VL_CHAT_CACHE=1`, `BENCH_VL_NATIVE_MTP_DEPTH=3`, and
`VMLINUX_NATIVE_MTP_HYBRID_VERIFY=chunk_commit`. It described the red/blue
gradient on a cold image turn, hit same-media disk restore `84/84`, correctly
missed a different-media probe, and answered the text-only follow-up with
`Red and blue`. Peak footprint was `20,528,187,816` bytes.

## Still Open

- This does not change the default: AR remains global default and native MTP
  remains tensor-gated plus explicit opt-in.
- Raw batched native-MTP scheduling is still not implemented; use the exclusive
  `BatchEngine.generate` / `Evaluate.generate` path.
- 27B MXFP8 D3 is correct but not the best speed policy; earlier speed sweeps
  still recommend D2 for the count-prompt speed row.
- 35B JANG_2K still has an unresolved strict VL+MTP length-exhausted row. The
  MXFP4 VL row above is now a pass with a sufficient token budget.
