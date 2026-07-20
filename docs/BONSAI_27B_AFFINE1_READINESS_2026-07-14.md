# Bonsai 27B affine-1 readiness — 2026-07-14

## Scope

This checkpoint adds native Metal consumption of the schema-2 affine one-bit
weights in `/Users/eric/models/Bonsai-27b-1bit-JANG`. It also preserves the
bundle's exact per-tensor quantization manifest, including its four-bit vision
tensors, and hardens the real image/video/tool/reasoning proof harnesses.

The bundle declares Qwen 3.5 VLM image and video support. It declares
`audio_verified=false`; audio is therefore not a supported modality row for
this bundle and was not treated as a pass requirement.

## Source checkpoint

- vmlx-swift base: `a9b10f60e330337a9de2d8ebe3ca74a7370525e4`
- MLX dependency PR: `osaurus-ai/mlx#2`
- merged MLX integration pin: `9e01acd573a18540468160ccaffeb6fb566e891e`
- 1-bit bundle size: 4,472 MiB measured by the release harness
- schema-2 manifest: 581 entries
  - 498 language affine1/group-128 entries
  - 83 vision affine4/group-64 entries

The runtime keeps affine1 tensors packed. A lossless Swift-side expansion to
two bits was evaluated and rejected because it raised peak physical footprint
to about 8.8 GiB (202.5% of bundle size).

## Current source tests

All commands used a release build and the full Xcode toolchain.

| Gate | Result |
| --- | --- |
| Fresh local MLX Python build, `TestQuantized.test_affine_one_bit_metal` | PASS, 1 test |
| MLX pre-commit hooks on the six-file kernel change | PASS |
| `swift test -c release --filter JangAffine1RuntimeContractTests` | PASS, 6 tests |
| `swift test -c release --filter VLMProcessorCacheScopeSaltTests` | PASS, 7 tests |

The affine1 suite covers schema validation, malformed-contract rejection,
unaligned QMV, optimized 1024-wide QMV, QMM, and exact manifest resolution for
the ambiguous four-bit vision packing.

The complete repository suite is **not green** and is not claimed as evidence
for this PR. A default parallel run deadlocked in the existing opposite lock
order between `/vmlx_mlx_lock` and `MLXMetalTestLock`. An explicit
`--no-parallel` run avoided that deadlock but recorded failures in untouched
Gemma4 source-contract tests, emoji detokenizer tests, memory-safety-settings
expectations, DSV4 prompt tests, and other unrelated suites before the test
runner exited with signal 5. No failure was recorded in the changed affine1 or
Qwen3VL processor suites. The two focused suites above were rerun separately
from the current PR source and passed.

## 1-bit live release matrix

All rows below used the real bundle and production mmap-backed load policy.
No prompt coercion, forced reasoning closer, sampler clamp, hidden repetition
penalty, or synthetic generation default was added.

| Row | Live result | Verdict |
| --- | --- | --- |
| Text multi-turn | `SAVED` at 44.52 tok/s; callback `ORCHID` at 44.35 tok/s; clean stops | PASS |
| Text physical footprint | peak delta about 1.55 GiB, 35.7% of bundle size | PASS |
| Image, compile off | coherent synthetic red/blue gradient description at 39.8 tok/s; follow-up `blue` at 34.9 tok/s | PASS |
| Image, compile on | same grounded description at 39.2 tok/s; follow-up `blue` at 38.4 tok/s | PASS |
| Image physical footprint | peak delta 3,151 MiB, 70.4% of bundle size | PASS |
| Structured image cache | cold A; same-media disk restore HIT 99/99; coherent A replay; different-media MISS; grounded follow-up | PASS |
| Video multi-turn | real triangle fixture; grounded circle/triangle answer at 41.8 tok/s; foreground follow-up at 42.6 tok/s | PASS |
| Video physical footprint | peak delta 3,671 MiB, 82.1% of bundle size; post-turn footprint dropped to 1,765 MiB | PASS |
| Tool parser | structured `get_weather({"location":"Tokyo"})`; one tool event; no raw XML marker leakage | PASS |
| Reasoning parser | natural stop after 372 tokens at 39.2 tok/s; 332 reasoning deltas; visible `Answer: 4.`; closed reasoning; no markers | PASS |

The tool-only envelope measured 17.14 tok/s. That row validates structured
parser behavior, not the text throughput target; the release text rows are the
approximately 45 tok/s performance gate.

## Ternary regression matrix

`/Users/eric/models/Bonsai-27b-Ternary-JANG` remained coherent after the
affine1 work.

| Row | Live result | Verdict |
| --- | --- | --- |
| Text multi-turn | `SAVED` at 38.63 tok/s; `ORCHID` at 33.73 tok/s | PASS |
| Text physical footprint | peak delta about 1.37 GiB, 18.3% of bundle size | PASS |
| Image | grounded gradient and color follow-up, compile off/on; peak 34.6% of bundle size | PASS |
| Video | grounded circle/triangle outputs; 34.2 and 16.0 tok/s; peak 27.4% of bundle size | PASS |

The ternary row is a regression check, not a 45 tok/s claim.

## Rejected or superseded diagnostics

- Original vmlx main crashed the 1-bit bundle at the Metal bits assertion.
- The lossless Swift 1-to-2-bit expansion was coherent at 35.59 tok/s but
  failed low-RAM requirements at 202.5% of bundle size and was removed.
- The old plain-loader VL harness reached 145.6% of bundle size. It bypassed
  the production mmap policy; all VL harness entry points now use the
  production loader.
- Video initially peaked at 113.7% because completed media arrays remained in
  allocator working-set caches. Post-slot GPU fencing and media-only cache
  cleanup reduced the current row to 82.1%.
- A 128-token reasoning diagnostic stopped inside reasoning and failed by
  design. The 512-token row closed naturally and passed; no forced closer or
  length-cap pass was used.
- Qwen emitted valid tool XML before the processor fix, but `LMInput` had
  dropped the active schemas and the stream parser stripped the call. The
  processor now preserves schemas on both text and media inputs, and both the
  focused test and real structured tool row pass.

## Still pending after this runtime PR

These items must not be described as complete until their own evidence lands:

1. vmlx-swift PR CI and merge.
2. Focused Osaurus pin PR to the merged vmlx-swift revision.
3. Isolated signed Osaurus development build that does not replace or disturb
   the user's installed/running app.
4. Computer Use visual proof in that isolated build: model discovery/load,
   coherent text multi-turn, image/video where exposed by the app, tool and
   reasoning presentation, physical-footprint observation, and clean unload.
5. Finder AppleScript re-confirmation after the prior AppleEvents TCC reset.
6. Sustained repeated-spawn/delegation physical-footprint proof.
7. Separate automatic model routing and clearer hardware guidance work in the
   preserved Osaurus routing lane. That work is intentionally paused until the
   runtime and pin chain is merged.

## 2026-07-19 Bonsai cache regression checkpoint

This follow-up uses the locally installed JANG bundle
`/Users/eric/models/dealign.ai/Bonsai-27b-1bit-JANG-CRACK`, not an MXFP4
bundle. It is a Qwen 3.5 hybrid with 16 attention-KV layers and 48
GatedDeltaNet/Mamba companion layers. No model-behavior guard, forced marker,
sampler override, prompt rewrite, or output cap is part of this checkpoint.

### Current evidence

| Gate | Current result | Status |
| --- | --- | --- |
| Isolated app identity | Release app at `/private/tmp/osaurus-bonsai-cache-candidate-derived-20260719/Build/Products/Release/osaurus.app`; bundle ID `com.dinoki.osaurus.bonsaicachestateproof`; binary SHA-256 `de00ead1be7f6681bbbe2a46965c2074d55d90e9a4b4df2512277f9685bee7b9`; ad-hoc signature verified with `codesign --verify --deep --strict` | VERIFIED-BUILD |
| Default cache policy | Real Settings UI showed prefix ON, paged RAM OFF, disk L2 ON, engine-selected KV, and SSM rederive ON. After saving and reloading the model, `/admin/cache-stats` reported `effective_kv_mode=fp16`, 16 KV + 48 Mamba layers, zero TurboQuant compressions, and paged cache disabled | VERIFIED-LIVE |
| Disk-only native restart | A fresh app process restored a 4,669-token disk boundary plus SSM companion state, then answered the visible continuation coherently at TTFT 0.72s / 56.6 tok/s. After the Settings round-trip back to engine-selected, a later 4,943-token hit plus 454-token replay answered coherently at TTFT 2.05s / 54.7 tok/s | VERIFIED-LIVE |
| Explicit TurboQuant 4/4 | The real Settings UI rejected TurboQuant until explicit key/value widths were entered, saved 4/4, unloaded the model, and reloaded it. Telemetry showed `turbo_quant_compressions=2`, exactly 16 converted KV layers, 48 native Mamba layers, paged hits remaining zero, disk-L2 hits 1 -> 3, and SSM hits 1 -> 3 | VERIFIED-LIVE / PERFORMANCE-PARTIAL |
| TurboQuant behavior and speed | The first explicit-TurboQuant answer was coherent at TTFT 10.79s / 39.3 tok/s. A disk-only partial continuation hit boundary 5,301 with 30 tokens replayed and answered coherently at TTFT 1.76s / 17.0 tok/s. Functional cache conversion/reuse is proven; the decode slowdown versus the restored fp16/default row remains open | PERFORMANCE-PARTIAL |
| Paged RAM + TurboQuant + SSM rederive baseline | Before the lifecycle fix, the same stored chat crashed after a 2,923-token disk hit and full-prefill fallback with real Settings enabling paged RAM (two blocks), disk L2, TurboQuant 4/4, prefix, and SSM rederive | REPRODUCED-LIVE baseline |
| Fresh replay lifecycle patch | `reDeriveSSMStatesAtBoundaries` enters the first independent replay chunk through `LanguageModel.prepare`, clearing request-scoped Qwen 3.5 position state, then preserves one continuous fresh cache while capturing later boundaries. The candidate crossed the prior crash boundary repeatedly, including more than 5,000 prompt tokens, without the stale-position shape failure | VERIFIED-LIVE |
| Focused regression suites | Xcode 26.6 / Swift 6.3.3: `SSMReDeriveParityTests` 8/8; filtered DiskCache, SSMStateCache, and BatchEngine cache coverage 33/33. The CommandLineTools-only invocation cannot import Swift Testing; the same current source passed under the full Xcode toolchain | PASS |
| Total disk cap migration | A clean exact-model root migrated from 21,120,148,753 bytes to 10,685,640,588 bytes under a 10 GiB cap by retiring 133 unlinked legacy companions and zero indexed KV entries. After the live matrix it held 18 indexed KV payloads (8,994,768,622 bytes) and 18 linked companion pairs (1,412,190,222 bytes), total 10,406,958,844 bytes; every companion `kv_hash` resolved to a current SQLite row | VERIFIED-LIVE |
| Transient companion writes | Replay, paged-boundary, first-token, inline-capture, and native-MTP seed paths now retain transient SSM state only in memory. Only durable exact generation boundaries write companion files; the live root no longer accumulated hundreds of per-16-token sidecars | VERIFIED-LIVE |
| Visible instruction following | One earlier two-sentence recall omitted the requested arithmetic confirmation; a corrective turn and all restart/TurboQuant/default-restored continuations answered the requested facts coherently. This is recorded as model-level variance, not hidden with a prompt, sampler, marker, or output guard | PARTIAL, transparent |

### Paged crash root trace

The Release app was rerun under LLDB against the same stored cache. It restored
the 2,923-token disk boundary, rejected the full hybrid hit because the seed
boundary SSM state was missing, and entered prompt-boundary SSM rederivation.
The crash stack was:

`reDeriveSSMStatesAtBoundaries` -> Qwen 3.5 VLM/text forward ->
`GatedDeltaNet.callAsFunction` -> MLX indexing precondition.

Immediately before the precondition, MLX reported incompatible shapes
`(1,24,16,64)` and `(1,1,3,64)`. The rederive helper allocated a fresh cache
but called the model forward directly. Qwen 3.5 VLM also retains
request-scoped MRoPE position arrays on the model object, and its normal
text-only `prepare` path resets that state. Bypassing `prepare` therefore
replayed a long prompt against stale, three-token position state. The patch
restores the architecture's normal fresh-request lifecycle instead of
changing generated content.

### Current closure and remaining limitation

The stale-position crash and the companion-sidecar quota defect now have both
owning-layer source fixes and live proof in the isolated signed Release app.
The UI was returned to prefix ON, paged RAM OFF, disk L2 ON, engine-selected KV,
and SSM rederive ON; effective runtime telemetry confirmed fp16 KV and zero
TurboQuant conversions after that restoration. No MXFP4 bundle was loaded or
used as evidence.

The explicit TurboQuant 4/4 path is functionally coherent and its hybrid cache
topology, partial disk reuse, and SSM companion reuse are proven, but its
17.0 tok/s continuation is materially slower than the 54.7 tok/s restored
default row. That performance issue remains separate follow-up work and must
not be described as fixed by this checkpoint. The first multi-clause recall's
omission is likewise retained above rather than masked by a forced behavior
guard.
