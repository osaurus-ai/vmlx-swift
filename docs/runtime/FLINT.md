# Flint runtime integration

Status: bounded runtime/app qualification below; long-context and performance limits remain open.

Flint is a text-only mixture-of-experts model with plain RMSNorm, learned
attention sinks, QK normalization, and mixed full/sliding attention. RoPE is
controlled by `no_rope_layer_indices`, independently of the attention window.
Sliding attention retains 513 keys including the current token. The router uses
BF16 projection followed by FP32 sigmoid/selection and unbiased normalized
weights. Shared experts are ungated. Activation storage remains BF16 because
large expert outputs can overflow FP16.

Bundle per-module quantization determines affine expert and dense precision.
JANGH banks use the existing validated codebook/Hadamard loader; their admission
is independent from affine loading. No new quantization kernels are introduced.
The architecture has no native MTP or bundled drafter. Sampling and EOS IDs come
from bundle configuration. Reasoning is optional model output, not a runtime
switch. Developer messages and assistant reasoning history must retain their
native roles/content. Tool syntax is GLM47.

Full-attention caches remain floating point because learned sinks require that
SDPA path. The sink-cache contract survives copy and disk serialization and
excludes lossy KV promotion. Sliding caches retain their ring state and absolute
positions. Completed checkpoint persistence is distinct from the live KV that
attention needs in RAM.

## Qualification ledger

- Independent tiny Python full/chunked fixture: deterministic BF16 logits match exactly.
- Eleven focused Swift graph/cache/identity tests pass: full/chunked execution, wrapped sliding state, sink-cache disk serialization, one- and three-token suffix continuation, mixed-length batching and forced-tool prefix policy.
- Twelve metadata-partition cases pass; one optional local-fixture case is skipped. Flint calibration metadata is not an executable quantization plan, and competing plans/headers remain rejected.
- Calibrated affine 4M: actual Osaurus connected tool turns return391 then420, each normal stop, about80–81tok/s on the current integrated build. Peak process footprint25.52decimalGB for23.37GB base weights. The forced-tool follow-up restores5153/6519 then6519/6632tokens. Earlier~30k app proof records tool/code recall, new-token prefill and checkpoint storage separately.
- Calibrated H2: final bundle loads through the actual app. Connected tools/SSD restore,~29k input and a real Helper delegation complete normally; free prose completes2044tokens at63.30tok/s, peak15.41decimalGB for13.19GB base weights. During delegation the42.60GB Ornith child is also resident by the selected keep-parent policy; total peak62.22GB falls to15.88GB after child cleanup.
- Calibrated affine 6M: two connected app tool turns return 391 then 420 with normal stops, about 73–77 tok/s. Peak process footprint is 35.05 decimal GB for 32.83 GB base weights. Actual Activity Monitor shows 31.34 GB for PID 90009 after the follow-up; the numeric process capture and UI units remain separate.
- The H2 sidecar admission failure and forced-tool restore omission were reproduced in the app before their fixes. No bundle, seed or sampler was edited. A later stale staging-path reload failure was external folder movement; final-path reload succeeds without a resolver change.

Runtime source598994d44b80c365020d47f7ea94b9b78d58fd71 is built into app786085ff2f4ef32ba75f391c8f21958de1a92a32, binary SHA256 f90d91610ddebc45b3aa3f317eede6eac120abb5dfa686dae2ee1fb2805605c9. Evidence: `~/vmlx-private-evidence/resident-weight-ownership-20261021/`, `FLINT-CALIBRATED-PROGRESS.md`, final-path launch identity and turns/F11–F18 receipts/screens/raw logs. These are actual Release-app results, not RunBench substitution.

Limits: H2 prefill is slower than4M here; no100+tok/s, optimal-kernel or universal speedup claim. One H2 explanation miscounts repeated additions while the calculator/result391 is correct; another prose answer under-delivers requested length. These are recorded quality limitations, not hidden or attributed to runtime without a numeric comparison.128k, aggregate concurrent writer/checkpoint memory and broader API/media qualification are separate gates. Flint is text-only; no media or speculative capability is invented.

Development RTN weights establish implementation compatibility only, not final model quality. Process footprint, host wired/compressed/swap, MLX active/cache, queued writes and on-disk bytes must be reported separately. Model-file size is not a total-memory accounting formula. No all-family cache-memory claim follows from this integration.


Longer live checkpoint: H2 grows from~29k to63,234 then129,022 and134,915 actual tokens. It recalls segment markers and totals; an explicit new calculation beyond134k actually calls the tool and returns4056. Peak process footprint reaches29.06decimalGB and falls to17.13GB after storage. Actual prefill progress was visually inspected. The automatic maxKV threshold caused a reproduced unnecessary full prefill because Flint ignores that parameter; see FLINT-CACHE-IDENTITY-2026-10-21.md for the targeted identity patch and completed post-fix app replay. These rows do not qualify every affine/H2 variant, model-quality prompt or concurrent128k load.
