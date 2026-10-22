# Flint runtime integration

Status: draft implementation, not release-qualified.

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

- Header/index validation of development RTN bundle: passed locally.
- Independent tiny Python full/chunked reference: identical; fixture retained.
- Swift fixture, chunk/ring/disk serialization restore, mixed-length batching: 8 focused tests passed. The deterministic tiny BF16 logits match the independent Python reference exactly.
- Real Osaurus loading, connected tool turns, SSD hit, unload and host/process
  memory attribution: pending.
- Calibrated affine and final JANGH2: pending separate bundle qualification.

Development RTN weights establish implementation compatibility only, not final
model quality. Process footprint, host wired/compressed/swap, MLX active/cache,
queued writes and on-disk bytes must be reported separately. Model-file size is
not a total-memory accounting formula. No all-family cache-memory claim follows
from this integration.
