# Append-only checkpoint lifetime correction

Status: under validation, not a universal RAM/speed pass.

The reported settled Flint ~17 GB is not proven live KV. The 129k raw log ends
with active=13.37 GB and allocator cache=3.34 GB beside 13.19 GB weights. Earlier
interpretation by adding logical KV geometry to weights was wrong. Logical KV
geometry is useful during compute; it cannot establish retained physical owners.

Source confirms separate N-1/prompt snapshots, copy-before-trim, and an allocator
purge before all owners finish. Existing writer-stage measurements still own the
current payload and do not constitute a buffer-by-buffer physical allocation map.

Implementation:
- For disk-only, uncompressed, noncompiled native full-KV/SWA stacks, retain layer
  indices for append-only KV and seal only overwrite-prone rotating state. No KV
  array view is retained through decode; donation remains possible. After decode,
  build prompt and N-1 views from the completed immutable cache and sealed rings.
- N-1 is published at finalization, not copied into an asynchronous writer during
  prefill. The exact same boundary tokens, salts, native dtypes and sink state
  are used. No synchronous SSD write is added before first token.
- A shorter trimmable boundary slices append-only arrays before owned copy, so a
  short prefix write need not hold an allocated full-prompt duplicate. The exact
  boundary uses its immutable snapshot directly rather than copying it again.
- Durable existing history/stripped rows are checked before boundary construction;
  existing rejected-entry/fingerprint/companion rules remain authoritative.
- Long-context cleanup purges after finished slots are removed. A separate idle
  task waits for writer payload closure release then requests an allocator purge
  on the engine actor only if no slots/admissions are active. It neither lowers
  the allocator limit nor blocks tool completion waiting for writer drain.

The descriptor-owned DiskPromptSnapshot candidate c16ea4718 is preserved on its
separate branch for audit. It adds prompt-time write and terminal read/rewrite,
so it is superseded here and is not intended to merge.

Bounded validation: native bf16 learned-sink + wrapped-SWA prompt and N-1 exactness
through subsequent live mutation and serialization; no append-only cache owner
retained; compact shorter-prefix capacity; unsupported recurrent topology refusal;
then the existing long Flint app conversation, real Activity Monitor, MLX
active/cache/physical footprint, TTFT, decode, terminal tool wait and growing-turn
SSD restore. Unit/build evidence is not actual app proof.

Unchanged: model weights/load ownership, quantization/Hadamard kernels, sampling,
seeds, reasoning/template contracts, speculative strategy/depth, configured RAM
safety and 30-second unload, persistent quota/chain semantics. Explicit compression,
paged RAM, compiled, recurrent/pool topologies retain their old snapshot contract
until separately qualified; this PR does not label them fixed.

Still open: process-wide incoming writer reservations (RAM/8 currently per cache,
old-pending check); cross-family speculative/recurrent lifetime work; all-family
128k resource/media/eviction coverage. Tool execution must retain the GPU lease
until engine work drains; early tool dispatch is not safe merely because tool JSON
is available (the tool may launch another GPU producer). Reduce finalization work
first; any lease split requires its own synchronization proof.
