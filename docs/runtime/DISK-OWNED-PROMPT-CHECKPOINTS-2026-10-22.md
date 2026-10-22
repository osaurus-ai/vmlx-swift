# Disk-owned prompt checkpoint lifetime

Status: implementation under validation, not merge-ready or a claim of all-family RAM closure.

The batch path retained an owned full prompt cache during the complete decode,
while prefill and terminal SSD writers could own other payloads. SSD persistence
did not release that retained prompt bank. With long full-attention state, this
is a distinct memory owner beyond the model and live compute cache.

For disk-only, self-contained native full-KV/rotating-SWA topologies, capture the
immutable prompt state directly from evaluated live buffers into an uncached SSD
workspace. The request retains a descriptor and byte/token counts, without cache
arrays. Unlink the workspace immediately after opening the descriptor: final
close frees it even on process death, and quota eviction cannot remove an active
checkpoint. At finalization, restore through the existing checked owned reader
before constructing the original persistent boundaries. No mmap or dtype conversion
is introduced. A failed restore must never use later decode state under a prompt
key. Disk-write failure logs explicitly and retains the pre-existing RAM checkpoint
path; that row is not an SSD-owned-memory pass.

This temporary workspace uses the configured cache volume but is not an indexed
reusable prefix. It temporarily consumes SSD bytes until completion/cancellation.
Its synchronous write and terminal read must be included in TTFT and tool-tail
measurements; reducing RAM must not be presented as a speed win without timing.
The live KV cache remains GPU accessible for computation. Permanent prefix keys,
chain-aware eviction, active-chat behavior and safety settings are unchanged.

Scope: batch full-KV/SWA; no family-name or quant-name matching. Recurrent companion,
pool, explicit paged RAM, encoded KV and speculative iterator lifetimes retain
existing behavior pending complete-state proof. No sampler, template, seed,
speculative depth or fused-kernel change.

Required bounded checks:
- Native bf16 state survives release of the original cache; snapshot holds no source owner.
- Wrapped SWA plus learned-sink state round-trips exactly after live decode mutation.
- Malformed checkpoint fails closed; descriptor and workspace cleanup on release.
- Existing resident-reader malformed-file/excluded-key regressions.
- Actual isolated Flint long-context app turn against existing F25 baseline: stage
  phys_footprint/MLX active+cache, checkpoint SSD bytes, TTFT, prefill/decode, terminal
  write/read wait, normal output, and next-turn disk reuse. Inspect Activity Monitor.

Remaining architecture work: recurrent companion snapshot lifetimes in BatchEngine,
TokenIterator, NativeMTP/DFlash/DSpark; process-wide incoming write-payload reservations;
peak ownership deduplication; full-quota active parent/child protection. The existing
29 GB peak remains unclosed until the modified app is measured.
