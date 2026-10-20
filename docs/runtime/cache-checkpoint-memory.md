# Hybrid checkpoint memory lifecycle

Canonical hybrid prompts replace their generation suffix on the next turn. The captured suffix-stripped checkpoint is reusable; the exact prompt is not. Non-trimmable recurrent caches cannot produce older boundaries by trimming that exact prompt snapshot either.

The AR and native-MTP iterators therefore keep the captured boundary instead of retaining a redundant exact prompt copy. For disk-only coordinators they publish that owned checkpoint after prefill, and release their references after the coordinator accepts the complete boundary. The asynchronous writer owns its arrays until publication finishes. A rejected store leaves the checkpoint available for the terminal store. Paged caching keeps its existing finalization route.

Terminal stores reuse already-owned snapshots. Live post-answer state is still copied before serialization so subsequent cache mutations cannot change the writer's payload. A non-trimmable cache is checked before attempting a boundary copy. Dtypes, cache salts, chain ownership, quotas, sampler settings and speculative controller behavior retain their existing contracts.

## Verification status

In progress on `perf/cache-ram-ssd-oct20`. Do not treat this document as merge approval.

Required checks:
- Exact KV and recurrent state after disk reopen, including a delayed writer whose caller releases its checkpoint.
- Failed publication leaves the checkpoint retryable; disabled disk, paged caching and invalid offsets do not report successful early spill.
- AR and native-MTP boundary capture, terminal storage, media salt and warm continuation regression suites.
- Built-app growing chat with tool and media continuation, Auto/Off, disk restore, RAM stages and writer completion. Report physical footprint separately from MLX active allocation.

Scope limits:
- DFlash2/DSpark and BatchEngine have separate snapshot ownership paths. This change does not claim to eliminate their retained checkpoints.
- Transient writer-owned buffers are still required. Early enqueue does not mean the bytes are already durable or no longer in RAM.
- This is not a change to long-context attention kernels or adaptive speculation policy.
