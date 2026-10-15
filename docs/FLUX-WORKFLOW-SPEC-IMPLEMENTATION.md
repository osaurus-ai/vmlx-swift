# Workflow media spec implementation

Active scope: the entire https://gist.github.com/tpae/bb9f4a8384abe07446448128bac24c7b, authorized 2026-10-14. Each item requires engine implementation, Osaurus consumption, automated contract tests and affected live-app evidence before promotion. No release is part of this lane.

## Ledger

| Item | Status | Remaining acceptance |
| --- | --- | --- |
| 1.1 cancellable upscale | Delivered for SeedVR2 3B | Preserve existing cancellation/recovery proof |
| 1.3 effective parameters | Delivered for Qwen2.1/Turbo and SeedVR2 | Other implemented families; per-family replay |
| 1.4 typed limits | Delivered for Qwen2.1/Turbo and SeedVR2 | Other implemented families; validation before weight loading |
| 1.2 native batch | In progress | Encode once, indexed completions, explicit seed list, cancellation, individual-image byte parity, wall time |
| 1.5 sampler/scheduler/shift | Pending | Real selectable implementations, capability query and invalid-field errors |
| 1.7 previews | Pending | Bounded cadence/size, cancellation, overhead <=5% |
| 1.8 memory estimates | Pending | Per-request estimates within 15% of measured peaks at required dimensions |
| 1.6 adapters | Pending | Supported targets, content identity, reuse, zero-scale and removal parity |
| 1.9 video | Pending | Real weights/text encoding; 49-frame480p clip, metadata and cancellation |
| Stage2 | Pending | Conditioning/latent handles, encode/mix/resize/sample/decode/release; parity, ownership, memory release |

Preserve existing seeds and independent random streams. Do not confuse reported sampler metadata with selectable implementations. Do not advertise a scaffold as an implemented backend. Keep Turbo's bundle-defined schedule. Existing singleton image admission and chat-model unload/restore remain authoritative. A source-only or RunBench row is not live-app proof.

Private evidence and historical proof: `~/vmlx-private-evidence/flux-workflows-20261014/`. Full-scope source starts on fresh main in `feat/flux-full-spec-oct14`, using the existing isolated engine/app worktrees; prior dirty default checkouts are untouched. Allosaurus PRs599/3097 remain separately held and their evidence is preserved.
