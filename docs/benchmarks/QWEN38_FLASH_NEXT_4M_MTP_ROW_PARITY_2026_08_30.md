# Qwen3.8 Flash-Next 4M native-MTP row parity

Status: **engine PASS; Auto/API/Osaurus gates remain separate**.

This change fixes the 4M checkpoint's whole-model greedy-parity failure. It
does not enable MTP, change sampling, add a bundle-name allowlist, or change
the SSD-backed PLE/n-gram path.

## Reproduction and source trace

The `JANGQ-AI/Qwen3.8-Flash-Next-JANG_4M` bundle uses a uniform affine q4/g64
routed expert bank. Greedy AR evaluates one hidden row at a time, while native
MTP verifies two to four positions together. Before this patch, the complete
`SparseMoeBlock` made width-sensitive router, routed-expert, shared-expert and
post-reduction decisions over the verifier block. On a rejection-heavy Swift
code prompt, AR and D3 were deterministic but different:

| Arm | Tokens | tok/s | Output SHA-256 |
| --- | ---: | ---: | --- |
| AR | 257 | 32.4 | `b37a49f464c026df63bbb1efe21e6c78ebd9291be120d4f8811f1336d44b4ccb` |
| D3 | 315 | 49.8 | `bf2643f855bd0bef688866b1e84e8a4c9e450b30b3f3a5e8f3e99c11ed52246e` |

Disabling only the fused affine routed-MoE path made D3 reproduce the default
AR bytes, isolating the defect to the MoE width contract. An attempted narrower
fix that split only the final fused routed kernel passed its synthetic test but
failed the real model unchanged. That experiment was discarded rather than
presented as a fix.

The first retained change detected projection topology from each loaded block.
That was sufficient for the 4M proof but not sufficient to isolate the 4M
checkpoint: 4S has a mixed q2/q3/q4 routed bank, so an individual q4 block
could still enter the decode-equivalent path. The private MTP head also uses
the same `SparseMoeBlock` type and can carry q4 tensors. A follow-up therefore
moved the selector to the decoded bundle-wide quantization map. Every trunk
layer's gate, up and down routed projections must resolve to affine q4/g64;
the decision is passed only to trunk blocks and is explicitly false for the
MTP head. The complete MoE block then evaluates each verifier position with
the same `[1,1,H]` contract as AR before concatenating the rows. Mixed 4S,
q2 2L, TurboQuant and non-Qwen4Exp paths do not match that gate.

The spill was caught by a cross-quant live row: 4S D3 fell from its previous
~67 tok/s range to 36.9 tok/s while still producing correct bytes. On the
narrowed selector it returned to 67.4 tok/s (72.0 tail estimate), produced the
same full-text SHA-256
`20dbc5d7831d7a43c23bfa2cedb198965012b4d5b722d8c81a47271eb8c58b33`,
stopped normally, and used zero AR fallback tokens. This correction is why a
single loaded block's bit layout must never be used as a model-family proxy.

## Focused regression

Command:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  swift test \
  --scratch-path /Users/eric/vmlx-qwen38-d3-coldstart/.build \
  --filter q4VerifierUsesDecodeEquivalentMoERows
```

Result after the scope correction: 4/4 passed. The suite quantizes a complete
synthetic Qwen4Exp MoE block to q4/g64, proves the four-row result is exactly
equal to four width-1 calls, proves 2/6-bit blocks do not enter the new path,
and proves mixed 4S plus q2 2L model-wide maps cannot enable a locally q4
block.

## Matched live Release proof

Binary SHA-256:
`e564b4ae0bd98827b1969b070572d1a93a3f688309f780ead7d73a90a4494f1d`.

Bundle: `JANGQ-AI/Qwen3.8-Flash-Next-JANG_4M`.

Settings: Release `RunBench`, mmap weights, no JangPress, greedy sampling,
seed 42, warmup 1, fixed D3 for measurement. PLE reported
`backend=pread-fnocache cache=F_NOCACHE` against 28.832 GiB of SSD-backed rows.

Code prompt:
`Write a Swift function that returns the first n Fibonacci numbers. Include input validation and a short explanation.`

| Arm | Tokens | tok/s | Stop | Output SHA-256 | Peak footprint |
| --- | ---: | ---: | --- | --- | ---: |
| AR | 257 | 34.4 | stop | `b37a49f464c026df63bbb1efe21e6c78ebd9291be120d4f8811f1336d44b4ccb` | 73,913 MiB |
| D3 | 257 | 51.4 | stop | `b37a49f464c026df63bbb1efe21e6c78ebd9291be120d4f8811f1336d44b4ccb` | 77,222 MiB |

D3 was 1.49x AR. It used 76 target verification calls, accepted drafts at
all depths, emitted 57 bonus tokens, and used zero AR fallback tokens. Output
was coherent Swift with no loop, marker leak, or unclosed reasoning.

Sustained count workload, two measured runs after one warmup:

| Run | Headline tok/s | Tail tok/s | Acceptance | Result |
| --- | ---: | ---: | --- | --- |
| 0 | 58.5 | 75.4 | 256/256 verifier inputs | exact count prefix, length cap |
| 1 | 58.2 | 75.3 | 256/256 verifier inputs | exact count prefix, length cap |

The 256-token cap truncates the requested 1-to-120 sequence, so this row is a
throughput/acceptance measurement, not the coherency proof. The stopped code
row above is the coherency and byte-parity gate.

Raw local evidence:

`/Users/eric/vmlx-private-evidence/qwen38-4m-row-parity-20260830/`

## Python and accelerator audit boundary

The source-matched Python runtime measured 4S D3 at 88.79 tok/s and 4M D3 at
74.76 tok/s. Its winning rows did not activate CoreML, ANE, QSA, PLE/GDN,
affine-MoE pair fusion, RMSNorm fusion, or parallel-read fusion. The Swift 4M
tail rate of 75.3-75.4 tok/s therefore matches the relevant Python steady-state
rate; Swift's lower headline includes an approximately one-second first-decode
materialization cost.

CoreML/ANE is not added by this change (ANE access depends on private APIs). Any future TensorOps,
NAX, NPP, CoreML or ANE path still requires its own real-bundle parity,
footprint and wall-clock win before production use.

## Remaining gates

- Merge this bounded engine fix before changing the 4M tuning sidecar.
- Re-run the 4M AR/D1/D2/D3/adaptive matrix on the merged head.
- Prove bundle-driven Auto resolves D3 without measurement-only environment
  variables.
- Repin Osaurus and prove served API plus visible native Chat, multi-turn tools,
  Stop/continue and cache restore from the exact Release app.
- Treat the 73-77 GiB footprint as measured truth; MTP speed does not waive the
  separate memory-footprint work.
