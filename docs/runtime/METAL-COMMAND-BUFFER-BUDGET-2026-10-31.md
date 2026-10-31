# Metal command-buffer storage-byte accounting

Core PR [#35](https://github.com/osaurus-ai/mlx/pull/35), commit
`7495d1165da54ef029a1af4c319ebe86f2349896`, corrects the Metal command-buffer
budget to multiply storage elements by dtype width. The previous counter
underestimated float16/bfloat16 inputs by 2x and float32 inputs by 4x.
The existing storage-size semantics, buffer deduplication, defaults and
commit comparison remain unchanged.

The runtime consumes the canonical Core submodule change. There is no duplicate
Swift workaround, driver flag, new limit or model-generation override.

## Validation

The canonical Metal tests fail three dtype assertions on the old backend and
pass 3 cases / 23 assertions on the corrected backend without skips. The tests
cover float32, float16, bfloat16, uint8, broadcast storage and aliased inputs.
The same canonical tests also pass against the uninstrumented merged Core
implementation compiled with the actual Osaurus backend build arguments.
Four real GPU old/new output hashes match under an explicit 1 MiB diagnostic
budget; this checks parity, not an independent numerical oracle.

An additional existing primitive cross-check passes 6/7 cases and 322 assertions.
The sparse-attention case cannot load its shader function from the selected
prebuilt library. The unchanged backend reproduces the same exception; this
remains a separate prebuilt-library qualification gap.

This is a proven byte-accounting correction. It does not establish that the
reported M2 Max interactivity error is caused or prevented by this defect.
Affected-hardware reproduction and prevention remain open. No release is
part of this change.
