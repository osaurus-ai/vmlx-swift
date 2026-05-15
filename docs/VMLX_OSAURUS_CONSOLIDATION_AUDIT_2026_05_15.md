# vMLX / Osaurus Consolidation Audit - 2026-05-15

This note records the live Osaurus pin state and the package-side consolidation
step in `vmlx-swift`. It is intentionally about wiring and release discipline;
it does not claim the model matrix is production-ready.

## Current Osaurus PR State

Latest Osaurus PR by `jjang-ai` checked during this pass:

- PR: `osaurus-ai/osaurus#1110`
- Title: `Harden DSV4 reasoning gates and runtime proof`
- URL: `https://github.com/osaurus-ai/osaurus/pull/1110`
- State: open, non-draft
- Branch: `feat/dsv4-vmlx-pin` into `main`
- Merge state at check time: `BLOCKED`
- Checks at check time:
  - `test-core`: in progress
  - `test-cli`: success
  - `swiftlint`: success
  - `shellcheck`: success
  - `update_release_draft`: success

Commits on PR #1110 at check time:

- `fed97d10` `docs(runtime): lock multimodal readiness checks`
- `8ec410e` `fix(dsv4): pin DSML template and live smoke`
- `814f3c2` `fix(runtime): harden DSV4 reasoning gates`
- `5763fa1` `docs(runtime): add release readiness proof`
- `79ab48b` `build(runtime): pin vmlx streaming tool test fix`

The PR touches the DSV4 parser/live-smoke path, tokenizer loader tests,
runtime readiness docs, and the runtime pin checker. It still depends on the
existing multi-package pin graph below.

Local `../osaurus-staging` is dirty during this audit, with changes in DSV4 live
smoke and sandbox lock tests plus untracked investigation material. Do not edit
or re-pin Osaurus from this package pass until that worktree is intentionally
settled or the exact Osaurus write scope is re-confirmed.

Previously referenced PR:

- PR: `osaurus-ai/osaurus#1073`
- Title: `Nemotron Omni live voice input path`
- State: merged on `2026-05-14T00:44:40Z`
- Merge commit: `27f357386eba713592b21f5c9631d0cee014d6eb`
- Checks: `test-core`, `test-cli`, `swiftlint`, `shellcheck`, and release
  drafter were green.

## Current Osaurus Runtime Pins

`/Users/eric/osaurus-staging/Packages/OsaurusCore/Package.swift` currently
pins the runtime stack as separate packages:

| Dependency | Revision |
| --- | --- |
| `https://github.com/osaurus-ai/mlx-swift` | `0a56f9041d56b4b8161f67a6cbd540ae66efc9fd` |
| `https://github.com/osaurus-ai/vmlx-swift-lm` | `c90898fb41955578d546cf8936acc813a53b0294` |
| `https://github.com/osaurus-ai/Jinja.git` | `58d21aa5b69fdd9eb7e23ce2c3730f47db8e0c9d` |
| `https://github.com/osaurus-ai/swift-transformers` | `087a66b17e482220b94909c5cf98688383ae481a` |

OsaurusCore currently imports products from those packages separately:

- `MLX`
- `MLXLLM`
- `MLXVLM`
- `MLXLMCommon`
- `Tokenizers`
- `Jinja`

That pin graph is exactly what this repo should replace after `vmlx-swift`
passes the runtime gates. Osaurus should not be moved to a single dependency
until the equivalent product surface exists here and the live matrix is proven.

## Moving Sibling Audit

`../vmlx-swift-lm` is currently not a clean upstream:

- It is behind `origin/main` by 5 commits.
- It has many dirty runtime/cache/template/model edits from parallel local work.
- It has untracked DSV4 reasoning-policy files whose behavior conflicts with
  the no-hidden-downgrade rule and must not be copied.
- It has in-flight batch/cache-salt test edits that should be treated as a
  signal to verify the same cache-key behavior here, not as an automatic patch.

Already reconciled in `vmlx-swift`:

- ZAYA/ZAYA1-VL nested JANGTQ_K metadata and `mxtq_seed` plumbing.
- VMLX umbrella product exposing the current Osaurus stable import surface.
- MTP status detection and preserved/disabled/error bundle metadata plumbing.
- DSV4 standalone `DSV4Minimal.jinja` no-system tool-schema rendering, aligned
  with the compiled Swift fallback and covered by a focused test.

Still open before the Osaurus single-package PR:

- DSV4 long-context CSA/HSA/SWA + prefix/paged/disk behavior, including vector
  drift status.
- BatchEngine continuous batching, cancellation, cache-key salting, and
  simultaneous session isolation with the current dirty sibling deltas compared
  one by one.
- JANGTQ active routed expert path: low physical footprint, usable token/s,
  coherent multi-turn, and no permanent prestack dependency unless explicitly
  diagnostic.
- VL/video/audio multi-turn media-salt behavior, including nil media salt on
  text-only turns and grounded output after media changes.
- SSM/hybrid async re-derive with companion-cache proof.
- Full Osaurus API/app gates after repin: HTTP routes, tray/process events,
  model picker, deep sleep/wake, packaged-app launch, and UI error surfaces.

## Package-Side Consolidation Added Here

`vmlx-swift` now exposes a single umbrella product:

- Product: `VMLX`
- Target: `VMLX`
- Source: `Libraries/VMLX/VMLX.swift`

The target re-exports the stable runtime modules Osaurus currently stitches
together in source:

- MLX core: `MLX`, `MLXRandom`, `MLXNN`, `MLXOptimizers`, `MLXFast`
- Template/tokenizer stack: `Jinja`, `Tokenizers`
- Model/runtime stack: `MLXLMCommon`, `MLXLLM`, `MLXVLM`

The `VMLX` target still depends on the broader package modules
(`MLXFFT`/`MLXLinalg`, `Hub`/`Generation`/`Models`, embedders/HuggingFace,
distributed, and `MLXPress`) so this package continues to build as the
consolidated runtime. Those modules are not all re-exported yet because some
public names collide when exposed through one import. Example: `MLXPressStatus`
exists both as a standalone `MLXPress` type and as an `MLXLMCommon` public
alias. The Osaurus migration should start with the stable current import set,
then explicitly promote any additional module only after ambiguity tests are
added.

Focused coverage:

- `Tests/MLXLMCommonFocusedTests/VMLXUmbrellaProductTests.swift`

This test imports only `VMLX` and then references representative public symbols
from the re-exported modules. That proves the Osaurus migration can move toward
one product import rather than carrying separate root dependencies for every
stable runtime layer.

## Intended Osaurus Migration

After this branch is pushed and validated, the next Osaurus-side change should
be small and mechanical:

1. Replace the four runtime packages with one `vmlx-swift` package pin.
2. Depend on `.product(name: "VMLX", package: "vmlx-swift")`.
3. Either keep existing explicit source imports for now, or migrate the Osaurus
   source imports to `import VMLX` one file group at a time.
4. Re-run the existing Osaurus pin checker with `vmlx-swift` as the canonical
   runtime pin.
5. Re-run the DSV4, Omni, ZAYA/VL, tool-call, reasoning, and cache-stack gates.

Do not claim production readiness from the package graph alone. The graph only
removes dependency fragmentation. Runtime readiness still requires live
multi-turn coherent generation, token/s, low physical footprint where relevant,
and cache-topology proof per model family.

## Non-Goals For This Patch

- No hidden DSV4 reasoning downgrade.
- No sampling clamps or fake EOS/repetition guards.
- No migration of Osaurus to the new product before this package is built and
  pushed.
- No replacement of behavioral proof with source-diff proof.
