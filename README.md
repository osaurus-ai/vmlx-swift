# vmlx-swift

vmlx-swift is Osaurus's unified Swift inference stack for MLX-powered local AI on
Apple Silicon.

The project is designed to become the single production package behind Osaurus
model serving: language models, vision-language models, multimodal models,
chat templates, tool calling, cache-aware generation, and quantized runtime
paths through one Swift interface.

The goal is simple: a fast, reliable, native MLX inference engine that product
teams can embed without stitching together several runtime repositories by hand.

## Why this exists

Local AI apps need more than a model loader. A production runtime has to keep
model behavior stable across real chat sessions, not just pass a one-token smoke
test. It has to apply generation config correctly, render the right chat
template, keep reasoning and tool-call streams parseable, preserve prefix-cache
contracts across turns, and handle dense, MoE, hybrid SSM, linear attention,
vision, audio, and multimodal model families without hidden per-app patches.

vmlx-swift exists to make that runtime surface explicit and testable.

It brings together the Osaurus Swift MLX stack into one package identity:

- MLX tensor, neural-network, random, FFT, linear algebra, optimizer, and fast
  kernels
- language-model and vision-language-model runtimes
- tokenizer, generation, Hub, and model utilities
- Jinja chat-template rendering
- cache, batching, streaming, reasoning, and tool-call integration surfaces

## Current status

This repository currently starts as a pinned SwiftPM facade over the Osaurus
runtime forks. It is intentionally buildable and conservative before becoming a
full source monorepo.

The `VMLXSwift` product re-exports the public modules Osaurus needs from:

| Dependency | Revision |
|---|---|
| `osaurus-ai/mlx-swift` | `0a56f9041d56b4b8161f67a6cbd540ae66efc9fd` |
| `osaurus-ai/vmlx-swift-lm` | `b166896353b9c95d773de993990c20a0b5ba6905` |
| `osaurus-ai/swift-transformers` | `087a66b17e482220b94909c5cf98688383ae481a` |
| `osaurus-ai/Jinja` | `58d21aa5b69fdd9eb7e23ce2c3730f47db8e0c9d` |

The first release target is not a marketing wrapper. It is a compatibility
package with reproducible remote pins, no local package paths, and a documented
runtime coverage matrix for deciding when Osaurus can safely consume this repo
as its only MLX dependency.

## Current validation snapshot

The current Swift runtime proof is intentionally split by runtime path. A
single token/s number is not enough without the load mode, memory footprint,
cache topology, and parser state attached.

| Model | Runtime path | Evidence | Status |
|---|---|---:|---|
| Nemotron Ultra 550B A55B `JANGTQ_1L` | resident Swift decode | 9.8 tok/s | Proven with bundle generation defaults, coherent output, and no parser leak |
| Nemotron Ultra 550B A55B `JANGTQ_1L` | low-footprint mmap decode | 6.7 tok/s full-run, 7.7 tok/s tail | Coherent, cache-correct, low-footprint; first-decode cost remains speed-open |

The resident Nemotron Ultra row confirms the documented 8-10 tok/s Swift decode
class. It uses about 100 GB physical footprint and is not the same claim as the
low-footprint mmap/JangPress path. The current mmap path decodes at
6.7 tok/s over the full short run while staying under 2 GB physical footprint;
excluding the first decode step, the tail estimate is 7.7 tok/s. It proves
hybrid SSM disk-backed prefix-cache restore, including SSM companion hits and
cache-salt isolation, but the first-decode and low-footprint speed gap remains
open. See
[`docs/NEMOTRON_ULTRA_RUNTIME_STATUS_2026_06_06.md`](docs/NEMOTRON_ULTRA_RUNTIME_STATUS_2026_06_06.md)
for the exact commands and artifacts.

## Install

Use a revision pin for production apps:

```swift
.package(
    url: "https://github.com/osaurus-ai/vmlx-swift.git",
    revision: "<pinned revision>"
)
```

Then depend on the facade product:

```swift
.product(name: "VMLXSwift", package: "vmlx-swift")
```

Import the unified surface:

```swift
import VMLXSwift
```

## Build

```sh
swift package resolve
swift build --target VMLXSwift
swift build --product vmlx-swift
swift run vmlx-swift version
```

The repository also includes a consolidation check:

```sh
./scripts/check-consolidation.sh
```

That check resolves the package, builds the facade library, builds the CLI,
runs the CLI version command, verifies dependency graph output, and rejects
local package paths in package manifests.

## Runtime scope

vmlx-swift is intended to cover the full Osaurus local-inference surface:

- Text generation and multi-turn chat
- Vision-language generation with image placeholder and processor handling
- Multimodal and omni-model paths, including image, video, and audio inputs
- RADIO-backed vision encoders where used by supported omni models
- Parakeet-style audio encoder paths where used by supported omni models
- Jinja chat-template rendering with per-family template behavior preserved
- Reasoning streams, reasoning parser stamps, and reasoning on/off controls
- Tool-call formatting, parsing, and streaming
- Generation config propagation, including temperature, top-p, top-k, min-p,
  stop/eos behavior, and family-specific defaults
- Prefix cache, paged KV cache, disk cache, rotating/sliding cache,
  path-dependent cache, and hybrid SSM companion cache paths
- JANG, JANGTQ, JANGTQ-K, MXFP4, TurboQuant, JangPress, and related quantized
  runtime surfaces as they are supported by the underlying engine

## Production validation standard

A model family is not considered supported just because it loads.

For this repo, support means the relevant architecture bucket has a live,
repeatable validation row in `docs/RUNTIME_COVERAGE_MATRIX.md` and the runtime
path has been checked through real generation behavior:

- load succeeds without local-only paths or private bundle assumptions
- first token arrives through the expected scheduler path
- multi-turn prefix caching behaves as designed for that topology
- generated text is coherent for the target quantization format
- stop tokens and eos handling terminate correctly
- reasoning on/off controls reach the chat template and parser
- tool-call parsing works when the family supports tools
- VL or omni inputs bind to the correct media token and encoder path
- cache restore does not corrupt the next turn
- stream events reach the caller in the expected channel

The matrix is organized by architecture, not by marketing name, so equivalent
runtime hazards get checked across families:

- dense KV attention
- dense MoE attention
- sliding or rotating KV attention
- hybrid SSM or Mamba-style cache
- linear-attention cache
- ZAYA CCA cache
- DSV4 MLA and compressor cache
- vision-language and omni-model media pipelines
- reasoning and tool-call parser families

## Repository hygiene

This repo is meant to be safe to consume from a clean checkout.

Public files must not include:

- private model paths
- local developer package paths
- API keys, tokens, or credential-shaped placeholders
- generated attribution footers
- hidden dirty-worktree dependencies

Package manifests should use remote revisions or normal public version
constraints. If a local path is needed for development, it should stay out of
mergeable public state.

## CLI direction

The current CLI is intentionally small while the package is still a facade:

```sh
swift run vmlx-swift version
```

Planned CLI commands should make this repo testable on its own, without
requiring Osaurus as the harness:

- `vmlx-swift run` for one-shot text generation
- `vmlx-swift chat` for multi-turn chat-template and cache validation
- `vmlx-swift vl` for image and video-language smoke tests
- `vmlx-swift audio` for audio and omni-model smoke tests
- `vmlx-swift smoke-matrix` for architecture-bucket validation
- `vmlx-swift cache-report` for prefix-cache and disk-cache diagnostics

Those commands should use public arguments and config files, not private local
paths baked into the source.

## Migration phases

1. **Facade package**: current state. One import surface over pinned Osaurus
   forks.
2. **Source import**: vendor the required MLX, vmlx-swift-lm,
   swift-transformers, and Jinja sources while preserving product names and
   module boundaries.
3. **Standalone runtime CLI**: add first-class commands for text, chat, VL,
   audio, cache, and smoke-matrix validation.
4. **Osaurus repin**: move Osaurus to consume only `osaurus-ai/vmlx-swift`.
5. **Legacy repo retirement**: keep older repos only as mirrors, upstream sync
   sources, or historical references after this package can build and validate
   the runtime matrix by itself.

Distributed and JACCL-facing products are not re-exported in the initial facade
commit. They should be added only when the pinned MLX C distributed surface is
buildable and the runtime matrix has a dedicated validation row for that path.

## Maintainers

vmlx-swift is an Osaurus project. The package is built for Osaurus first, with
public APIs and validation standards intended to be usable by other Swift MLX
applications over time.

## License

License information will be added before the first public release intended for
external adoption.
