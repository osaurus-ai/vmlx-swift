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

The consolidated runtime preserves upstream MLX Swift APIs while adding the
Osaurus production layer that used to live in `vmlx-swift-lm`:

- **Continuous batching** through `BatchEngine`, with per-slot KV isolation,
  image-mask isolation, SSM-state merge for hybrid models, and cancellation
  boundaries
- **Multi-tier KV cache**: paged in-memory L1, SQLite-indexed disk L2, and
  SSM/CCA companion tiers for path-dependent caches
- **TurboQuant KV compression** for large-context cache footprint reduction
- **Speculative decoding** surfaces for classic autoregressive drafting,
  DFlash, DDTree, and native MTP where the model bundle supports it
- **JANG mixed precision** from `jang_config.json`, including per-layer
  attention, MLP, and routed-expert bit widths
- **MoE, MLA, and hybrid SSM dispatch reduction** for routed and recurrent
  architectures where upstream Swift paid avoidable Metal kernel overhead
- **Native model-family additions** beyond upstream: Gemma 4, Mistral Small 4,
  Qwen 3.5/3.6 text and VL, DeepSeek V4, NemotronH, Hunyuan v3, ZAYA/ZAYA1-VL,
  and the JANGTQ variants of those families

## Why consolidate here

The older Osaurus runtime work lived across multiple repositories and package
identities. That made fast iteration possible, but it also made production
pinning fragile: Osaurus could accidentally mix a chat-template fix from one
checkout with a cache/runtime fix from another, or ship an app that had the
right model registry but the wrong tokenizer, parser, or MLX kernel behavior.

vmlx-swift is the consolidation point for that work. The package is meant to
make the runtime contract auditable from one revision:

- one SwiftPM dependency for MLX tensors, LLM/VLM model code, processors,
  chat templates, parser stamps, and cache-aware generation
- one place to pin Osaurus-facing behavior for Gemma, Qwen, Mistral, Nemotron,
  MiniMax, DeepSeek, ZAYA, JANG, JANGTQ, MXFP, TurboQuant, and omni/VL families
- one benchmark and production-gate surface for speed, coherency, multi-turn
  cache reuse, reasoning/tool parsing, media inputs, and low-RAM behavior
- no private local package paths or hidden per-app patches required to reproduce
  a shipped runtime

The practical goal is not only "Swift can load the model." The goal is that a
native macOS app can serve long-running local chats with predictable templates,
stable tool calls, correct cache topology, and competitive token throughput
without falling back to Python or embedding one-off runtime forks.

## Performance contract

The rows below preserve the performance target established in
`vmlx-swift-lm`, which is being consolidated into this repository. They are
included here because speed regressions are production bugs for local model
serving: a runtime that is coherent but half as fast is still not the runtime
Osaurus users expect.

### Single-stream decode (sustained tok/s)

| Model | Architecture | Upstream Swift | This fork | Python mlx_lm | Gain vs upstream |
|---|---|---:|---:|---:|---:|
| Qwen 3.5-35B-A3B | hybrid SSM + MoE | 41 | **103** | 94 | +151% |
| Gemma 4 26B-A4B | dense MoE | 27 | **87** | - | +222% |
| Gemma 4 E2B | dense | 120 | **121** | 128 | - |
| Mistral Small 4 119B | MLA + MoE | 16 | **70** | 45-50 | +338% |
| NemotronH 30B-A3B | hybrid SSM + MoE | 45 | **110** | 15.5 | +144% |
| MiniMax M2.5 172B | MoE (256 expert) | 14 | **46** | 51 | +229% |

Python baselines were measured on M3 Ultra 256 GB, which has about 1.5x more
memory bandwidth than the M4 Max benchmark host. When Swift matches Python on
M4 Max, Swift is faster per unit of memory bandwidth.

The large wins on MoE, MLA, and hybrid SSM models come from cutting Metal
kernel-dispatch overhead. The fork reduced graph-level `AsType` operations by
71-95% across those families. Dense models like Gemma 4 E2B were already near
optimal upstream, so their gains are intentionally small.

The root cause is concrete: both Python `mlx-lm` and Swift use the same
C++/Metal backend, but Swift scalar creation historically inserted extra
float32 values into bfloat16 graphs. Every inserted `AsType` is a separate
Metal dispatch. On Qwen 3.5-35B-A3B, roughly 1,100 extra casts per decode step
cost about 22 ms of pure dispatch overhead, which accounts for the gap between
41 and 103 tok/s. Fixes include scalar dtype inference, precise softmax usage,
sigmoid/cast cleanup, universal bfloat16 conversion, identity-weight dtype
fixes, MoE gate zero-out dtype fixes, and compiled GeGLU/SwiGLU activations.

| Model | Upstream Swift `AsType` ops | Consolidated fork |
|---|---:|---:|
| Qwen 3.5-35B | 1,176 | 60 (-95%) |
| Mistral Small 4 119B | 988 | 72 (-93%) |
| MiniMax JANG | 1,245 | 248 (-80%) |
| NemotronH Cascade | 562 | 161 (-71%) |

Contributor rule: every `MLXArray` scalar created on a runtime hot path must
specify `dtype:` when it interacts with model tensors.

```swift
// Bad: triggers an AsType cascade when the tensor is bfloat16.
MLXArray(someFloat) * bfloat16Tensor
softmax(x.asType(.float32), axis: -1)

// Good: preserves the graph dtype and avoids unnecessary dispatches.
MLXArray(someFloat, dtype: tensor.dtype) * bfloat16Tensor
softmax(x, axis: -1, precise: true)
```

### Multi-turn throughput

Multi-turn speed matters separately from single-stream decode because real chat
sessions grow context, reuse prefix/cache state, and exercise prompt prefill on
every turn. These rows use five turns at 256 generated tokens per turn.

**Qwen 3.5-35B-A3B** — long context, 21 to 10,932 tokens:

| Backend | Decode T1 | Decode T5 | Prefill avg | TTFT T1 | Overall T2-5 |
|---|---:|---:|---:|---:|---:|
| Python mlx_lm 0.31.2 | 122.1 | 106.1 | 1520 tok/s | 281 ms | 62.4 tok/s |
| vmlx-swift-lm | 106 (peak 111) | 96.2 | 1335 tok/s | 53 ms | ~58 tok/s |
| omlx 0.3.2 | ~83 | ~57 | broken streaming TTFT | broken | 47.0 tok/s |
| LM Studio 0.4.x | 107.5 | 25.5 (no prefix cache) | n/a | broken | 42.4 tok/s |

The Swift runtime had the fastest cold-start TTFT in that row, roughly 5x
Python, and was the fastest Swift-binding runtime: +37% over LM Studio and
+23% over omlx. It trailed Python by about 10% on long-context decode while
running on the lower-bandwidth machine.

**Gemma 4 26B-A4B** — short context, 25 to 5,499 tokens:

| Backend | Decode T1 | Decode T5 | Avg decode T2-5 |
|---|---:|---:|---:|
| vmlx-swift-lm | 98.2 | 86.5 | 88.4 |
| Python mlx_lm 0.31.2 | 71.6 | 78.1 | 77.4 |
| omlx 0.3.2 | 77.7 | 68.6 | 71.0 |

**Llama 3.2 1B 4-bit** is the dense baseline from the older benchmark set:
with 47 to 11,022 tokens of growing context, all four tested runtimes landed
within about 8% of one another. That is why the performance focus here is not
"Swift is always faster"; it is that Swift should not pay extra dispatch tax on
MoE, MLA, hybrid SSM, and quantized routed-expert graphs.

These numbers are not a substitute for current production gates. They are the
minimum performance lineage this repository must preserve while consolidating
the old `vmlx-swift-lm` runtime into the public `vmlx-swift` package.

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

## Engine modes

### Single-stream generation

Plain prefill and decode remain available through the normal upstream-style
container APIs. This is the baseline path for smoke tests, model bring-up, and
single-user chat.

### Continuous batching

`BatchEngine` admits multiple concurrent requests, runs prefill and decode with
per-slot isolation, and streams tokens back to each caller independently. Each
slot owns its own generation settings, KV/cache state, media mask, reasoning
salt, and cancellation lifecycle.

Osaurus defaults to the compile-friendly single-slot path for local app usage,
but the multi-slot scheduler remains part of the runtime contract for server
or agent workloads that intentionally trade single-request latency for
aggregate throughput.

### Multi-tier cache

The cache stack combines:

| Tier | Storage | Granularity | Persistence |
|---|---|---|---|
| L1 paged | in-memory block pool | token blocks with chained hashes | per-process |
| L2 disk | SQLite index + tensor payloads | prompt/cache hash | survives restart |
| Companion state | in-memory or disk-assisted | SSM, Mamba, CCA, hybrid pool state | topology-dependent |

Cache identity includes model/config salt, reasoning/media salt, and relevant
runtime topology tags so a cache written under one parser, media request, KV
mode, or MoE top-k setting is not reused under a different contract.

### TurboQuant KV cache compression

TurboQuant compresses ordinary KV cache layers for large-context inference and
skips non-KV companion caches automatically. It is not a substitute for
path-dependent SSM, CCA, rotating-window, or hybrid compressor/indexer state;
those cache types keep their native serializers.

### Speculative decoding

The runtime exposes classic autoregressive drafting plus DFlash, DDTree, and
native MTP-style launch plans where a model family has proven support. Greedy
speculative rows must preserve byte identity with plain greedy decoding.

### JANG / JANGTQ mixed precision

JANG bundles load automatically when `jang_config.json` is present. Capability
stamps such as `reasoning_parser`, `tool_parser`, `supports_thinking`,
`think_in_template`, `cache_type`, and `draft_strategy` are honored at load
time so Osaurus does not need per-bundle prompt hacks.

Nemotron Ultra JANGTQ_1L has two distinct rows today. The saved `8.335 tok/s`
speed-gate artifact is a Python/JANG reference row using bundle generation
defaults, not a Swift RunBench row. Current Swift production-default RunBench
rows are coherent and cache-correct, with prompt reuse warming across turns, but
decode remains speed-open around `3.6-3.8 tok/s`; do not describe the Swift or
low-footprint mmap/JangPress path as an 8-10 tok/s row until a fresh Swift live
artifact proves it.

## Model coverage

The target coverage inherits the old `vmlx-swift-lm` surface and extends it as
families are consolidated into this repository.

| Family | vmlx-swift target | Python mlx_lm / omlx | LM Studio mlx-engine |
|---|:---:|:---:|:---:|
| ZAYA / ZAYA1-8B top-1 CCA + MoE | yes | no native equivalent | no native equivalent |
| Ling / Bailing Hybrid recurrent GLA + MoE | yes | partial upstream architecture | no native equivalent |
| Hunyuan v3 / Hy3 | functional, speed-open | no native equivalent | no native equivalent |
| NemotronH / Nemotron Omni | yes | partial | no native equivalent |
| Mistral Small 4 MLA + MoE | yes | partial | no native equivalent |
| MiniMax M2 / M2.5 | yes | yes | no native equivalent |
| Qwen 3.5 / 3.6 text and VL | yes | partial by family | partial by family |
| Gemma 4 text and VL | yes | partial by family | partial by family |

This is why some performance rows do not have LM Studio or omlx columns: those
runtimes often cannot load the same bundle. Where multiple Apple-Silicon
runtimes can load the family, the expectation is head-to-head multi-turn proof,
not a single isolated decode row.

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

MIT License. See [LICENSE](LICENSE).
