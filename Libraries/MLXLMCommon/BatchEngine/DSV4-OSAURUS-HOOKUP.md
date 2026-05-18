# DSV4 + osaurus integration recipe

> **Scope**: getting DeepSeek-V4-Flash JANGTQ working end-to-end through
> osaurus on top of vmlx-swift-lm. Companion to `OSAURUS-INTEGRATION.md`
> (general consumer surface) and `JANGTQ-RUNTIME-PATCH-GUIDE.md`
> (loader behaviour).
>
> If you're seeing **multilingual gibberish output** (`"matters Reasons | claiming…"`,
> word-soup with mixed CJK/Cyrillic/Latin tokens), 9× of 10 it's one of the
> three causes in §5 below. Read that section first.

---

## 1. Pin verification — do this every time you debug a regression

```bash
# In the osaurus repo
cat Package.resolved | jq '.pins[] | select(.identity == "vmlx-swift-lm")'

# Expected: revision >= origin/main fa77575 (2026-04-26) or newer
# Pin staleness is the single most common cause of "loads but
# garbage decode" because the JANGTQ Metal kernel + mxtq_bits
# resolution + weight_format auto-correct all shipped recently.

# Then force a clean rebuild — Package.resolved alone doesn't
# guarantee the BUILT BINARY links the resolved revision:
rm -rf .build && swift build -c release
```

Sanity-check the binary actually links the new code:

```bash
# This symbol was added in fa77575 — if it's not there, your binary is stale:
nm .build/release/osaurus 2>/dev/null | grep -i "sniffCodebookBits"
```

If grep returns nothing, the binary is older than `fa77575` and DSV4-Flash
JANGTQ bundles with mislabeled `weight_format: "bf16"` will fall through to
the affine `DeepseekV4Model` and trip `Unhandled keys ['tq_norms', 'tq_packed']`.

---

## 2. Bundle preflight — three checks before you blame the runtime

```bash
BUNDLE=~/.mlxstudio/models/JANGQ-AI/DeepSeekV4-Flash-JANGTQ

# (a) Does the bundle have a non-empty sidecar?
ls -la "$BUNDLE/jangtq_runtime.safetensors"
# Expected: > 1 KB. A 29-byte file is a corrupted upload — replace.

# (b) What does the sidecar contain?
python3 -c "
from safetensors import safe_open
with safe_open('$BUNDLE/jangtq_runtime.safetensors', framework='pt') as f:
    print(sorted(f.keys()))
"
# Expected: ['codebook.{N}.{B}', 'signs.{N}.{seed}'] — at least one
# codebook.*.* entry MUST be present.

# (c) Confirm jang_config.json's stamp + bits resolution path
jq '{weight_format, profile, mxtq_bits, routed_expert_bits}' \
  "$BUNDLE/jang_config.json"

# Expected for JANGTQ_2L: weight_format="mxtq", routed=2 somewhere
# Expected for JANGTQ_4:   weight_format="mxtq", routed=4 somewhere
# If weight_format="bf16" with sidecar present, our loader auto-
# corrects to "mxtq" with a stderr log [Load] line.
```

Bundles that PASS all three checks should load coherently. Bundles that fail
(b) need a re-download; we can't recover from a missing codebook.

---

## 3. Required loader / runtime config

osaurus already does most of this — these are the documented contracts:

```swift
// Load — same path as any model
let context = try await MLXLMCommon.loadModel(
    from: bundleURL,
    using: #huggingFaceTokenizerLoader())
// `context.model` will be DeepseekV4JANGTQModel for JANGTQ bundles,
// DeepseekV4Model for plain affine. Auto-detect handles bundles
// with mislabeled stamps via sidecar codebook sniff (fa77575).

// Cache coordinator — paged + L2 disk for cross-turn reuse
var coordCfg = CacheCoordinatorConfig()
coordCfg.usePagedCache = true
coordCfg.enableDiskCache = true
coordCfg.diskCacheDir = osaurusDiskCacheDir
coordCfg.modelKey = bundleURL.lastPathComponent  // disk-key isolation
let coord = CacheCoordinator(config: coordCfg)

// BatchEngine — maxBatchSize=4 is osaurus's documented default
let engine = BatchEngine(
    context: context,
    maxBatchSize: 4,
    cacheCoordinator: coord)
```

---

## 4. Sampling + chat template — the canonical DSV4 recipe

```swift
// User input — kwargs flow through additionalContext to the bundle's
// chat template. The local DSV4-Flash bundle has no tokenizer_config
// chat_template, so Swift's tracked DSV4Minimal fallback is the
// production path and honors:
//   - enable_thinking: Bool        — toggles <think> open vs </think> closed tail
//   - reasoning_effort: "max"|nil  — prepends max-effort system preface
//   - drop_thinking on prior msgs  — strips earlier reasoning blocks (multi-turn)
let userInput = UserInput(
    chat: messages,
    tools: tools,                          // optional; pass actual schemas, not placeholder text
    additionalContext: [
        "enable_thinking": isReasoningWorkload,    // false for casual chat
        "reasoning_effort": isHardProblem ? "max" : nil
    ])

// Sampling — normal chat can run deterministic validation settings.
// Product hard-reasoning traffic may use the bundle's sampling defaults.
var params = GenerateParameters(
    maxTokens: isReasoningWorkload ? 8192 : 512,    // 8k for reasoning, 512 for chat
    temperature: isReasoningWorkload ? 0.6 : 0,
    topP: 0.95,
    prefillStepSize: 512)
// kvMode: leave .none. DSV4's production compression is SWA+CSA+HSA.
// DSV4_KV_MODE=tq is a diagnostic override that trades away the hybrid pool.

// Submit + iterate
let prepared = try await context.processor.prepare(input: userInput)
let stream = await engine.generate(input: prepared, parameters: params)
for await event in stream {
    switch event {
    case .chunk(let text):     appendToContentBubble(text)
    case .reasoning(let text): appendToThinkingPane(text)        // DO NOT show as content
    case .toolCall(let call):  await dispatchTool(call)
    case .info(let info):      logCompletion(info)
    }
}
```

**Sampling note.** Earlier DSV4 debugging saw degenerate output when greedy
sampling was paired with stale pins or the wrong prompt mode. The current
2026-05-06 strict DSV4 gate passes at `temperature=0` for normal chat and
reasoning rows; max-effort reasoning uses a larger token budget and the
request/bundle sampling values as provided. If product traffic uses the bundle
defaults (`temperature=0.6`, `top_p=0.95`), keep the reasoning parser and
stop-state telemetry enabled so loops or length finishes are visible. Do not
add a hidden repetition-penalty rescue, top-k clamp, forced stop, or fake
reasoning close when a row fails.

---

## 5. Three causes of the "multilingual gibberish" symptom

If a JANGTQ bundle loads but emits `"matters Reasons | claiming aims allow…"`:

| Cause | How to test | Fix |
|---|---|---|
| **Stale vmlx pin** | `nm osaurus \| grep sniffCodebookBits` — empty? | `rm -rf .build && swift package update && swift build -c release` |
| **Prompt/sampling mismatch** | Check the rendered chat prompt tail, reasoning kwargs, token budget, and explicit request/bundle sampling settings. | Use `UserInput(chat:)`; `enable_thinking=false` for normal chat; allocate larger budgets for reasoning/max; preserve explicit request or bundle sampling. Do not add hidden repetition penalties or sampler clamps as a rescue. |
| **Empty/corrupted sidecar** | `ls -la $BUNDLE/jangtq_runtime.safetensors` — < 1 KB? | Re-download bundle (this is the 29-byte case) |

If all three are clean and gibberish persists, capture for the vmlx maintainers:

```
1. The exact prompt string passed to applyChatTemplate (tokens + decoded text)
2. The full GenerateParameters (temperature, topP, maxTokens, kvMode)
3. The vmlx-swift-lm SHA the binary actually links against
4. Whether osaurus is rendering .reasoning(String) as visible content or
   routing it to a separate thinking-pane UI
```

Without those four data points, vmlx-side debugging is guesswork.

---

## 6. Auto-detect logging — what you'll see in stderr

When a JANGTQ bundle is loaded, you may see one or more of these lines.
None are errors; all are informational:

```
[Load] sidecar codebook present (2-bit) — forced weight_format "mxtq" (was: "bf16"); fix the bundle's jang_config.json
```
Auto-correct fired because the bundle's `weight_format` was mislabeled.
The dispatch routes correctly anyway. Operator should patch the
`jang_config.json` at the source so this stops firing.

```
[Load] JANG shape walk produced 250 per-layer quant override(s) over default (bits=2, gs=32)
```
Per-layer quantization overrides emitted. Normal — JANG/JANGTQ
attention layers use 8-bit while routed experts use 2-bit/4-bit, so
overrides over the chosen default are expected.

```
[Load] config per-layer quant disagreed with safetensors shapes — patched 312 layer(s) from shape walk
```
Defense-in-depth correction — the bundle's `config.json` per-layer
overrides disagreed with actual tensor shapes. Shape walk took
precedence. Operator should fix the bundle's config at source.

---

## 7. Reasoning channel + tool format — auto-detected, no caller config needed

| Detection | Source | Result for DSV4 |
|---|---|---|
| Reasoning parser | `model_type == "deepseek_v4"` → `reasoningStampFromModelType` | `think_xml` |
| Tool format | `model_type == "deepseek_v4"` → `ToolCallFormat.infer` | `.dsml` (curly-quote `<｜DSML｜tool_calls>`) |
| Chat template | tokenizer template or Swift `DSV4Minimal` fallback | uses `enable_thinking` + `reasoning_effort` kwargs; the local Flash bundle currently relies on the Swift fallback |

Streaming events are split by the reasoning parser (post-`c98ae5c`):
- pre-`</think>` → `.reasoning(String)` events (route to thinking pane)
- post-`</think>` → `.chunk(String)` events (route to content bubble)
- DSML tool blocks → `.toolCall(ToolCall)` events (dispatch to tool handler)

Stray `</think>` markers from interleaved-thinking pathologies are
stripped silently in the think_xml family (NOT in harmony — see
parser doc-comment for the trade-off).

---

## 8. Two known DSV4 model-behaviour quirks (not bugs)

### 8a. Short-prompt thinking with `enable_thinking=true` may inline the answer

DSV4 (and Qwen3.6-A3B fine-tunes) sometimes generates the answer INSIDE
the `<think>` block on validation-style prompts ("Give me a 20-digit
number", "Hi", "What is 7+5?") and emits EOS without ever closing
`</think>`. The streaming pipeline routes everything to `.reasoning`,
`.chunk` is empty, `finish: stop`. UIs that only render `.chunk` see
an empty bubble while the answer lives in the thinking pane.

**Diagnostic signal — `GenerateCompletionInfo.unclosedReasoning`:**
The library detects this case automatically. Inspect `info.unclosedReasoning`
on the `.info(GenerateCompletionInfo)` event — `true` means the parser
was still inside reasoning when the stream ended (model never emitted
`</think>`). Use this to drive the UI fallback without re-instrumenting
the parser yourself:

```swift
case .info(let info):
    if info.unclosedReasoning {
        // model got trapped — surface a banner or auto-mirror the
        // last sentence of the accumulated `.reasoning` text into
        // the content bubble.
        ui.showTrappedThinkingFallback(reasoningBuffer)
    }
```

This is a 2026-04-26 addition; check that your vmlx-swift-lm pin is
≥ 510fe47 (search the binary for the `unclosedReasoning` symbol via
`nm` if in doubt).

**Other recommended handling** (osaurus-side):
- Maintain a per-model `AutoThinkingProfile` defaulting DSV4 / Qwen3.6
  to `enable_thinking: false` for chat workloads. Flip on for explicit
  reasoning intents (a "Reasoning" toggle, an `:explain`-prefixed
  command, etc.). This stops the loop before it starts.
- Cap `max_tokens` lower for chat (e.g. 512) so trapped reasoning
  terminates earlier with `finish: length` + `unclosedReasoning: true`,
  letting the UI fallback fire quickly.

**A/B-verified diagnostic data (M4 Max, this session):**

```
prompt: "Give me a random 20-digit number. Only return the number itself."

  Qwen3.6-A3B-JANGTQ-CRACK (bits=2)  enable_thinking=true   → LOOPS, unclosedReasoning=true
  Holo3-A3B-JANGTQ4 (bits=4)         enable_thinking=true   → OK, content="78901234567890123456"
  Qwen3.6-A3B-JANGTQ-CRACK (bits=2)  enable_thinking=false  → OK, content="384729105638472910"
```

The loop is NOT a quantization-bits issue (bits=2 loops, bits=4 doesn't on
the same architecture). It's a fine-tune-specific training pattern in the
Qwen3.6-A3B family that activates on validation-shaped prompts. The fix
is at the prompt-construction layer (`enable_thinking: false`), not at
the runtime layer.

### 8b. Multi-turn truncation cascade with low max_tokens

If `enable_thinking=true` and `max_tokens` is too low for the model to
close `</think>` before the cap, the truncated mid-reasoning text gets
injected verbatim into the next turn's prompt and the model goes
off-distribution (gibberish, leaked markers).

**Recommended handling**:
- Raise `max_tokens >= 4096` per turn for reasoning workloads.
- Or set `drop_thinking: true` on prior assistant messages before
  re-applying the chat template — the bundle's Jinja honors
  `message.get('drop_thinking', false)` for non-last messages.

---

## 9. Cache modes + long context

DSV4 ships with three cache modes via `DSV4_KV_MODE`. There is no
auto-pick from `parameters.kvMode`; global TurboQuant defaults do not
replace the DSV4 hybrid cache.

| Mode | Cache | Memory @ 8K out | Coherence | Use when |
|---|---|---|---|---|
| unset / `sliding` (default) | `RotatingKVCache(128)` on SWA-only layers plus `DeepseekV4Cache(128, cr)` on CSA/HSA layers | context-dependent; roughly a 90% KV-row cut at long context | production hybrid long-context path | chat, reasoning, normal traffic |
| `full` | `KVCacheSimple` on every layer; no CSA/HSA pool | ~360 MB | diagnostic full KV | debugging only |
| `tq` | `KVCacheSimple` → promoted to `TurboQuantKVCache` only when request kvMode is also `.turboQuant` | ~14 MB | diagnostic compressed KV, no CSA/HSA pool | memory experiments only |

The bench `BENCH_DSV4_FIM_VS_CHAT` covers all three modes at multiple
prompt lengths. Live-verified coherent on M4 Max + M3 Ultra (Mac Studio).

The Compressor + Indexer path is the proper memory-efficient long-context
solution and is now the default vmlx-swift-lm path through
`DeepseekV4Cache`. `DSV4_KV_MODE=full` is a diagnostic comparison mode,
not the osaurus serving path.

---

## 10. Live-verification matrix for vmlx-side claims (M4 Max, fa77575)

Every JANGTQ bundle on disk in `~/.mlxstudio/models/` was tested in
this session. All pass cache-hit + multi-turn + reasoning end-to-end:

| Bundle | bits | sidecar | dispatched class | result |
|---|---|---|---|---|
| DSV4-Flash-JANGTQ | 2 routed | ✅ | DeepseekV4JANGTQModel | ✓ coherent |
| DSV4-Flash-JANG_2L | 2 affine | absent | DeepseekV4Model | ✓ no false-positive |
| MiniMax-M2.7-Small-JANGTQ | 2 routed | ✅ | MiniMaxJANGTQModel | ✓ coherent |
| Holo3-35B-A3B-JANGTQ | 2 routed | ✅ | Qwen35JANGTQModel | ✓ coherent |
| Holo3-35B-A3B-JANGTQ4 | 4 routed | ✅ | Qwen35JANGTQModel | ✓ coherent (was broken pre-`b9779d1`) |
| Qwen3.6-35B-A3B-JANGTQ-CRACK | 2 routed | ✅ | Qwen35JANGTQModel | ✓ matches Python baseline |

**Qwen3.6-35B-A3B-JANGTQ4** was not directly tested (no full bundle on
disk — HF cache only has metadata) but uses the same Swift class
(`Qwen35JANGTQModel`) as the verified Holo3-JANGTQ4. By transitivity
it should work; download a complete bundle to confirm directly.
