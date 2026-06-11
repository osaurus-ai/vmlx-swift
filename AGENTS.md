# Codex Configuration - vmlx-swift

See `~/AGENTS.md` for the global Codex environment, wiki protocol, hard rules,
machine context, and useful commands.

## Active Focus: Gemma Cache Defaults + Harness Proof

Until this lane is merged, focus only on the Gemma cache-default and harness
compatibility PR:

- Default model loading must not enable the paged RAM KV cache. Paged RAM cache
  may still be exercised by explicit benchmark or regression-test settings, but
  ordinary single-batch chat/server loads should start with it off.
- Gemma JANG and MXFP bundles must use real TurboQuant KV cache by default when
  loaded through Osaurus chat or Osaurus server settings. Prove this through the
  server runtime settings contract and runtime cache telemetry; do not use
  prompts, sampler tweaks, or parser masking as a substitute.
- Compare Gemma BF16/source bundles against the QAT/MXFP/JANG quantized bundles
  in the Osaurus harness compatibility lane. Record pass/fail counts, token/s
  where generation occurs, cache topology, and every failure cause.
- Add real prefill progress telemetry so Osaurus can show prompt-processing
  percentage during long first-token waits. Progress must come from runtime
  prefill/cache/media stages, not from a UI timer guess.
- Keep progress and remaining proof in the repo docs before claiming the PR is
  merge-ready. This branch is not a broad model-runtime cleanup lane.

## MLXPress Non-Negotiables

For any MLXPress, JANGTQ, cache-stack, model-runtime, or Osaurus-facing work,
agents must treat these as hard gates, not nice-to-have follow-ups:

- Never add forced-behavior "fixes" to make a model look coherent. This
  includes forced thinking tags, forced reasoning closers/openers, hidden
  repetition penalties, synthetic temperature/top-p/top-k overrides, decode-loop
  close-token biasing, or prompt/template coercion that is not part of the
  model's own template/config contract. If any such guard exists, treat it as a
  bug: document where it came from, remove it, and fix the real root cause in
  template parsing, runtime/decode, cache, matmul/Hadamard/2D/3D kernels, model
  loading, or API wiring.
- Default generation parameters must come from the active model's
  `generation_config.json` or equivalent bundle config. Chat/API/CLI defaults
  may pass user-explicit overrides, but must not silently invent sampler,
  repetition-penalty, or thinking-mode defaults to hide a runtime issue.
- Low RAM means Activity Monitor physical footprint stays low. A load or
  generation row that reaches full model size in Activity Monitor is a failure,
  even if the output is coherent.
- Always record token/s when generation occurs. If no token/s is emitted,
  document that as a failed/blocked row.
- Coherency matters every time: visible answer, reasoning channel, no looping,
  no hidden reasoning-only output, and no length-cap fake pass.
- Multi-turn proof is required before calling a model family working. Single
  prompt or load-only evidence is not enough.
- Cache-stack proof must include the relevant cache topology: prefix/paged/L2
  disk hits, TurboQuant KV when enabled, and architecture-specific companion
  state for VL, video, SSM/linear-attention, or other path-dependent caches.
- VL/video rows need real media payloads and cache-hit validation, not only
  text-path evidence.
- Unstacked routed JANGTQ must use the low-RAM active-streaming path by
  default. Do not write or load permanent prestacked routed overlays unless the
  user explicitly asks for an overlay diagnostic.
- Report-only memory-gate runs are diagnostics only. They never make a row
  production-ready.
- The historical JangPress target is resident compute with macOS reclaiming
  cold routed pages, not per-token SSD/active-bank streaming. A row around
  1 tok/s with tiny Activity Monitor footprint is still rejected for the
  user-facing methodology. Recovering the old MiniMax target means usable
  decode speed while staying below the family Activity Monitor gate, with
  coherent multi-turn output and low effective read pressure.
- Old JangPress MiniMax notes are a target, not proof: MiniMax Small measured
  about 43.74 tok/s at pct=70 with ~5.5 GB RSS post-decode and 22.9 GB
  reclaimed, but those rows predate Activity Monitor `phys_footprint` gates and
  had a chunk-buffer output bug. Current rows must re-prove speed, low
  footprint, and coherency together.
- Faster rows that loop are failures. MiniMax compiled-decode diagnostics are
  useful only after `KVCacheSimple` is promoted to `CompilableKVCache`; promote
  no compiled row unless it passes the same no-loop, no-length-stop, multi-turn
  coherency gates as the default decode path. The current MiniMax compiled
  TurboQuant-KV row closes the old speed/RAM target only for the explicit
  ephemeral-prestack diagnostic; it remains partial until explicit overlay
  dependence is removed.
- Activity Monitor gates must measure/enforce `phys_footprint`, not secretly
  throttle `MLX.Memory.memoryLimit`. Throttling can turn a valid MiniMax
  compiled row from ~49 tok/s into ~22 tok/s while footprint remains low.
If any of RAM, speed, coherency, multi-turn, or cache/VL proof is missing, say
the row is blocked or partial and record the exact artifact path and reason.
