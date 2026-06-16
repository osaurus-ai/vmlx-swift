# Codex Configuration - vmlx-swift

See `~/AGENTS.md` for the global Codex environment, wiki protocol, hard rules,
machine context, and useful commands.

## Direct-Only Work Rule

For Gemma, Osaurus release, MLXPress, JANG/JANGTQ, cache-stack, or Sentry
runtime work in this repo, do not spawn recursive local agents, Python
subagents, delegated helper agents, Codex/Claude subprocesses, or local LLM
workers. Do the work directly in this session with source reads, patches,
builds, tests, app launches, curl/API probes, and proof artifacts.

Python may be used only as a deterministic parser or proof helper. It must not
orchestrate, delegate, prompt, or supervise another agent/model. This repo rule
overrides generic parallel-agent, subagent, or delegation workflow advice unless
Eric explicitly asks for that workflow in the current turn.

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
- Never add placeholder guards, fake pass/fail switches, hardcoded family
  allowlists, synthetic output filters, or "temporary" behavior enforcement to
  make JANG, JANGTQ, MXFP, VL, audio, or tool rows appear release-ready. A row
  is only fixed when the model's real bundle contract, runtime path, parser,
  cache topology, and live Osaurus behavior work together. If the proof is
  missing, mark it `PARTIAL` or `BLOCKED`; do not hide it in code.
- Default generation parameters must come from the active model's
  `generation_config.json` or equivalent bundle config. Chat/API/CLI defaults
  may pass user-explicit overrides, but must not silently invent sampler,
  repetition-penalty, or thinking-mode defaults to hide a runtime issue.
  Native-trained defaults such as top-k matter for quality and speed; preserve
  them unless the request explicitly overrides them.
- JANG, JANGTQ, MXFP, VL, audio, video, reasoning, and tool defaults must all
  resolve from the model bundle's generation/config/tokenizer/template contract
  plus explicit user settings. Do not fork special defaults by quantization
  type, media type, or model nickname unless the bundle or a documented runtime
  capability contract requires it.
- Reasoning/tool/chat behavior must be auto-detected from the model bundle,
  tokenizer, chat template, and declared runtime config. Do not synthesize fake
  thinking envelopes, strip visible output to hide parser bugs, or coerce a
  model family into another family's template.
- Memory limits and low-RAM behavior must be driven by the user's selected
  memory-safety settings and the resolved runtime plan, not hardcoded RAM
  percentages, fake global blocks, or hidden load refusals. If a model/request
  exceeds the resolved user policy or cannot be estimated in a strict mode,
  fail before unsafe MLX/Metal allocation with a typed user-facing error. In
  performance or explicit user-override modes, do not silently block loading
  merely because RAM is already full; let the documented user setting decide
  and preserve graceful failure if allocation is impossible.
- Low RAM means Activity Monitor physical footprint stays low. A load or
  generation row that reaches full model size in Activity Monitor is a failure,
  even if the output is coherent.
- Always record token/s when generation occurs. If no token/s is emitted,
  document that as a failed/blocked row.
- Coherency matters every time: visible answer, exact-copy/factual text when
  the prompt or tool result requires it, reasoning channel, no looping, no
  hidden reasoning-only output, and no length-cap fake pass.
- Weird visible text is a hard regression, not a cosmetic issue. Any response
  with corrupted ordinary words, dropped/duplicated letters, wrong copied
  numbers, mojibake, replacement/control/sentinel characters, protocol marker
  residue, or spelling/character drift such as `seleed`, `protctoocol`, or
  `28a` fails the row. A tool card plus cache hits is only `PARTIAL` if the
  final visible answer is corrupted; trace tokenizer, detokenizer, tool-result
  history, cache restore, sampling, and UI/API transport until clean text is
  proven.
- Multi-turn proof is required before calling a model family working. Single
  prompt or load-only evidence is not enough.
- Cache-stack proof must include the relevant cache topology: prefix/paged/L2
  disk hits, TurboQuant KV when enabled, and architecture-specific companion
  state for VL, video, SSM/linear-attention, or other path-dependent caches.
- Server setting wiring is a live-runtime gate. Do not call a setting wired
  because a source field, JSON key, or UI control exists. For generation
  defaults/overrides, tool availability/tool choice, memory-safety, prefix
  cache, paged KV, disk L2, TurboQuant KV, media/VL/audio controls,
  concurrency, and reasoning controls, read the current setting through the
  Osaurus API/panel/CLI surface, apply or toggle it, run a real live Osaurus
  chat turn, then verify the response, token/s, health, and telemetry changed
  as expected or returned a typed incompatibility/refusal. If a setting is
  next-load-only, reload the model and prove the new topology took effect.
- Cache proof must match the model architecture. Full-attention models need
  real KV/prefix/L2 proof; Qwen-style hybrid SSM needs KV plus SSM companion
  rederive/hit proof; ZAYA/CCA and HY3-style models need companion cache and
  pooling proof; DeepSeek-V4's CSA/HSA/SWA hybrid pool needs prefix/L2 plus
  pool restore/hit proof and must not use TurboQuant KV as a substitute.
- VL/video rows need real media payloads and cache-hit validation, not only
  text-path evidence.
- Audio rows need real audio payloads, processor/token wiring, cache behavior,
  and parser/output checks. Do not claim audio support from text-only Gemma,
  JANG, JANGTQ, MXFP, or VL evidence.
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
- Do not run validation/build/signing paths that trigger macOS Keychain,
  code-signing identity, notarization, `security`, certificate, or
  "wants to use your confidential information" prompts. Osaurus/vMLX testing
  must use noninteractive, keychain-free source tests, unsigned/no-signing
  builds only when explicitly safe, or live app/runtime probes that do not ask
  the user for a password. If a prompt appears, stop the lane, document the
  blocked artifact, and switch to a keychain-free proof path.
- Do not spawn recursive local "agent" workers, Python subagents, or delegated
  helper agents for Gemma/Osaurus release work unless the user explicitly asks.
  Do not use Python or shell wrappers as an orchestration layer to farm work out
  to Codex, Claude, local LLMs, or other helper agents. Work directly in the
  current session, keep status artifacts current, and use normal shell, test,
  build, and proof commands for evidence. Python is allowed for deterministic
  parsing or proof harnesses, but never to recursively run another agent.
  This overrides any generic parallel-agent or subagent workflow advice for
  this repo: do the Gemma/Osaurus release work here, with direct source reads,
  direct patches, direct builds, and direct app/runtime proof.
If any of RAM, speed, coherency, multi-turn, or cache/VL proof is missing, say
the row is blocked or partial and record the exact artifact path and reason.

## Active Gemma Release Focus

Until the Gemma checkpoint is explicitly merged or the user reopens scope, keep
Osaurus release work focused on Gemma MXFP4 and JANG_4M: E2B, E4B, 12B, 26B
A4B, and 31B. Do not drift into MiMo, NeX N2, Qwen MTP, or broad Sentry rows
while closing this Gemma PR unless a current Gemma proof directly depends on
that work.

For this Gemma release lane, the live proof must stay app-facing: Osaurus
chat/API multi-turn output, exact tool-call arguments, exact copied text and
numbers when requested, no corrupted ordinary words, no weird/control
characters, no loops, no protocol/reasoning/tool marker leakage, bundle-driven
generation config, token/s recorded, and cache telemetry from the active
architecture. Current Gemma cache proof is rotating KV plus disk-backed
restore/L2 where `turbo_quant_kv_layer_count=0`; do not claim TurboQuant KV for
those rows until the runtime reports and proves it.

Every Gemma 4 QAT MXFP4/JANG_4M model in this lane must also get a real
Osaurus tool-call and AgentLoop harness row before any benchmark score is used
for release judgment. The minimum useful row is: model loads from the
OsaurusAI QAT bundle, executes at least one real tool with exact name and
parseable JSON arguments, continues from tool results into clean visible text,
records token/s and cache topology with paged KV off, and produces a scored
harness artifact with every failed case attributed. A decent non-perfect score
can be teammate-testable; missing tools, corrupted text, marker leakage,
source/BF16 bundles, or unproven cache telemetry keep the row partial or
blocked.

## Osaurus PR / Sentry Crash Coordination

When two agents are active, split ownership clearly:

- Osaurus PR / engine agent owns the active Osaurus PR branch, app integration,
  engine pinning, runtime policy, panel/API behavior, and release-readiness
  proof.
- Sentry crash agent owns Sentry issue intake, local crash-library docs,
  reproduction attempts, root-cause traces, and focused vMLX/Osaurus crash
  fixes.
- Release coordination must keep exactly one comprehensive Osaurus PR for a
  release/runtime lane. Do not create multiple overlapping Osaurus PRs for the
  same Sentry/runtime work; carry all related Osaurus-side pins, guards, docs,
  and app integration through that one PR.
- vMLX engine/runtime fixes must land on `osaurus-ai/vmlx-swift` `main`. If a
  recent vMLX PR exists for the lane, merge it before final Osaurus pinning and
  pin Osaurus to the resulting vMLX main SHA, not to a temporary vMLX PR branch.

The active split is: the PR/engine agent keeps the Osaurus PR coherent and
mergeable, while the Sentry crash agent builds the reproducible crash-fix
library and prepares narrowly scoped patches or handoff notes for that PR.
Neither agent should mark another agent's lane complete without checking the
current issue writeup and proof log.

The Sentry crash library lives in:

- `.agents/sentry-crashes/2026-06-07-sentry-crash-ledger.md`
- `.agents/sentry-crashes/issues/`

This library is the canonical status source for crash rows. If an Osaurus-side
fix is found from this checkout, mirror the exact issue status, files touched,
verification commands, and remaining proof into the Osaurus repo's `.agents/`
handoff note before asking the PR/engine agent to carry it into the PR.

For every Sentry crash family, the Sentry agent must keep a separate writeup
with:

- Sentry issue IDs and release/device/model evidence.
- Reproduced symptom and exact local reproduction command or app flow.
- Root cause or current hypothesis, clearly marked when not proven.
- Real fix target and touched files.
- Verification status: `FIXED`, `PARTIAL`, `BLOCKED`, or `REPRODUCE`.
- If the failure is user/device/resource-side, the graceful in-app refusal or
  error path that prevents a native crash.

Each writeup must include a `PR handoff` note when the fix touches Osaurus:
which branch/repo currently contains the patch, whether it is source-only or
runtime-proven, whether it changes the vMLX pin, and the exact next command or
app flow the PR/engine agent should run.

Do not mark a Sentry issue fixed from source inspection alone. A row needs a
passing focused regression, a live keychain-free repro/proof, or a clearly
documented `BLOCKED` reason. Source-only guards, placeholder catches, crash
swallowing, retry-after-abort logic, and behavior-masking prompts do not count.

Before touching shared runtime files also being edited by the Osaurus PR /
engine agent, inspect the current diff and update the Sentry writeup with the
intended file scope. Do not revert or overwrite the other agent's changes. If a
fix must cross both repos, document the vMLX SHA / Osaurus branch pairing and
which repo owns the final PR.

For Sentry crash work, preserve these boundaries:

- Native Metal/MLX aborts must be prevented before entering the unsafe path; do
  not add fake catch/retry wrappers after process-fatal failures.
- Memory/OOM issues must fail gracefully before `loadModelContainer` or the
  first unsafe MLX allocation.
- Gemma/VL/router index traps must become typed runtime/model-config errors or
  correct shape handling, never silent clamping unless the bundle contract
  proves it is valid.
- Directory watcher, keychain, UI snapshot, and startup hangs are app-side
  graceful-failure/performance issues unless a trace proves vMLX involvement.

## Comprehensive Crash / Runtime Release Gate

Before asking to merge an Osaurus-facing vMLX or Sentry-fix PR, agents must
prove both crash safety and runtime correctness for the affected model families.
Keep the work in one coherent Osaurus PR plus the required vMLX main/PR change;
do not scatter related fixes across multiple Osaurus PRs.

Required evidence:

- Sentry crash fixes: every issue row must have a current writeup in
  `.agents/sentry-crashes/issues/` with issue IDs, affected release/model/device
  evidence, root cause or explicit hypothesis, files touched, and verification
  status. Mark rows `FIXED`, `PARTIAL`, `BLOCKED`, or `REPRODUCE`; do not blur
  those states.
- Crash prevention: resource, shape, router, media, cache, and model-config
  failures must fail before entering process-fatal MLX/Metal paths. Prefer typed
  errors and user-facing refusal messages over catch-after-crash wrappers.
- Model behavior: each claimed-working family needs multi-turn live proof with
  coherent visible output, no loops, no hidden reasoning-only answer, no leaked
  tool/reasoning tags, and no length-cap fake pass.
- Tool behavior: run tool-call rows that validate exact tool names, exact JSON
  arguments where required, verbose/built-in tool prompts, tool-result history,
  and required-tool fallback behavior.
- Reasoning behavior: verify bundle-driven reasoning detection and UI/API
  metadata. Reasoning selectors should appear only when the model contract
  supports them, and parser output must not leak internal reasoning wrappers.
- Cache behavior: validate the architecture-specific cache stack for the row:
  prefix/paged/L2 disk for full attention, KV plus SSM companion rederive/hits
  for hybrid SSM, VL/video media payload cache for multimodal rows, ZAYA/CCA
  companion/pooling cache when applicable, and TurboQuant KV only when it is the
  real configured path.
- Settings behavior: validate every Osaurus server setting touched by the
  release lane with read-toggle-chat-status evidence. Include tools and memory
  wiring in the same live multi-turn run where possible, and record settings
  that are unsupported, ignored, stale until reload, or incompatible as
  `PARTIAL`/`BLOCKED` instead of assuming they work.
- Memory behavior: large-context and low-RAM rows must report physical footprint
  and either run successfully within the gate or gracefully refuse with a clear
  in-app message. Do not enforce fake context limits or silently alter user
  behavior unless there is a real user setting and documented policy.
- Regression scope: edge-case tests must include the model families touched by
  the change and any nearby parser/template/cache code paths. Do not spend time
  on unrelated benchmark matrices unless they map to a Sentry issue, PR blocker,
  or affected architecture.
- Merge hygiene: keep `.agents/` private/ignored, keep `AGENTS.md` intentional,
  run keychain-free source tests/builds, verify CI, and record exact PR numbers,
  vMLX SHAs, Osaurus pins, artifact paths, and remaining truthful boundaries in
  the private ledger before final merge.
