# Allosaurus 0.2 (qwen4_exp JANGHT, 48 GB Macs): runtime contract and verification checklist

Branch `feat/allosaurus-48gb-janght` = `main` @ `f3586f93f` + `fix/allosaurus-wire-budget-oct15` (#608, which carries
#599) + the sub-2-bit JANGT port (`Q38-JANGHT-LOWBIT-SWIFT.md`). Status: **not yet verified on the real bundle in this
runtime.** Nothing here may gate an upload until the checklist below passes.

## Bundle contract

- `model_type` `qwen4_exp`, text-only: no `vision_config`, no vision weights. The text-only path synthesizes the
  processor configuration and refuses media.
- Routed experts: JANGT (`mode: jangt` per module, `kbits` per (layer, gate+up | down)), top-level `jangt` block with
  `code: v2_halfbits`, `state_bits: 12`, `rotation: hadamard128-signs`, `code_multipliers`, `step_bits`,
  `outlier_columns`. Widths present: K = 1, 1.5, 2. Tensors per projection: `jt_packed (E, N, words_per_row)`,
  `jt_scale`, `jt_su`, `jt_ocol`, `jt_wout`, and `jt_cs` on down. Gate and up are separate. Down input width is 640
  (not a multiple of 512): the masked tail block is required for every K.
- No `jangtq` codebook header (there is no `jangtq2` projection). `jang_config.format` reads `jangtq2`; do not use it to
  pick a decode path.
- Non-expert linears affine; lm_head 8-bit; measured proposal-head stamp `vmlx_mtp_proposal_head.json` (q4-eligible).
  MTP head (1 layer) is inside the main shards.
- PLE n-gram table: 128 tensors, ≈53 GiB, read from SSD (pread through the page cache), never wired, never counted as
  resident.
- Sampling defaults live in BOTH `generation_config.json` and `config.json::jang_config.chat.sampling_defaults`; they
  agree by construction (top_k 20, top_p 0.95, temperature below 1.0). The template's `reasoning_effort` default is xhigh.

## Memory on a 48 GiB Mac

- GPU-resident weights ≈27.1 GiB (bundle minus the n-gram table). XNU wire cap on 32–63 GiB Macs is 79 % (37.92 GiB);
  ≈7 GiB is wired by the OS; past the cap decode roughly halves (`vm.add_wire_count_over_global_limit`).
- `main`'s residency target (35 % OS reserve, floor 8 GiB, cap 24 GiB) gives 31.2 GiB on 48 GiB: fits with ≈2 GiB
  margin. #608 bounds the wired working set by weights + measured overhead under the live `vm.global_user_wire_limit`.
- Measure the serving peak at 8k and 32k context with the default prefill step; prefill transients must stay inside
  the wire budget. Estimates may warn; they must never refuse a load.

## Native MTP

- qwen4_exp is on the native-MTP allowlist; Auto cold-starts at D3 with the live acceptance governor (AR-safety
  pause/resume). Product modes: Off and Adaptive.
- Reference (vMLX Python, same bundle, sampled at the shipped defaults, adaptive): 75.9 % of drafts accepted
  (d1 85 %, d2 73 %, d3 61 %); the governor moves between D3 and D2 and calibrates against AR.
- Sampled verify must be exact rejection sampling; greedy verify accepts iff draft == target argmax. A multi-row verify
  forward is not bit-identical to 1-row decode, so greedy MTP-on output may differ from MTP-off on near-ties; measure
  1-row vs multi-row logits on the same tokens (KL, argmax agreement) to tell drift from a kernel bug.

## Prompt cache / SSD

- QSA layers persist KV + the raw indexer-key lane as one record; GDN recurrent state is stored natively.
- Tool-call turns store the stripped hybrid boundary before the continuation.
- The prompt-cache model key must change whenever the JANGT decode kernels change (the 640-tail fix changes logits);
  add the JANGT kernel identity to the key before shipping.

## Verification checklist (real bundle, this runtime)

1. Build `RunBench` (release). Load with `BENCH_PERF=1 BENCH_MODEL=<bundle> BENCH_PERF_USE_GENERATION_CONFIG=1
   BENCH_PERF_FULL_TEXT=1`, MTP off, then `BENCH_PERF_LOAD_NATIVE_MTP=1`. Coherent answers on short maths/science/
   counting prompts and one tool call; thinking closes; no loop.
2. 1-row decode logits vs vMLX Python on identical token ids: KL ≪ 0.01, argmax ≈ 100 %.
3. Multi-row (2–4 rows, verify-shaped) vs 1-row logits in Swift on the same tokens.
4. Decode/prefill speed (≥ 3 probes at a fixed condition, median); measure the JANGT prefill sort threshold on Qwen
   shapes (512 here vs 72 in vMLX Python) before keeping either.
5. Adaptive MTP: acceptance by depth, AR-safety trips, wall-clock vs AR, on prose, code and tool-call text.
6. SSD cache: partial hit after a tool result (same prefix, different suffix), MTP on; restored == cold (greedy);
   prompt lengths that are not multiples of the cache block.
7. 48 GiB hardware: wired bytes, `vm.add_wire_count_over_global_limit` stays flat, decode speed stable over a long
   multi-turn agentic session.
