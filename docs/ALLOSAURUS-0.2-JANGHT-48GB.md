# Allosaurus 0.2 (qwen4_exp JANGHT, 48 GB Macs): runtime contract and verification checklist

Branch `feat/allosaurus-48gb-janght` = `main` @ `f3586f93f` + `fix/allosaurus-wire-budget-oct15` (#608, which carries
#599) + `perf/nemotron35-fast-exact-verify-oct17` (#612) + the sub-2-bit JANGT port (`Q38-JANGHT-LOWBIT-SWIFT.md`) +
the Allosaurus 0.2 runtime work (PR #613; MLX fork fix osaurus-ai/mlx#25).

Status (2026-10-17): checklist items 1-6 verified on the real bundle in this runtime (see below and PR #613); item 7
(48 GiB hardware) is still open — memory was measured with MLX accounting on a 128 GiB Mac.

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

1. **Done.** Loads; coherent answers; thinking closes; tool calls and continuations work (in-app, osaurus #3125).
2. **Done.** `BENCH_FIDELITY`: 1-row-chunked logits vs the bf16 reference match vMLX Python within cross-runtime noise
   (top-1 75.0 vs 75.4 %, median KL 0.058 vs 0.057; v0.1 control 80.8 vs 80.4 %).
3. **Done.** `BENCH_ROWEXACT` W = 2/3/4: verify rows bitwise equal to the 1-row AR step (256/256, prose and code).
4. **Done.** Greedy decode 56 tok/s (M5 Max); JANGT prefill sort threshold kept at 512 (verify windows stay below it).
5. **Done, with an open item.** Adaptive MTP: protocol (`BENCH_SPEED` sampled, filler 4000, 1,500 tokens) code +21-32 %,
   prose -4..-21 % vs AR; in-app Auto vs Off tool turns +23-43 %, reasoning-heavy replies -2..-3 %. Open: AR-safety
   governor sensitivity at thin margins.
6. **Done (in-app).** SSD prompt cache restores on every turn including after tool results with MTP on (follow-up TTFT
   0.66-1.0 s). Prompt cache key carries the JANGT computation identity.
7. **Open.** 48 GiB hardware run. Measured on 128 GiB: MLX peak active incl. weights 29.3 GiB (8k) / 30.6 GiB (32k)
   with MTP, against ~30.9 GiB wire headroom on a 48 GiB Mac.
