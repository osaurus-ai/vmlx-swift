# Gemma QAT tied-head precision through the production loader

## Problem and correction

Optional conversion of a floating tied embedding ran before the loader's
existing parameter dtype materialisation. For the Gemma 4 MXFP4 QAT bundles,
the checkpoint embedding was F16 but the effective passthrough model used
BF16 parameters. Conversion pinned embedding output to F16 before that
materialisation. Mixing those activations with BF16 norm and layer-scale
parameters promoted decoder state, including persisted KV, to F32.

The optional head codec now runs after existing parameter materialisation.
It preserves the effective unconverted-head activation dtype. This changes
neither the bundle-driven sampler nor prompts, reasoning tags, model defaults,
attention kernels, parameter materialisation policy, or KV serialization.
Quantizing an embedding remains lossy; no bit-identical output claim is made.
Gemma JANG affine bundles retain their existing F16 preservation policy.

Cache arithmetic identity derives from the actual loaded embedding codec,
group size, mode and pinned output dtype. It composes with any existing model
computation identity in both the topology snapshot and coordinator namespace.
It never consults a later change to global head policy. Old state remains on
disk in its original namespace and cannot be restored as corrected state.

## Evidence and remaining gates

An unchanged Release development app using runtime
`ca5bdc98313aafd080475207cda4c29ec85392ee` ran the actual E4B MXFP4 QAT bundle.
Cold q6 and passthrough controls both started at zero restored tokens out of
2,069, then restored 2,069 out of 2,125 from SSD after the calculator result.
q6 persisted F32 KV; passthrough persisted BF16 KV. Both executed 83 * 19 and
returned 1577. Only saved head codec differed, apart from the generated current
datetime in the prompt. Live process logs showed the generic BF16 conversion
in both controls, with head quantization preceding it only in the q6 control.
An initial warm q6 run is retained but excluded from the cold comparison.

The production-loader regression uses tiny safetensor fixtures with the real
text and VLM model classes, dense and per-layer inputs, and MXFP4 and affine
decoders. The unmodified loader failed 44 assertions across four MXFP4 cases;
four affine controls passed. The corrected loader passes the original eight
q6 cases. Expanded tests cover 4/6/8-bit head conversion, embedding/logit/KV
dtypes, crossing a sliding-window boundary, codec namespace separation,
sync/async coordinator entry points and immunity to subsequent global policy
changes. Test results and exact source identity are recorded in the PR.

One expanded-test attempt incorrectly required equal strings for keys supplied
with differently ordered host components. That test failure is retained. The
corrected test requires equal components with one computation identity and
exact sync/async equality for the same host input; production code was unchanged.

Fresh candidate native multi-turn, media, typed SSD restore, restart and cache
upgrade checks are still required. Component tests do not qualify actual model
quality or performance. The earlier ten-bundle native matrix contains OCR,
tool compliance, reasoning and RAM failures; those remain open. Owned packed
loading is retained by explicit user choice, with physical footprint reported
honestly. Recorded rates are correctness observations, not controlled speed
benchmarks. No BF16/source bundle, audio, release tag or publication is in scope.
