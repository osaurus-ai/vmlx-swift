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

## Candidate runtime and native observations

Runtime `f1747342d881ba7b8b6a3bed84fd5fd60b99e96c` has now been exercised
through isolated Release development apps. Runtime `c3f3cb1d13a330b37a00666ff037b369c73e5db1`
only formats the regression test; its production sources are unchanged.

The E4B MXFP4 q6 upgrade/restart rows preserve seven old F32 payloads while
writing and restoring new BF16 payloads in the new namespace. Actual calculator
continuations return 3154 and 9462. A real image row reads the code and amount
and calculates 8194, but its next turn selects an older tool result instead of
8194. A later restart restores SSD state and completes another calculation.
Thinking Off still produces visible reasoning in the first process. These are
partial quality/settings results, not an all-pass model-family claim.

On consumer `d409fb86cf5acd5ab83fe984662a2511c05850da`, E4B JANG_4M q6
writes F16 KV with four full-attention and twenty rotating cache layers;
TurboQuant and paged RAM cache are off. A real terminal workflow waits through
15 silent seconds before a separate dependent hash command, with both commands
exiting successfully and the final nonce/hash matching the file. The calculator
workflow is partial: its post-tool request produces zero tokens, then existing
consumer code injects an empty-turn prompt notice before the answer. Another
follow-up uses an old memory value rather than the current chat result. Those
failures remain open. Three exact-prompt direct API replays with prefix cache
On/Off/On answer correctly, but stochastic replay alone does not prove fidelity.

Consumer `5be949779fb11dabefb54de287f2137d05096392` adds only the missing
localization key. All ten consumer CI checks pass on that head. Runtime #726
and consumer #3315 remain drafts; runtime Mac/CUDA checks are queued and
the existing broad formatting check fails. Green consumer CI is not native
acceptance or authorization to skip the outstanding runtime gates.

## Actual-model cache numerical diagnostic

An independent, private Swift executable uses the same VLM loader/prepare path,
q6 setting and full native metallib for the actual E4B JANG_4M bundle. A comparison
of 556 production Swift files against the native app's runtime checkout finds
no differences; MLX core and C bindings also have the same commits. The copied
production tokenizer bridge and Gemma processor reconstruct the actual empty
request. Both the 4,760-token prefix hash and complete 4,816-token hash match
the existing native SSD index, and decoded prompt bytes match after removing
the dump's diagnostic header. No sampling, seed, prompt or model default changes
are involved.

| Comparison of 262,144 next-token logits | Maximum absolute difference | Total variation | Top token |
| --- | ---: | ---: | --- |
| Repeated cold prefill | 0 | 0 | The |
| Fresh prefix split versus cold | 0.0759082 | 0.00174284 | The |
| Original native SSD prefix versus cold | 0.0672073 | 0.00105679 | The |
| Fresh disk round trip versus identical in-memory split | 0 | 0 | The |

The fresh round trip uses the real serializer, safetensors file, restore path
and remaining 56 prompt tokens. All 262,144 final logits are bit-identical to
the same in-memory prefix. This isolates that serialization case from the
smaller arithmetic differences associated with changing prefill boundaries;
it does not establish universal numerical parity or a tolerance policy.

The runtime's own sampler filter, with the bundle's temperature 1, top-p .95 and
top-k 64, retains EOS token 1 with probability 0.0134212 cold and 0.0134872 with
the original SSD prefix. EOS 50 and 106 are filtered out in these rows. An
immediate sampled EOS is therefore possible without a cache error. The original
terminating token and RNG state were not recorded, so its actual cause remains
unproven. Do not suppress EOS, change sampling, or count the consumer's forced
notice as a clean result.

Source snapshots, commands, failed diagnostic attempts and numerical outputs are
bound by `REAL-MODEL-CACHE-FIDELITY-RECEIPT.json` under
`/Users/eric/vmlx-private-evidence/goal-storage-cache-resume-20261104/`.
The first reader included the dump metadata as input; its token hash gate
rejected the comparison before loading weights. A later attempt selected a
stale automatically discovered shader and failed before forwards; the successful
runs explicitly select the native app's byte-identical full metallib. These
diagnostic failures are retained separately from product runtime findings.

## Remaining acceptance work

Current-app localization/menu proof, actual head-codec change/save/reload,
JANG media/restart controls, and the complete affected ten-bundle settings and
scored harness matrix remain required. The earlier ten-bundle native matrix
contains OCR, tool compliance, reasoning and RAM failures; those remain open.
The empty-response handling and stale-memory grounding require separate root
diagnosis and affected native/multi-turn/harness proof before any loop change.
Normal runtime merge, repinning the consumer to the actual merge SHA, and
affected integration/CI proof are still pending.

Owned packed loading is retained by explicit user choice, with physical
footprint reported honestly. The original native JANG process peaked at
13,889,939,224 bytes. The numerical diagnostic's observed peak was
11,929,150,272 bytes across nine samples in its first completed run; collection
started after token preparation, so this is not a full-startup memory gate.
No low-RAM certification or speed gain is claimed. Recorded generation rates
are correctness observations, not controlled speed benchmarks. No BF16/source
bundle, audio, release tag or publication is in scope. The broader storage/cache,
terminal wait, native MTP, DFlash2 and DSpark goal remains active.
