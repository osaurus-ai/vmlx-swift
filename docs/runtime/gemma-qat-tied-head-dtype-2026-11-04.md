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

## Exact media-history cache trajectory control

The current consumer's E2B MXFP4 q6 native run stores BF16 KV, with three full
layers and twelve rotating layers. Fifteen payloads validate tensor types,
token offsets, circular write-index bounds and file ranges. Fifty-seven
payload hashes listed by the two preceding controls remain unchanged. This is
the actual storage dtype; the admin API's `effective_kv_mode=fp16` is an engine
category, not a claim that these tensors contain F16. Paged RAM is off and
TurboQuant layer count is zero.

With Memory and Thinking explicitly Off, the three native workflows score one
pass and two failures: calculator 59 * 67 correctly returns 3953; the actual
image turn invents a nonexistent PDF path; after restart, the next tool call
uses 5953 * 2 instead of 3953 * 2. All six model requests emit output and rates.
UI rates are 99.4, 95.9 and 95.9 tok/s for these respective workflows, not a
controlled speed comparison. Observed peak physical footprint is
5,280,879,248 bytes. These results neither qualify the model nor establish
that Memory injection explains every earlier failure.

A separate real-bundle numerical control reconstructs the failed media request
through the production tokenizer, image preprocessing and Gemma processor.
Before loading weights it verifies 5290 prompt tokens, the 5002-token prefix,
5349 post-tool tokens, their exact native cache hashes, the decoded prompt,
actual persisted image bytes and media salt. It teacher-forces the exact
18-token native `calculate` call, including its wrong operand, comparing all
262144 logits at each step across cold, repeated cold, fresh split, fresh SSD
restore and original native SSD restore paths.

| Comparison across all 18 steps | Equal values / total | Worst absolute difference | Worst full-distribution TVD |
| --- | --- | --- | --- |
| Repeated cold / cold | 4718592 / 4718592 | 0 | 0 |
| Fresh SSD / same in-memory split | 4718592 / 4718592 | 0 | 0 |
| Fresh split / cold | 434982 / 4718592 | 0.613259882 | 0.033485396 |
| Original native SSD / cold | 530753 / 4718592 | 0.609848976 | 0.010827719 |

All paths have the same top-ranked token at every tested step. At the first
operand digit, cold prefers the correct `3` with probability 0.9100968 while
wrong `5` retains probability 0.0376182; native restore gives 0.9030914 and
0.0404297 respectively under the actual bundle sampler (T=1, p=.95, k=64).
Thus restore is not necessary for the wrong digit to remain sampleable. This
does not identify the original random draw, erase the cold/split differences,
or convert the failed native task into a pass. No prompt, sampler, EOS,
reasoning or harness behavior was changed. There are no sampled generation
tokens in this diagnostic, so generation tok/s is not applicable.

The numerical collector was invoked with an extra argument and exited 64;
that run has no physical-footprint measurement. The native workflows have
their own valid collectors. Source snapshots, commands, exact inputs and
results are bound by `REAL-MEDIA-TEACHER-CACHE-FIDELITY-RECEIPT.json`;
`BUILD25-E2B-MXFP4-NATIVE-RECEIPT.json` binds the separate UI evidence. Both
are under the same private evidence root. The earlier diagnostic package
manifest changed when this second executable was added; its original exact
bytes remain in `real-cache-logits-diagnostic/source-snapshot/Package.swift`.
The original receipt is retained unedited.

## Remaining acceptance work

Release 25 now has actual localization/menu and q6/passthrough change/save/reload
proof on consumer 5be949779. Both selections survive normal process restart and
produce different effective runtime cache keys. No production code changed for
these controls. The E4B JANG native results are:

| Workflow | Result |
| --- | --- |
| Passthrough calculator, history-dependent double, restart continuation | Three passes: 1333, 2666, 7998 |
| q6 new image and calculator | Partial: amount/color/5408 correct; code loses its final digit |
| q6 media-history follow-up | Fail: uses stored-memory 1577 and G918-NZ37 instead of this chat's 5408 and attached code |
| q6 media after process restart | Partial: real SSD restore and correct 5408/color; code remains truncated |

There are fifteen actual requests, including three direct API controls. All
have positive output/rates and none contains the empty-turn notice. The original
empty row remains a failure. In the new fixed media request, cache On/Off/On
restores 5460/0/5460 tokens out of 5463 and always selects the same wrong operand
1577. Four decoded prompt dumps, original plus replays, are byte-identical.
The exported image is intentionally redacted; replays restore the actual
content-addressed persisted image blob with its matching history reference,
not a guessed replacement. This proves cache reuse is not necessary for that
failure; it does not establish numerical media parity or the full root cause.
Twenty-nine new F16 snapshots validate all layer kinds, offsets and bounds;
the previous 53 payload hashes remain unchanged. The three owned app processes
and their collectors exit normally. Peak observed physical footprint is
14,576,364,512 bytes; no low-RAM pass is claimed.

E2B JANG has a separate matched native Memory setting control. The normal
per-agent switch is saved through the editor and survives relaunch. Old memory
and failing chats remain intact; Off uses a separate fresh chat so the previous
injected blocks do not remain in history. Thinking is explicitly Off in both
conditions; bundle temperature 1/top-p .95/top-k 64 and all other controls stay
the same.

| E2B JANG workflow | Memory On | Explicit Memory Off |
| --- | --- | --- |
| Calculator 59 * 67 | Pass: 3953 | Pass: 3953 |
| Actual image and amount + 3 | Fail: invents a file path and reports its read error | Partial: all three image fields and 7268 correct, but emits a visible reasoning pass despite Thinking Off |
| Restart, double first-turn multiplication | Fail: selects old 89 * 23 = 2047, then 4094 | Pass: actual 3953 * 2 = 7906 |

The twelve requests have positive output/rates. Captured raw input contains
`[Memory]` in all six On requests and none of the six Off requests. E2B stores
F16 with three full KV and twelve rotating layers. Twenty-eight typed snapshots
validate; the previous 82 payload hashes remain unchanged. The observed peak
physical footprint is 10,140,802,800 bytes. The Off control has three correct
semantic results, but only two complete passes because the media row violates
the selected reasoning behavior. These stochastic, separately rendered inputs
do not prove numerical causality, fix Memory On, or qualify all settings.

Actual stored-memory UI also exposes a misstated episode: it describes
7998 * 3 = 23994 although the corresponding executed tool was 2666 * 3 = 7998.
Together with the retained stale pinned facts, this is a concrete host-input
quality concern. No memory store, prompt, sampler, reasoning output or agent
loop has been patched to hide it. Rate observations span different UI, runtime
and request-duration intervals and are not a speed comparison.

The exact controls and failures are bound by
`BUILD25-CODEC-MEDIA-RECEIPT.json` and
`BUILD25-E2B-JANG-MEMORY-CONTROL-RECEIPT.json` in the same private evidence root.
The complete affected ten-bundle settings and scored harness matrix remain
required; seven bundle variants still need current-consumer controls. The earlier ten-bundle native matrix
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
