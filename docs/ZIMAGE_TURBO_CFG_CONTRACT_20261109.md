# Z-Image Turbo conditioning contract

Status: candidate; runtime qualification, consumer proof and normal PR/CI/merge
remain required. No model-family or hardware qualification is claimed.

The installed OsaurusAI Z-Image-Turbo mflux 8-bit bundle identifies
Tongyi-MAI/Z-Image-Turbo and mflux 0.18.0. Its publisher distinguishes the
distilled Turbo checkpoint, which does not use CFG, from base Z-Image.
At mflux tag v.0.18.0, commit48e5ae662c003db697af733f677885739483ff28,
the Turbo ModelConfig sets supports_guidance=False; generate_image resolves
effective guidance to zero and does not encode a negative branch. Current
mflux preserves that distinction.

The Swift port registered only the Turbo pipeline but included a generic
negative-conditioning branch. Its limits advertised negative prompting with
guidance above zero. Actual native Osaurus controls accepted a negative prompt,
guidance1 and seed42, then reported CFG1 on a generated 512x512 image. That
image is retained as a reproduction of the contract violation, not a correct
Turbo CFG result. The ordinary CFG0 image is a separate retained row.

The correction removes the Turbo negative encoder/second transformer pass,
advertises no negative conditioning, and reports effective CFG0. Legacy request
options remain admitted but ignored, matching mflux's documented Turbo behavior.
This matters because the generic ImageGenRequest initializer carries guidance
3.5 even when a caller omits it; blanket rejection would break existing calls.
Non-finite guidance remains invalid. No request initializer, step default,
seed, sampler, schedule, tensor, model-loading policy or GPU kernel changes.

The companion app correction must derive offered negative controls from native
limits and suppress Turbo's unsupported guidance control. A merged SDK commit
does not establish downstream consumption. The SDK branch is based on main
1a227377fcca0e94ea8b6304f7cd9dff68791130; Osaurus currently consumes
1bcd74ad182103a3b83941f61e57ee0bac23b330. A consumer repin would also include
the separately investigated rotating-cache restore fix and must disclose and
qualify that scope rather than bypass its existing consumer evidence boundary.

The old exact Release image modules executed all three new XCTest cases.
The Turbo capability assertions fail, and the ignored-options case throws on
negative prompting at CFG0. The separate Qwen/FLUX/Ideogram contract case passes.
Original logs, source, binary and two dependency-metadata preparation failures
are retained. The existing test that endorsed the port's CFG branch is updated
to the publisher's Turbo contract, with other families' assertions preserved.

Candidate component qualification: all33 production image-module sources compile
in Release (exit0). Standard XCTest executes9cases:7pass,2preexisting
opt-in real-model cases skip,0fail. No GPU/model load occurs in that lane.
The companion app canonical Xcode run passes132/132 in23affected image suites,
with all5private environment flags verified in actual xctest PID39610. Its
public dependency remains1bcd for that component run.

A new ordinary native reference on the unchanged image implementation uses
the same installed8-bit bundle, blue-sphere prompt,512x512 and seed42, with
steps/guidance/sampler/scheduler left at UI defaults. It completes at4steps,
CFG0 in4.7s and produces a correct blue sphere on yellow. Its original PNG
SHA256 is e2c4d14f5a9e10692072aa5f57d8f72dd430ae432d7b9e89f462c82dd20b27a8.
Originals, metadata, screenshots and physical samples are retained; the app
quits normally. This is the old-path reference, not candidate equality proof.

Remaining proof includes native candidate legacy option/default behavior,
fixed-seed equality of the ordinary path, actual native/API effective values,
honest UI capabilities, cancellation/drain/recovery, model counts/resolution,
affected chat/delegation flows, and normal exact-head CI. Source-level M2/M3
compatibility review does not substitute for physical affected-hardware proof.
The existing four-step native default is a separate audit question and is not
changed to disguise output quality or performance.

SOURCE EVIDENCE: NativeImageRequestContracts.swift, ZImage.swift and
ZImageNative.swift; pinned [mflux model configuration](https://github.com/mflux-community/mflux/blob/48e5ae662c003db697af733f677885739483ff28/src/mflux/models/common/config/model_config.py),
[mflux implementation](https://github.com/mflux-community/mflux/blob/48e5ae662c003db697af733f677885739483ff28/src/mflux/models/z_image/variants/z_image.py),
and [publisher model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/f332072aa78be7aecdf3ee76d5c247082da564a6/README.md).

LIVE EVIDENCE: private sentry-inference-20261108/
cache-settings-contract-audit-20261109/image-discovery-audit1 retains native
controls/results, both original PNGs, the frozen pre-generation model clone,
resource samples, pinned upstream sources and zimage-contract-fix1/baseline3.
Candidate component receipts are zimage-contract-fix1/SDK-COMPONENT-QUALIFICATION2.json
and app-candidate1/CANDIDATE-QUALIFICATION1.json. The ordinary reference is
image-discovery-audit1/zimage-default-reference1/OUTPUT-RECEIPT1.json.
No native candidate pass or production readiness follows from these components.
