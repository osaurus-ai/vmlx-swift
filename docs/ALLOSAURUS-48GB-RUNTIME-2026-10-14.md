# Allosaurus on 48 GiB Macs: runtime correction in progress

Status: partial; no real 48 GiB hardware result yet.

The automatic JANGH allocator budget previously reverted to 70% of RAM if weights plus 4 GiB exceeded Metal's recommended working set. A 34.43 GiB Allosaurus bundle consequently received a 33.6 GiB limit. The allocator accounts mapped buffers and drains its reusable pool above its GC threshold. The correction clamps the automatic allowance to the smaller of the GPU ceiling and 85% physical RAM when the weights fit; explicit user limits and Strict are preserved.

`memorySafety.keepModelInMemory` is a persisted Boolean, default true, forwarded through both load resolvers. The loader applies the bounded residency target, or zero for Off/non-fitting models so the previous model's process-wide budget does not remain. The 48-to-below-64-GiB machine class retains a ten-GiB OS reserve. Smaller and larger machines keep their prior reserve. This does not change idle unload, n-gram SSD access, KV precision, samplers, kernels, or MTP acceptance/depth policy.

Native MTP now forwards completed model-prepare progress to the BatchEngine stream, translating segment-local counts after cache restores and prompt-boundary capture. No new forward calls or evaluation boundaries are introduced. Completion is emitted after evaluation of the first sampled token. RunBench records the first prefill stage independent of its detail string and prints originating failure metadata, which its previous zero-token report omitted.

## Verification so far

- 22 XCTest cases passed: nine JANGH load-policy, eleven native MTP boundary/progress, two failure-transport tests.
- Eight residency-policy tests passed, including 16-GiB safety and 48-GiB ceiling/reserve cases.
- BatchEngine MLX failure test passed both preparation/decode injected-error cases.
- MTP progress tests preserve forward lengths, input order and output tokens, including restored disk prefixes and segmented prefill.
- Isolated Release UI Off/Save/relaunch proved the setting reaches the loader (target zero). Native prefill progress was observed through Accessibility. App CI is green on fe9fc250e. Full device/performance acceptance remains open.

## Remaining gates and caveats

- Real M4 Pro 48 GiB recommendedMaxWorkingSetSize is unknown. Simulated pressure does not prove that ceiling.
- This allocator additionally caps GC at 95% of recommended working set; an insufficient ceiling cannot be solved by raising the Swift limit alone.
- Allosaurus proposal head adds about 0.33 GiB beyond root-shard accounting. Its actual load path and transient allowance must be included in proof.
- Need isolated Release UI toggle/persistence/reload, cold and restored multiturn, sustained prose/easy-code MTP on/off, clock/page-ins/phys_footprint/wired/SSD timing receipts.
- The original zero-token result is now traced to a Metal GPU timeout during iterator construction (details below). Preventing the timeout under target-device pressure remains unproven; do not call error reporting a performance fix.
- No release/tag or all-device performance claim.

## Live pressure result: not merge-ready

The Release app completed literal easy-code output at 66.7 tok/s without pressure and 20.7–31.1 tok/s with a 76 GiB CPU-locked balloon on the 128 GiB M5 Max. A logging-only allocator build measured 71.6 versus 28.0–30.4; removing the balloon while keeping the model loaded recovered 63.1. Prose measured 46.4 versus 16.7 tok/s. These were sampled outputs with natural stops, not identical-token A/Bs. No seed or sampler workaround was used. A 38.5 GiB allowance did not restore speed.

Actual tracked residency allocations covered active allocations at generation boundaries, but that counter is not proof that macOS has pinned every backing page. Apple documents that `requestResidency()` can postpone preparation when other apps compete for resources: https://developer.apple.com/documentation/metal/mtlresidencyset/requestresidency(). The 128 GiB host's residency budget is larger than the proposed real 48 GiB budget. Neither this balloon nor passing policy tests certify a 48 GiB device.

Approximate decode-window GPU medians from the existing live logger were 1601 MHz (easy code unpressured), 1614 MHz (both pressure allowances), and 1616/1617 MHz (prose unpressured/pressured). Thus a lower median GPU clock does not explain this gap. These windows use stored request/load/TTFT timestamps and are not a per-kernel profile.

## Original zero-token diagnosis

Retained unified logs for original RunBench PID 94171, 2026-10-14 19:38–19:41 local, identify:

- 19:40:22.254: Metal command-buffer execution aborted with `kIOGPUCommandBufferCallbackErrorTimeout`.
- 19:40:24.946: `generateLoopTask` reported iterator construction failed with the same MLX/Metal timeout.
- The run then reported zero tokens and cancelled. Its old harness omitted `generationFailure`; current RunBench prints it.

`Evaluate.swift` already transports this catch as a preparation-stage GenerationFailure. It is not evidence of a lost native-MTP head, a sequential-request busy rejection, or a normal empty model answer. The timeout occurred in the original 80 GiB Metal-balloon test; the logs establish its immediate cause, not the specific GPU workload/page-pressure mechanism.

New bounded controls without pressure and with 72 GiB Metal pressure completed both requests. The 80 GiB reproduction was stopped at reported system-wide free memory 0% during warmup, then all owned allocations were released. That operator stop is not reproduction of the historical timeout. These 96-token greedy diagnostics are not live-app/coherence acceptance.

Private detailed receipts remain in the campaign's codex-fix folder: LIVE-48GB-PROOF-STATUS.md, ZERO-TOKEN-FOLLOWUP.md, original-cancel-unified.log, existing-app-clock-windows.json. No kernel/controller changes or pressure-induced automatic sampler fallbacks were added. Actual 48 GiB hardware proof and the retained-speed gate remain outstanding.
