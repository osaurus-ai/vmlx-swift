# Compiled rotating-cache window correctness

Single-token compiled decode must apply an explicit sliding window to logical token positions. The ring buffer's physical slot numbers are not token positions after wrap. Previously, a capacity 4/window 4 cache admitted only 3 slots at token 4 and zero slots at token 7. This also affected a smaller window inside a larger rotating cache. The corrected mask computes each rotating slot's age relative to the incoming token's write position; pinned prefix rows retain their original positions and are excluded when outside an explicit sliding window.

Promotion at the exact capacity boundary also left the next-write index equal to capacity. Dynamic slice update clamps that index rather than implementing ring rotation. Normalize it to the first non-pinned slot before writing and advancing counters.

## Runtime scope

This change covers one-token decode. `Evaluate.swift` installs compiled decode after prompt preparation (`setupCompiledDecode` at2237–2246); subsequent `next()` consumes the single sampled token (`step(previous:)` at2916–2921). `BatchEngine.stepCompiledDecode` passes the slot's next sampled token to the compiled forward (2818–2833). `BatchCompile.compileForward` adds the batch axis.

`CompilableRotatingKVCache` remains a public cache type and does not reject multi-token updates. This fix does not establish multi-token wrap, DFlash verification, or MTP verification correctness. The existing multi-token mask/write behavior must be audited separately before those paths reuse this cache. It does not enable compiled decode by default or change generation defaults.

## Regression matrix

`CompilableRotatingExplicitWindowTests` runs 30 cases in each eager/compiled mode (1020 attention steps total): capacity/window/keep=(4,4,0),(8,4,0),(8,4,2); seeds 0,capacity-1,capacity,2*capacity-1,2*capacity+1; original versus restored state/metaState. Zero queries/keys with sequential values provide an independent chronological mean oracle, alongside an exact allowed-slot-count assertion. Each test holds `MLXMetalTestLock`.

Baseline: explicit compile opt-in runs both tests and produces 5676 assertion failures. Default compile-disabled run skips the compiled test and produces 2838 eager failures. The opt-in is restricted to an isolated tiny-array diagnostic process. These tests are not model speed measurements or production Sentry crash reproductions.

Post-fix: new matrix 2/2 tests passed with zero skips and all 1020 attention steps; existing CompilableRotatingKVCacheTests 4/4 passed with maximum absolute logit difference 1.73e-6. Both suites used a per-process explicit compile opt-in; the ordinary default remains disabled. Actual Osaurus multi-turn, settings, and model-family evidence remains required before app/family claims. This change does not identify or close the originating exception in Sentry 19K.
