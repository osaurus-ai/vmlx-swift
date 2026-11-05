# Batch cancellation terminal-metrics regression

The cancellation regression previously stopped consuming as soon as it called
`BatchEngine.cancel`. Receiving two tokens did not establish that cancellation
closed the stream or preserved its completion statistics.

The test now continues through stream closure, shuts the engine down, and checks
the cancelled reason, original prompt count, agreement between all received and
reported generated tokens, positive decode duration, and finite positive token
rate. In-flight tokens received after the cancellation request count toward that
agreement. Runtime generation, cache storage and app orchestration are unchanged.

## Validation

On the M5 Max2, the changed `BatchEngineTests.swift`, `TestTokenizer.swift` and
existing serialization helpers compiled into an isolated XCTest bundle. The
bundle linked the retained runtime objects whose production `Libraries` and
`Source` match `309c2a2bae541188164af54f58e0740046f459a4`, and explicitly loaded
the existing dev app's Cmlx shader. This was an incremental-object validation,
not a fresh full-package build. The ordinary Xcode build attempt failed to locate
its installed Metal toolchain; its logs remain retained.

- Changed cancellation test: 1 executed, 0 failures.
- Surrounding `BatchEngineIntegrationTests`: 30 executed, 0 failures, including
  queue/capacity, shutdown and cancellation/drain cases. Incidental synthetic
  throughput output is not a real-model speed comparison.
- Separate real Qwen3.8-27B-JANGH2 runtime control: requested cancellation after
  32 received tokens, consumed 33 total, received `.cancelled` with 33 reported
  tokens and 10.1198 tok/s. Bundle temperature 1, top-p .95, top-k 20; no sampler
  or reasoning override. This intentionally incomplete answer is not a quality
  pass or a speedup measurement.
- Actual dev-app HTTP control: two requests decoded together, one client
  disconnected, the other completed naturally; its follow-up returned the
  surviving identifier and restored 6,185 cached tokens. The native Live Activity
  screen showed one active slot after disconnect, high-water two, and zero
  decode splits. The unchanged GUI second-tab guard was not bypassed.

The disconnected HTTP request still has no retained generated-token count/rate
in Insights. Keeping a diagnostic consumer attached demonstrates runtime
terminal accounting; it does not repair or substitute for that HTTP metadata.
Native request-duration and SSE decode rates are retained separately. Physical
footprint exceeds the primary weight size: no low-RAM or complete model-family
qualification claim.

## Evidence

Retained local artifacts under
`/Users/eric/vmlx-private-evidence/goal-storage-cache-resume-20261104/`:

- `qwen27-concurrency/batch-cancel-regression/`: compiled test source, commands,
  shader bootstrap, link identities and XCTest logs.
- `qwen27-concurrency/batch-cancel/NATIVE-RECEIPT.json`: actual app request,
  cancellation/isolation, native UI, SSD restore and resource evidence.
- `qwen27-concurrency/runtime-cancel-info/`: real-model terminal-info control.

The separate DFlash2 long-reasoning and numerical-parity investigation remains
open. This change does not alter inference defaults or claim to fix that issue.
No release or tag is part of this change.
