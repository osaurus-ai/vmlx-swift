# Runtime diagnostic capture contract

macOS RuntimeDiagnostics intentionally writes private unified-log messages,
so the library survives closed, read-only and readerless-pipe stderr. Disk
cache tests still redirected descriptor2 and expected those messages there.
A broad cache run at102278d41 passed47XCTest but failed53 assertions in326
Swift Testing cases/22suites:52missing diagnostic assertions plus one separate
IO-gate worker startup timeout. The old global capture also reprinted another
test's failure. Storage/content assertions were retained, not weakened.

Add an internal task-local observer to RuntimeDiagnostics. It observes the
actual payload without replacing or making public the platform logger.
Nested/throwing scopes restore their parent; parallel tasks have independent
captures. Detached OS threads explicitly forward the observer. There is no
process-wide descriptor redirection and no new model, cache, RAM Safety,
sampler, retry, GPU recovery or inference policy.

The IO-gate regression used the shared global worker pool while other suites
occupied it waiting for the Metal test lock. Its dedicated worker queue can
reach the IO gate independently. Both before/after lock assertions remain.

First candidate8b0057f2c reduced the matrix to one failure (FIFO detached
thread did not inherit scope), which is retained. During development an errno
probe also failed (exit3:34rather than7); moving observer execution inside
macOS errno preservation makes it pass (exit0:7). These candidate failures
never qualified or merged.

Exact codeb0c173576df2a0153ca787fea6f091560fb3f994 on local Max2 passes:

- 45/45XCTest and332/332Swift Testing cases in23suites, zero failures.
- 6 new capture tests: exact/empty/return value, nesting, throwing restoration,
  simultaneous tasks (32x64messages), detached forwarding and errno.
- 6 child controls: closed/read-only/readerless-pipe stderr, observer off/on,
  100writes each, normal exit and preserved errno in every row. These compile
  the actual RuntimeDiagnostics source and verify descriptor flags before use.

The matrix includes disk index migration, hostile payload/value/row handling,
quota/accounting, write-behind, active readers, hybrid/media coordinator
contracts, resident readers, mapped banks and GPU architecture/kernel tests.
This branch has no SQLite failed-open repair; that independent#652adds two
XCTest cases and must run its combined matrix separately.

Private raw logs/receipts remain in
`/Users/eric/vmlx-private-evidence/sentry-allosaurus-20261024/`:
`sqlite-final-8264442da96d.log`, `diagnostic-capture-9d4f155d821d.log`,
`diagnostic-capture-final-9d4f155d821d.log`,
`diagnostic-observer-errno.receipt.json` and
`diagnostic-capture-broken-sinks.receipt.json`.
No native model/performance or GPU-watchdog reproduction is claimed here.


Final formatted code `87e8047e3a244d2252b97956ea6bf9b938e6e038` passes
45/45 XCTest and 332/332 Swift Testing cases in 23 suites with explicit
`swift test --no-parallel`. All ten touched Swift files pass strict local
swift-format603 lint. The library implementation is unchanged from b0c173576.

Later intermediate/combined parallel runs reset the process-wide MLX test
lock after90seconds. Those numeric passes are **INVALID as qualification**
and are retained. The final serial run has zero resets and zero issues;
individual concurrency tests still spawn their own parallel operations.
The proof runner now fails closed on a lock reset. A six-test scheduling
smoke verifies the explicit option. See [Swift Testing's parallelization
contract](https://developer.apple.com/documentation/testing/parallelization).
The earlier b0c173576 run also has no reset; it is not retracted.

Final receipt/log: `diagnostic-capture-serial-validation.receipt.json` and
`diagnostic-capture-serial-9d4f155d821d.log` in the private campaign root.
SQLite#652 still requires its own clean combined qualification.
