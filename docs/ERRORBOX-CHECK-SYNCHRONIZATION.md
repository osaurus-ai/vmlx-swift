# ErrorBox check synchronization

`ErrorBox` is shared between scoped MLX error callbacks and code checking whether an operation failed. Its first-error getter and setter use `NSLock`, but `check()` read the same stored property without that lock. Concurrent publication and checking therefore raced even though the class declares `@unchecked Sendable`.

`check()` now reads through the existing locked getter. It still throws the first recorded error, does not reset the error, and does not replace it with later errors or nil.

## Evidence

A Foundation/Dispatch harness extracts the actual `ErrorBox` declaration unchanged from `Source/MLX/ErrorHandler.swift`. Two concurrent workers publish an error and call `check()` across 100000 boxes. Swift 6 ThreadSanitizer reports the baseline read in `check()` racing with the locked setter; the process aborts. With only the getter change, the same harness completes successfully with no ThreadSanitizer report.

This is primitive synchronization evidence. It is not a whole-package build, app proof, or a causal explanation for a production Sentry event. The package regression `ErrorBoxConcurrencyTests` additionally checks post-join visibility, first-error retention, nil-assignment behavior, and repeated checking. Its package execution is pending.
