// Copyright © 2026 osaurus.

/// Task-local callback used by model `prepare` implementations to report
/// completed prompt-processing units without changing the `LanguageModel`
/// protocol signature.
public enum PrefillProgressReporter {
    public typealias Handler = @Sendable (Int) -> Void

    @TaskLocal public static var current: Handler?

    public static func reportCompletedUnits(_ count: Int) {
        current?(max(0, count))
    }
}
