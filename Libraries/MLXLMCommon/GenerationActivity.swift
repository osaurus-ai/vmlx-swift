// Copyright © 2026 Apple Inc.

import Foundation
#if canImport(os)
    import os
#endif

/// One finite generation's process-activity lifetime, not a model-residency lease.
///
/// Task QoS alone does not prevent macOS from suppressing a hidden app while
/// its API is serving a request. Keep prefill/decode/cache finalization eligible
/// to run, but permit idle system/display sleep and end the activity on every
/// exit. Each loop owns its own instance; overlapping requests remain protected
/// until their respective scopes exit. This is deliberately not Sendable.
final class GenerationActivity {
    #if os(macOS)
        typealias Begin = (ProcessInfo.ActivityOptions, String) -> NSObjectProtocol
        typealias End = (NSObjectProtocol) -> Void

        static let options: ProcessInfo.ActivityOptions = .userInitiatedAllowingIdleSystemSleep
        private static let logger = Logger(subsystem: "vmlx", category: "GenerationActivity")
        private static let disabledByEnvironment =
            ProcessInfo.processInfo.environment["VMLX_DISABLE_GENERATION_ACTIVITY"] == "1"

        private var token: NSObjectProtocol?
        private let endActivity: End

        init(
            disabled: Bool = GenerationActivity.disabledByEnvironment,
            begin: Begin = { ProcessInfo.processInfo.beginActivity(options: $0, reason: $1) },
            end: @escaping End = { ProcessInfo.processInfo.endActivity($0) }
        ) {
            endActivity = end
            if !disabled {
                token = begin(Self.options, "Local model generation")
                Self.logger.debug("begin userInitiated=1 preventIdleSystemSleep=0")
            } else {
                Self.logger.debug("disabled diagnostic=1")
            }
        }

        func end() {
            guard let token else { return }
            self.token = nil
            endActivity(token)
            Self.logger.debug("end")
        }

        deinit { end() }
    #else
        init() {}
        func end() {}
    #endif
}
