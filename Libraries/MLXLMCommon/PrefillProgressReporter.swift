// Copyright © 2026 osaurus.

import Foundation

/// Scoped callback used by model `prepare` implementations to report
/// completed prompt-processing units without changing the `LanguageModel`
/// protocol signature.
public enum PrefillProgressReporter {
    public typealias Handler = @Sendable (Int) -> Void

    private final class HandlerBox {
        let handler: Handler

        init(_ handler: @escaping Handler) {
            self.handler = handler
        }
    }

    private static let threadDictionaryKey = "ai.osaurus.vmlx.prefillProgressReporter"

    public static func withHandler<T>(
        _ handler: Handler?,
        operation: () throws -> T
    ) rethrows -> T {
        guard let handler else { return try operation() }

        let dictionary = Thread.current.threadDictionary
        let previous = dictionary[threadDictionaryKey]
        dictionary[threadDictionaryKey] = HandlerBox(handler)
        defer {
            if let previous {
                dictionary[threadDictionaryKey] = previous
            } else {
                dictionary.removeObject(forKey: threadDictionaryKey)
            }
        }
        return try operation()
    }

    public static func reportCompletedUnits(_ count: Int) {
        guard let box = Thread.current.threadDictionary[threadDictionaryKey] as? HandlerBox else {
            return
        }
        box.handler(max(0, count))
    }
}
