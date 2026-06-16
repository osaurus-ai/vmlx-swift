// Copyright © 2026 Jinho Jang. All rights reserved.

/// Task-scoped hooks for callers that need to mutate a freshly-instantiated
/// model before weights are loaded.
public enum ModelLoadHooks {
    @TaskLocal public static var preLoadModelMutation:
        (@Sendable (any LanguageModel) throws -> Void)?
}
