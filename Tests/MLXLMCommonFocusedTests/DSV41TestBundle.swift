// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Where the DeepSeek-V4.1 focused tests find a real bundle: `VMLX_DSV41_BUNDLE` (the bundle directory) and
// `VMLX_DSV41_CONFIG` (a config.json; defaults to the bundle's). There is no default path: tests that need the bundle
// are disabled without it, so they are reported as skipped instead of passing on no input. Under xcodebuild, pass
// them as `TEST_RUNNER_VMLX_DSV41_BUNDLE` / `TEST_RUNNER_VMLX_DSV41_CONFIG`.

import Foundation

enum DSV41TestBundle {
    private static func path(_ key: String) -> URL? {
        guard let raw = ProcessInfo.processInfo.environment[key], !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: raw)
    }

    /// The bundle directory, when set.
    static let directory: URL? = path("VMLX_DSV41_BUNDLE")

    /// A V4.1 config.json: `VMLX_DSV41_CONFIG`, else the bundle's, when the file exists.
    static let config: URL? = {
        guard
            let url = path("VMLX_DSV41_CONFIG") ?? directory?.appendingPathComponent("config.json"),
            FileManager.default.fileExists(atPath: url.path)
        else { return nil }
        return url
    }()

    /// A file or directory inside the bundle, when the bundle is set and has it.
    static func item(_ name: String) -> URL? {
        guard let url = directory?.appendingPathComponent(name),
            FileManager.default.fileExists(atPath: url.path)
        else { return nil }
        return url
    }
}
