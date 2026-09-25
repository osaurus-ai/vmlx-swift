// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

// swift-corelibs-foundation has no `String(localized:)`. Linux has no localization catalog for
// these messages anyway, so they read as written. Used from MLXLMCommon and MLXLLM.

#if os(Linux)
    import Foundation

    extension String {
        package init(localized value: String, comment: StaticString? = nil) {
            self = value
        }
    }
#endif
