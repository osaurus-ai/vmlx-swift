// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `NativeMTPActivationError` conformed only to `CustomStringConvertible`, so
// Foundation rendered it through the generic Error bridge:
//
//   "The operation couldn't be completed.
//    (MLXLMCommon.NativeMTPActivationError error 1.)"
//
// A case INDEX. The enum already carried an exact, actionable sentence for
// every case and none of it reached the user. `LocalizedError` is what makes
// `localizedDescription` use it.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Native MTP activation error text")
struct NativeMTPActivationErrorTextTests {

    @Test func everyCaseSurfacesItsOwnSentence() {
        let cases: [NativeMTPActivationError] = [
            .requestedButMissingArtifact(nil),
            .requestedWithoutUsableTuning(nil),
            .requestedForUnsupportedModel(["gemma4"]),
            .invalidConfigData,
        ]
        for error in cases {
            let localized = (error as Error).localizedDescription
            #expect(
                localized == error.description,
                "localizedDescription fell back to the generic bridge: \(localized)")
            #expect(
                !localized.contains("The operation couldn"),
                "still rendering the Foundation placeholder")
            #expect(!localized.contains("error 1"), "still rendering a case index")
        }
    }

    /// The case the user actually hit. Its text must name the tuning file, or
    /// it is no more diagnosable than the index was.
    @Test func theTuningCaseNamesTheFile() {
        let text = (NativeMTPActivationError.requestedWithoutUsableTuning(nil) as Error)
            .localizedDescription
        #expect(text.contains(NativeMTPTuning.fileName))
    }
}
