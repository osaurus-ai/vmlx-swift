import XCTest

@testable import MLXLMCommon

final class JANGHPrefillArchitectureTests: XCTestCase {
    func testBaseGPUKindsRetainTheTrailingG() {
        XCTAssertEqual(JANGHPrefillKernel.backend(forMetalArchitecture: "applegpu_g17g"), .nax)
        XCTAssertEqual(JANGHPrefillKernel.backend(forMetalArchitecture: "applegpu_g18g"), .nax)
    }

    func testProMaxUltraAndPhoneEligibilityIsUnchanged() {
        for name in ["applegpu_g17s", "applegpu_g17d", "applegpu_g18p"] {
            XCTAssertEqual(JANGHPrefillKernel.backend(forMetalArchitecture: name), .nax, name)
        }
    }

    func testOlderArchitecturesKeepSteel() {
        for name in [
            "applegpu_g14g", "applegpu_g14s", "applegpu_g15g", "applegpu_g15s", "applegpu_g15d",
            "applegpu_g16g", "applegpu_g16s", "applegpu_g16d", "applegpu_g17p",
        ] {
            XCTAssertEqual(JANGHPrefillKernel.backend(forMetalArchitecture: name), .steel, name)
        }
    }

    func testMalformedMetadataKeepsSteel() {
        for name in ["", "applegpu_g", "applegpu_g17", "applegpu_g17bad", "applegpu_gXXs"] {
            XCTAssertEqual(JANGHPrefillKernel.backend(forMetalArchitecture: name), .steel, name)
        }
    }

}
