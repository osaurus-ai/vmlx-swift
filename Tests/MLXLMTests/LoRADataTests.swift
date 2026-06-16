// Copyright © 2024 Apple Inc.

@testable import MLXLLM
import XCTest

final class LoRADataTests: XCTestCase {
    func testUnsupportedLoRADataFileTypeThrows() throws {
        let url = URL(fileURLWithPath: "/tmp/train.csv")

        do {
            _ = try loadLoRAData(url: url)
            XCTFail("Expected unsupported LoRA data file type to throw")
        } catch let error as LoRADataError {
            guard case .unsupportedFileType(let thrownURL) = error else {
                XCTFail("Unexpected LoRADataError: \(error)")
                return
            }
            XCTAssertEqual(thrownURL, url)
        }
    }
}
