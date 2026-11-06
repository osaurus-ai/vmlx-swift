// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import XCTest

@testable import MLXLMCommon

/// Exercises production inspection against an explicit read-only header census.
/// It never constructs a model or reads tensor payloads.
final class SSDTableReadOnlyInventoryTests: XCTestCase {
    func testProductionFactsMatchExplicitHeaderInventory() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let specification = environment["VMLX_SSD_INVENTORY_SPEC"],
            let output = environment["VMLX_SSD_INVENTORY_OUTPUT"]
        else {
            throw XCTSkip("Set the explicit read-only SSD header specification and output paths")
        }
        let document = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: URL(fileURLWithPath: specification)))
                as? [String: Any])
        let bundles = try XCTUnwrap(document["bundles"] as? [[String: Any]])
        XCTAssertFalse(bundles.isEmpty, "an empty inventory is not qualification")
        var rows: [[String: Any]] = []
        for bundle in bundles {
            let path = try XCTUnwrap(bundle["path"] as? String)
            let total = try XCTUnwrap(bundle["expectedWeightBytes"] as? NSNumber).uint64Value
            let ssd = try XCTUnwrap(bundle["expectedSSDBytes"] as? NSNumber).uint64Value
            let facts = LoadBundleFacts.inspect(bundleURL: URL(fileURLWithPath: path))
            XCTAssertEqual(facts.totalSafetensorsBytes, total, path)
            XCTAssertEqual(facts.ssdResidentTableBytes, ssd, path)
            XCTAssertEqual(facts.gpuResidentWeightBytes, total - ssd, path)
            rows.append([
                "path": path, "totalWeightBytes": facts.totalSafetensorsBytes,
                "ssdTableBytes": facts.ssdResidentTableBytes,
                "residentWeightBytes": facts.gpuResidentWeightBytes,
            ])
        }
        try JSONSerialization.data(
            withJSONObject: ["rows": rows], options: [.prettyPrinted, .sortedKeys]
        )
        .write(to: URL(fileURLWithPath: output), options: .withoutOverwriting)
    }
}
