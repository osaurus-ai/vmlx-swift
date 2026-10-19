import Foundation
import MLX
import XCTest
@preconcurrency import VMLXTokenizers
@testable import MLXHuggingFace
@testable import MLXLMCommon
@testable import MLXVLM

/// Opt-in loader coverage with the converter's small random-weight rehearsal.
/// This checks execution and state shape, never model quality or app readiness.
final class Ornith122BJANGHLoadTests: XCTestCase {
    func testMixedRoutedBundleLoadsWithAndWithoutAffineMTP() async throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_ORNITH_REHEARSAL"] else {
            throw XCTSkip("Set VMLX_ORNITH_REHEARSAL to the tiny mixed JANGH/affine bundle")
        }
        for enabled in [false, true] {
            try await NativeMTPActivation.$manualDepthOverride.withValue(enabled ? 1 : nil) {
                try await NativeMTPActivation.withExplicitRequest(enabled) {
                    let (context, _) = try await MLXLMCommon.loadModel(
                        from: URL(fileURLWithPath: path), using: #huggingFaceTokenizerLoader(),
                        loadConfiguration: LoadConfiguration.default)
                    let model = try XCTUnwrap(context.model as? Qwen35MoE)
                    XCTAssertFalse(model.customDenseTensorNames.isEmpty)
                    XCTAssertFalse(model.customDenseTensorNames.contains { $0.contains("mtp.") })
                    let cache = model.newCache(parameters: nil)
                    let logits = model(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: cache)
                    eval(logits)
                    XCTAssertEqual(logits.shape, [1, 3, 248320])
                    XCTAssertTrue(logits.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
                    let next = model(MLXArray([Int32(4)]).reshaped(1, 1), cache: cache)
                    eval(next)
                    XCTAssertEqual(next.shape, [1, 1, 248320])
                    if enabled {
                        let draft = model.nativeMTPForward(
                            hiddenStates: MLXArray.zeros([1, 1, 128]),
                            nextTokenIds: MLXArray([Int32(4)]).reshaped(1, 1),
                            cache: model.makeNativeMTPCache())
                        eval(draft.logits)
                        XCTAssertEqual(draft.logits.shape, [1, 1, 248320])
                        XCTAssertTrue(draft.logits.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
                    }
                }
            }
        }
    }
}
