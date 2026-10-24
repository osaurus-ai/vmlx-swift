import Cmlx
import CoreImage
import Foundation
import MLX
import XCTest
@testable import MLXLMCommon
@testable import MLXVLM

final class D3DecisionTests: XCTestCase {
    func testJSONPreservesOrderAndFalsyState() throws {
        let value = try DecisionJSON.parse(Data(#"{"z":0,"a":false,"unicode":"हिन्दी","nested":{"b":2,"a":1}}"#.utf8))
        XCTAssertEqual(value.entries?.map(\.0), ["z", "a", "unicode", "nested"])
        XCTAssertEqual(value.json, #"{"z": 0, "a": false, "unicode": "हिन्दी", "nested": {"b": 2, "a": 1}}"#)
        XCTAssertEqual(D3Prompt.stateText(value["z"]!), "0")
        XCTAssertEqual(D3Prompt.stateText(value["a"]!), "false")
        XCTAssertEqual(D3Prompt.stateText(.object([])), "{}")
        XCTAssertEqual(D3Prompt.stateText(.array([])), "[]")
        XCTAssertEqual(D3Prompt.stateText(.null), "(empty)")
        XCTAssertThrowsError(try DecisionJSON.parse(Data(#"{"a":1,"a":2}"#.utf8)))
    }

    func testPromptAndPublicAnswers() throws {
        let q = try D3Question(DecisionJSON.parse(Data(#"{"type":"choice","criteria":{"z":null,"a":"First"}}"#.utf8)))
        XCTAssertEqual(q.keys, ["z", "a"])
        XCTAssertEqual(D3Prompt.user(state: .bool(false), question: q, codes: ["A", "B"]),
            "State:\nfalse\n\nQuestion:\nChoose the best matching option.\n\nOptions:\nA: z\nB: a: First\n\nReply with only the code of the best option.")
        let n = try D3Question(DecisionJSON.parse(Data(#"{"type":"noul"}"#.utf8)))
        XCTAssertEqual(try n.answer(probabilities: [0.75, 0.25]), .object(["type": .string("noul"), "noul": .double(0.25)]))
        let score = try D3Question(DecisionJSON.parse(Data(#"{"type":"score","criteria":["bad","good"]}"#.utf8)))
        XCTAssertEqual(score.optionTexts, ["bad", "good"])
        XCTAssertThrowsError(try D3Question(DecisionJSON.parse(Data(#"{"type":"score","criteria":["same","same"]}"#.utf8))))
    }

    func testNFCStateBoundaryMatchesTokenBytes() throws {
        let state = DecisionJSON.string("e\u{301}")
        let text = "prefix State:\né\n\nQuestion:\nChoose."
        let ids = text.utf8.map(Int.init)
        XCTAssertEqual(try D3Prompt.stateTokenCount(ids: ids, state: state, tokenizer: D3ByteTokenizer()),
            "prefix State:\né".utf8.count)
    }

    func testDecisionImageUsesNCHWPatchLayout() throws {
        if let path = ProcessInfo.processInfo.environment["MLX_TEST_METALLIB_PATH"] {
            XCTAssertEqual(path.withCString { mlx_metal_set_metallib_path($0) }, 0)
        }
        let config = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: Data(#"{"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"merge_size":2,"patch_size":16,"temporal_patch_size":2,"image_processor_type":"Qwen3VLImageProcessor","size":{"min_pixels":2048,"max_pixels":2048}}"#.utf8))
        let processor = Qwen3VLProcessor(config, tokenizer: D3ByteTokenizer(), decisionMediaProcessing: true)
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 32))
        let (pixels, grid) = try processor.preprocess(images: [image], processing: nil)
        XCTAssertEqual([grid.t, grid.h, grid.w], [1, 2, 4])
        XCTAssertEqual(pixels.shape, [8, 1536])
        let first = pixels[0].asArray(Float.self)
        XCTAssertTrue(first[0..<512].allSatisfy { abs($0 - 1) < 1e-5 })
        XCTAssertTrue(first[512..<1536].allSatisfy { abs($0 + 1) < 1e-5 })
    }

    func testTimestampSeparatedVideoPositionGroups() throws {
        // Text, one 2x2 merged video grid, timestamp text, next 2x2 grid, final text.
        let positions = try D3MRoPE.positions(tokens: [7,9,9,9,9,8,9,9,9,9,6], images: [],
            videos: [THW(2,4,4)], merge: 2, imageToken: 10, videoToken: 9)
        XCTAssertEqual(positions[0], [0,1,1,1,1,3,4,4,4,4,6])
        XCTAssertEqual(positions[1], [0,1,1,2,2,3,4,4,5,5,6])
        XCTAssertEqual(positions[2], [0,1,2,1,2,3,4,5,4,5,6])
        XCTAssertThrowsError(try D3MRoPE.positions(tokens: [9], images: [], videos: [THW(1,4,4)],
            merge: 2, imageToken: 10, videoToken: 9))
    }

    func testMixedMediaAndTextPositions() throws {
        let positions = try D3MRoPE.positions(tokens: [7,10,10,8,9,9,6], images: [THW(1,2,4)],
            videos: [THW(1,4,2)], merge: 2, imageToken: 10, videoToken: 9)
        XCTAssertEqual(positions[0], [0,1,1,3,4,4,6])
        XCTAssertEqual(positions[1], [0,1,1,3,4,5,6])
        XCTAssertEqual(positions[2], [0,1,2,3,4,4,6])
    }
}

private struct D3ByteTokenizer: Tokenizer {
    var incrementalByteLevelDecoder: (@Sendable (Int) -> ByteLevelDecodingPiece)? {
        { .bytes([UInt8($0)]) }
    }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { String(decoding: tokenIds.map(UInt8.init), as: UTF8.self) }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
}
