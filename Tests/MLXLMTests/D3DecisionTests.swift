import Foundation
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
