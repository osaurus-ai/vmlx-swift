import Foundation
@testable import MLXLMCommon
import Testing

@Suite struct StreamTimingRecorderTests {
    @Test func disabledWithoutAbsoluteDirectory() {
        #expect(StreamTimingRecorder(environment: [:]) == nil)
        #expect(StreamTimingRecorder(environment: ["VMLX_STREAM_TIMING_DIR": "relative"]) == nil)
    }

    @Test func terminalCauseIsPreservedInReceipt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("stream-cause-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = try #require(StreamTimingRecorder(environment: ["VMLX_STREAM_TIMING_DIR": root.path]))
        recorder.record()
        recorder.recordTermination("stop_token:151645")
        recorder.finish(tokenCount: 1, stopReason: "stop")
        let files = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        #expect(files.count == 1)
        let data = try Data(contentsOf: #require(files.first))
        let payload = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(payload["termination_cause"] as? String == "stop_token:151645")
        #expect(payload["generation_token_count"] as? Int == 1)
        #expect(payload["stop_reason"] as? String == "stop")
        #expect(payload["token_arrival_elapsed_ns"] is [NSNumber])
    }
}
