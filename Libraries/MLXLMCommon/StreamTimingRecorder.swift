import Foundation

/// Opt-in proof instrumentation. Measures iterator delivery, not GPU completion
/// or UI rendering. No token text, synchronization, or per-token I/O.
/// A terminating special-token ID may be recorded to distinguish stop causes.
final class StreamTimingRecorder {
    private let directory: URL
    private let start = DispatchTime.now().uptimeNanoseconds
    private var arrivals: [UInt64] = []
    private var truncated = false
    private var terminationCause = "iterator_exhausted"

    func recordTermination(_ cause: String) {
        terminationCause = cause
    }

    init?(environment: [String: String] = ProcessInfo.processInfo.environment) {
        guard let path = environment["VMLX_STREAM_TIMING_DIR"], path.hasPrefix("/") else {
            return nil
        }
        directory = URL(fileURLWithPath: path, isDirectory: true)
        arrivals.reserveCapacity(8192)
    }

    func record() {
        guard arrivals.count < 1_000_000 else { truncated = true; return }
        arrivals.append(DispatchTime.now().uptimeNanoseconds - start)
    }

    func finish(tokenCount: Int, stopReason: String) {
        let end = DispatchTime.now().uptimeNanoseconds - start
        let payload: [String: Any] = [
            "schema": 1, "clock": "monotonic_uptime_ns",
            "measurement": "iterator_delivery_not_gpu_or_ui",
            "start_uptime_ns": start, "end_elapsed_ns": end,
            "token_arrival_elapsed_ns": arrivals,
            "generation_token_count": tokenCount, "stop_reason": stopReason,
            "termination_cause": terminationCause,
            "truncated": truncated,
        ]
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            let url = directory.appendingPathComponent("stream-\(UUID().uuidString).json")
            try data.write(to: url, options: .atomic)
            FileHandle.standardError.write(Data("[StreamTiming] path=\(url.path) tokens=\(arrivals.count) truncated=\(truncated)\n".utf8))
        } catch {
            FileHandle.standardError.write(Data("[StreamTiming] FAILED: \(error)\n".utf8))
        }
    }
}
