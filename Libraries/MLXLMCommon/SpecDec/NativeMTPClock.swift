import Dispatch
import Foundation

/// One monotonic time domain for iterator phases and governor decisions.
enum NativeMTPClock {
#if DEBUG
    @TaskLocal static var testingNow: TimeInterval?
#endif
    @inline(__always)
    static func now() -> TimeInterval {
#if DEBUG
        if let testingNow { return testingNow }
#endif
        return Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }
}
