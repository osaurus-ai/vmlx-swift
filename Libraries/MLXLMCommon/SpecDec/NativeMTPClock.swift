import Dispatch
import Foundation

/// One monotonic time domain for iterator phases and governor decisions.
enum NativeMTPClock {
    @inline(__always)
    static func now() -> TimeInterval {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
    }
}
