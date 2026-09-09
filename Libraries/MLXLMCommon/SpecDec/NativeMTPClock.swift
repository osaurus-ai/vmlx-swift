// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Dispatch
import Foundation

/// Process-local elapsed-time clock, never a calendar timestamp. All iterator
/// phase durations and governor windows must use the same time domain so a
/// wall-clock correction cannot suppress a loss or fabricate a slow cycle.
enum NativeMTPClock {
    @inline(__always)
    static func now() -> TimeInterval {
        seconds(uptimeNanoseconds: DispatchTime.now().uptimeNanoseconds)
    }

    static func seconds(uptimeNanoseconds: UInt64) -> TimeInterval {
        Double(uptimeNanoseconds) / 1_000_000_000
    }
}
