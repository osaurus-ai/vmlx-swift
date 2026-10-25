// SPDX-License-Identifier: MIT
import Foundation
import MLXLMCommon
import SQLite3
import SQLiteProbeSupport

@main
struct DiskCacheOpenFailureProbe {
    static func emit(_ x: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: x, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
        fflush(stdout)
    }
    static func releaseScope(_ body: () -> Void) {
        #if canImport(ObjectiveC)
            autoreleasepool(invoking: body)
        #else
            body()
        #endif
    }
    static func main() throws {
        let config = vmlxProbeEnableSQLiteMemoryTracking()
        guard config == SQLITE_OK else {
            emit([
                "status": "blocked", "configure_rc": config,
                "reason": "SQLite initialized before isolated tracking setup",
            ])
            exit(2)
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "disk-open-probe-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fileRoot = root.appendingPathComponent("file-root")
        try Data("owned fixture".utf8).write(to: fileRoot)
        let directoryIndex = root.appendingPathComponent("directory-index")
        try FileManager.default.createDirectory(
            at: directoryIndex.appendingPathComponent("cache_index.db"),
            withIntermediateDirectories: true)
        // Warm SQLite's process-wide initialization, then show its real failed
        // handle allocation is tracked and is released by close.
        var handle: OpaquePointer?
        let initial = sqlite3_memory_used()
        let rc = sqlite3_open(fileRoot.appendingPathComponent("cache_index.db").path, &handle)
        let failed = sqlite3_memory_used()
        guard rc != SQLITE_OK, handle != nil, failed > initial else {
            if let handle { sqlite3_close(handle) }
            emit([
                "status": "blocked", "reason": "Failed-open allocation accounting unavailable",
                "open_rc": rc,
            ])
            exit(2)
        }
        precondition(sqlite3_close(handle) == SQLITE_OK)
        let closed = sqlite3_memory_used()
        emit([
            "phase": "control", "open_rc": rc, "before": initial,
            "failed_open_bytes": failed, "after_close_bytes": closed,
        ])
        var failures = 0
        for (name, path) in [("file-root", fileRoot), ("directory-index", directoryIndex)] {
            let before = sqlite3_memory_used()
            for _ in 0 ..< 100 {
                releaseScope {
                    let cache = DiskCache(cacheDir: path)
                    precondition(cache.snapshotStats().currentEntryCount == 0)
                    withExtendedLifetime(cache) {}
                }
            }
            let after = sqlite3_memory_used()
            let delta = after - before
            let pass = delta == 0
            if !pass { failures += 1 }
            emit([
                "phase": "actual_DiskCache", "fixture": name, "attempts": 100,
                "before_bytes": before, "after_bytes": after, "retained_sqlite_bytes": delta,
                "status": pass ? "pass" : "resource_leak",
            ])
        }
        let validBefore = sqlite3_memory_used()
        releaseScope {
            let cache = DiskCache(cacheDir: root.appendingPathComponent("valid"))
            cache.clear()
            precondition(cache.snapshotStats().currentEntryCount == 0)
        }
        let validAfter = sqlite3_memory_used()
        if validAfter != validBefore { failures += 1 }
        emit([
            "phase": "valid_open_clear_close", "before_bytes": validBefore,
            "after_bytes": validAfter,
            "status": validBefore == validAfter ? "pass" : "retained_bytes",
        ])
        emit([
            "phase": "end", "failures": failures, "generation": false,
            "scope": "isolated allocation ownership; not a Sentry WAL crash reproduction",
        ])
        if failures > 0 { exit(3) }
    }
}
