import Foundation
import SQLite3
import XCTest

@testable import MLXLMCommon

final class DiskCacheOpenFailureTests: XCTestCase {
    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("disk-open-failure-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func assertHealthyIndex(_ directory: URL) throws {
        do {
            let cache = DiskCache(cacheDir: directory)
            XCTAssertEqual(cache.snapshotStats().currentEntryCount, 0)
            cache.clear()
        }
        var db: OpaquePointer?
        let path = directory.appendingPathComponent("cache_index.db").path
        XCTAssertEqual(sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nil), SQLITE_OK)
        defer { if let db { sqlite3_close(db) } }
        var statement: OpaquePointer?
        XCTAssertEqual(
            sqlite3_prepare_v2(db, "PRAGMA integrity_check", -1, &statement, nil), SQLITE_OK)
        defer { if let statement { sqlite3_finalize(statement) } }
        XCTAssertEqual(sqlite3_step(statement), SQLITE_ROW)
        let result = try XCTUnwrap(sqlite3_column_text(statement, 0))
        XCTAssertEqual(String(cString: result), "ok")
    }

    func testRegularFileRootIsPreservedAndLaterDirectoryRecovers() throws {
        let parent = try root()
        defer { try? FileManager.default.removeItem(at: parent) }
        let directory = parent.appendingPathComponent("cache")
        let payload = Data("an owned regular-file fixture".utf8)
        try payload.write(to: directory)
        do {
            let cache = DiskCache(cacheDir: directory)
            XCTAssertEqual(cache.snapshotStats().currentEntryCount, 0)
            cache.clear()
        }
        XCTAssertEqual(try Data(contentsOf: directory), payload)
        try FileManager.default.removeItem(at: directory)
        try assertHealthyIndex(directory)
    }

    func testDirectoryAtIndexPathIsPreservedAndLaterValidOpenRecovers() throws {
        let directory = try root()
        defer { try? FileManager.default.removeItem(at: directory) }
        let obstruction = directory.appendingPathComponent("cache_index.db")
        try FileManager.default.createDirectory(at: obstruction, withIntermediateDirectories: true)
        let marker = obstruction.appendingPathComponent("keep.txt")
        let payload = Data("keep this owned fixture".utf8)
        try payload.write(to: marker)
        do {
            let cache = DiskCache(cacheDir: directory)
            XCTAssertEqual(cache.snapshotStats().currentEntryCount, 0)
            cache.clear()
        }
        XCTAssertEqual(try Data(contentsOf: marker), payload)
        try FileManager.default.removeItem(at: obstruction)
        try assertHealthyIndex(directory)
    }
}
