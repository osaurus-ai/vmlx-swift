import Foundation
import SQLite3

/// Schema versioning for the disk cache's SQLite index (`cache_index.db`).
///
/// The index is shared. Every loaded model opens its own connection to the
/// same file, and an older build may read and write the same directory after
/// a downgrade or from a second install. Two rules follow:
///
/// - A migration only ever ADDS: every new column is nullable or carries a
///   DEFAULT, and nothing is renamed or dropped, so the v1 statements
///   (`INSERT OR REPLACE … (hash, token_count, file_size)`, the v1 SELECTs and
///   DELETEs) keep working against a v2 index.
/// - A migration that cannot run is not an error for the cache. The index
///   stays a working v1 index and the caller carries on; users of the v2
///   columns check `hasV2Columns` first.
enum DiskCacheIndexSchema {
    static let currentVersion: Int32 = 2

    /// How long a migration waits for another connection's write lock.
    static let defaultBusyTimeoutMs: Int32 = 5000

    /// Entry kinds. `history` rows belong to one conversation; `stableRoot`
    /// rows (system prompt + tools) are shared across conversations.
    enum Kind: Int32 { case history = 0, stableRoot = 1 }

    /// The v1 index, exactly as every earlier build created it. `migrate`
    /// runs these under the write lock as well: `DiskCache.init` issues them
    /// with no busy timeout and ignores the result, so when several
    /// connections open a fresh directory at once every one of those CREATEs
    /// can lose to another connection's lock, leaving no table at all.
    static let v1Statements: [String] = [
        """
        CREATE TABLE IF NOT EXISTS cache_entries (
            hash TEXT PRIMARY KEY,
            token_count INTEGER,
            file_size INTEGER,
            created_at REAL DEFAULT (julianday('now'))
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cache_entries_token_count
        ON cache_entries(token_count DESC)
        """,
    ]

    static let v2Statements: [String] = [
        "ALTER TABLE cache_entries ADD COLUMN model_key TEXT",
        "ALTER TABLE cache_entries ADD COLUMN kind INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE cache_entries ADD COLUMN chain_id TEXT",
        "ALTER TABLE cache_entries ADD COLUMN companion_key TEXT",
        "ALTER TABLE cache_entries ADD COLUMN companion_bytes INTEGER NOT NULL DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_cache_entries_chain ON cache_entries(chain_id)",
        "CREATE TABLE IF NOT EXISTS legacy_companions (key TEXT PRIMARY KEY, bytes INTEGER NOT NULL, modified REAL NOT NULL)",
        modelTokensIndexStatement,
    ]

    /// What `DiskCache.candidateTokenCounts` reads: one model's lengths,
    /// longest first, without touching another model's rows or the table.
    /// Additive like everything else here — an older build neither sees nor
    /// minds an index, and SQLite keeps it current under that build's writes.
    static let modelTokensIndexName = "idx_cache_entries_model_tokens"
    static let modelTokensIndexStatement =
        "CREATE INDEX IF NOT EXISTS \(modelTokensIndexName) ON cache_entries(model_key, token_count)"

    private static let v2ColumnNames: [String] = [
        "model_key", "kind", "chain_id", "companion_key", "companion_bytes",
    ]

    /// Brings the index to `currentVersion`. Idempotent and safe when several
    /// connections race: runs inside BEGIN IMMEDIATE and re-reads user_version
    /// after the write lock is held. Returns the version now in force.
    ///
    /// Never throws and never deletes. On any failure the transaction, if one
    /// was opened, is rolled back, which leaves a working v1 index, and the
    /// return value is `user_version` as read afterwards. When that read
    /// fails too, the version seen under the lock is returned, or 0 when
    /// there is none: the lock was never taken, or `user_version` could not
    /// be read under it (that exit returns 0 without reading again). A
    /// returned 0 therefore means "use this index as v1", not necessarily
    /// "the file says 0". A `user_version` above `currentVersion` belongs to
    /// a newer build and is left exactly as found.
    @discardableResult
    static func migrate(
        _ db: OpaquePointer?, busyTimeoutMs: Int32 = defaultBusyTimeoutMs
    ) -> Int32 {
        guard let db else { return 0 }
        return withBusyTimeout(db, busyTimeoutMs) { migrateWaiting(db) }
    }

    private static func migrateWaiting(_ db: OpaquePointer) -> Int32 {
        // Cheap exit without taking the write lock: the common case is an
        // index that is already current.
        if let version = userVersion(db), version >= currentVersion {
            return version
        }

        guard exec(db, "BEGIN IMMEDIATE") else {
            warn("could not take the write lock: \(lastError(db)); index stays as it is")
            return userVersion(db) ?? 0
        }

        // The write lock is held: whatever another connection did is visible.
        guard let version = userVersion(db) else {
            warn("could not read user_version: \(lastError(db))")
            exec(db, "ROLLBACK")
            return 0
        }
        if version >= currentVersion {
            // Nothing was written. ROLLBACK cannot fail; a COMMIT that failed
            // would leave this connection holding the write lock.
            exec(db, "ROLLBACK")
            return version
        }

        for statement in v1Statements + v2Statements {
            // A racing connection or a crash between statements may already
            // have added the column; ADD COLUMN has no IF NOT EXISTS.
            if let column = addedColumnName(statement), columnExists(db, column) {
                continue
            }
            guard exec(db, statement) else {
                warn("\(statement) failed: \(lastError(db)); index stays v\(version)")
                exec(db, "ROLLBACK")
                return userVersion(db) ?? version
            }
        }

        guard exec(db, "PRAGMA user_version = \(currentVersion)"), exec(db, "COMMIT") else {
            warn("could not commit: \(lastError(db)); index stays v\(version)")
            exec(db, "ROLLBACK")
            return userVersion(db) ?? version
        }
        return currentVersion
    }

    /// Whether every v2 column and `legacy_companions` exist on this index.
    /// Independent of `user_version`, so it stays truthful for an index that a
    /// newer build has taken further.
    ///
    /// Waits like `migrate` does, so that contention is not mistaken for
    /// absence: on a database that is not in WAL mode yet (a fresh directory
    /// several connections are opening at once) a plain read can come back
    /// SQLITE_BUSY. An index that still cannot be read once the timeout has
    /// run out reports `false`, the conservative answer: the caller then
    /// uses the v1 statements only.
    static func hasV2Columns(
        _ db: OpaquePointer?, busyTimeoutMs: Int32 = defaultBusyTimeoutMs
    ) -> Bool {
        guard let db else { return false }
        return withBusyTimeout(db, busyTimeoutMs) {
            for column in v2ColumnNames where !columnExists(db, column) {
                return false
            }
            return scalarInt(
                db,
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='legacy_companions'"
            ) == 1
        }
    }

    /// Adds the indexes a v2 index may have been created without: `migrate`
    /// leaves an index that is already at `currentVersion` alone, and one
    /// that reached it before an index was introduced does not have it.
    /// Only at exactly `currentVersion` — a newer build's schema is left as
    /// found — and only with the v2 columns really there.
    ///
    /// A read when there is nothing to add. A failure is not an error for
    /// the cache: the statements that would use the index work without it,
    /// slower, and the next open tries again.
    static func ensureV2Indexes(
        _ db: OpaquePointer?, version: Int32, hasV2Columns: Bool,
        busyTimeoutMs: Int32 = defaultBusyTimeoutMs
    ) {
        guard let db, version == currentVersion, hasV2Columns else { return }
        withBusyTimeout(db, busyTimeoutMs) {
            let present = scalarInt(
                db,
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='\(modelTokensIndexName)'"
            )
            guard present == 0 else { return }
            if !exec(db, modelTokensIndexStatement) {
                warn("\(modelTokensIndexStatement) failed: \(lastError(db)); carrying on without it")
            }
        }
    }

    /// Runs `body` with a busy timeout when the connection has none, so it
    /// waits for another connection's lock instead of failing on it. The
    /// previous value is put back afterwards: how the cache's ordinary
    /// statements behave under contention is not this type's call.
    private static func withBusyTimeout<T>(
        _ db: OpaquePointer, _ timeoutMs: Int32, _ body: () -> T
    ) -> T {
        let previous =
            scalarInt(db, "PRAGMA busy_timeout").map(Int32.init(truncatingIfNeeded:)) ?? 0
        guard previous <= 0 else { return body() }
        sqlite3_busy_timeout(db, timeoutMs)
        defer { sqlite3_busy_timeout(db, 0) }
        return body()
    }

    // MARK: - SQLite helpers

    private static func userVersion(_ db: OpaquePointer) -> Int32? {
        scalarInt(db, "PRAGMA user_version").map(Int32.init(truncatingIfNeeded:))
    }

    /// `ALTER TABLE cache_entries ADD COLUMN <name> …` → `<name>`.
    private static func addedColumnName(_ statement: String) -> String? {
        let prefix = "ALTER TABLE cache_entries ADD COLUMN "
        guard statement.hasPrefix(prefix) else { return nil }
        return statement.dropFirst(prefix.count).split(separator: " ").first.map(String.init)
    }

    private static func columnExists(_ db: OpaquePointer, _ column: String) -> Bool {
        var stmt: OpaquePointer?
        guard
            sqlite3_prepare_v2(
                db, "SELECT COUNT(*) FROM pragma_table_info('cache_entries') WHERE name = ?",
                -1, &stmt, nil) == SQLITE_OK
        else { return false }
        defer { sqlite3_finalize(stmt) }
        return column.withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
            return sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_int64(stmt, 0) > 0
        }
    }

    private static func scalarInt(_ db: OpaquePointer, _ sql: String) -> Int64? {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        return sqlite3_column_int64(stmt, 0)
    }

    @discardableResult
    private static func exec(_ db: OpaquePointer, _ sql: String) -> Bool {
        sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK
    }

    private static func lastError(_ db: OpaquePointer) -> String {
        String(cString: sqlite3_errmsg(db))
    }

    private static func warn(_ message: String) {
        FileHandle.standardError.write(
            Data("[vmlx][cache/disk-index] schema migration: \(message)\n".utf8))
    }
}
