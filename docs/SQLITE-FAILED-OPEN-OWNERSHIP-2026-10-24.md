# Disk-cache failed-open handle ownership

`sqlite3_open` normally returns a connection handle even when opening fails.
The caller must close that handle: see the [SQLite API contract](https://www.sqlite.org/c3ref/open.html).
`DiskCache` discarded it by setting `db = nil` on the error path, so its
ordinary deinitializer could no longer release the connection resources.
Close the nonnil failed-open handle before clearing the pointer. Successful
connections retain their existing lifetime and locking.

Local reproduction uses the actual `DiskCache` initializer in an isolated
process. SQLite allocation accounting is enabled before its first connection;
the app and shared test runner are not reconfigured. A direct failed-open/
close control goes from 0 to 1,424 SQLite bytes and back to 0. On the original
runtime, 100 failed initializations retain 142,400 bytes for each of two
fixtures: a regular-file cache root and a directory at `cache_index.db`.
The repaired runtime retains zero bytes after all 200 failures. A valid
open/clear/close also returns to zero. No model or generation is involved.

The committed standalone probe is reproducible with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift run --package-path tools/SQLiteOpenFailureProbe SQLiteOpenFailureProbe
```

Two focused tests also pass: both obstructions and their marker contents
survive the fallback/clear path, and a later valid reopen passes SQLite's
integrity check. The standalone public probe compiles and passes locally.

This is a small cache error-path allocation leak, not a diagnosis of the
historical Sentry `walIndexTryHdr` crashes or Allosaurus's reported 11 tok/s.
It changes no memory-safety settings, limits, model math, generation defaults,
cache format or successful database behavior. Native app qualification of
the failed-index/retry scenario remains pending; keep the change draft.
