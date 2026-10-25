# Disk-cache failed-open ownership probe

Run in its own process on macOS:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift run --package-path tools/SQLiteOpenFailureProbe SQLiteOpenFailureProbe
```

The C shim enables SQLite allocation accounting before the first connection.
It does not shut down or reconfigure SQLite in the app or a shared test runner.
The probe refuses to run if SQLite is already initialized. It uses owned
temporary paths only, opens no user database and loads no model.

A direct failed-open/close control must return to its original allocation
count. Then 100 actual `DiskCache` constructions are released for each of two
failure fixtures: a regular-file cache root and a directory at the index path.
Both must retain zero SQLite bytes. A valid open/clear/close must still work.
The original implementation retains 142,400 SQLite bytes per 100 failures in
each fixture on the measured macOS build; it returns exit 3. This probes an
allocation leak, not the historical Sentry WAL crash.
