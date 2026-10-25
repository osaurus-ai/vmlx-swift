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
cache format or successful database behavior. Native qualification below covers the failed-index/retry lifecycle. It does
not qualify unrelated crash, model-quality or performance rows.


## Native Release integration on Max2

Tested app source `a07b0e5c82b9589f8f444920c427a84af9ae6a36`, runtime
`2f19fe7159cf07b1e66b5eb028ca5a8fa371c5ad`; Release executable SHA256
`0f35f36265bc39b1a75a52edd82b0aaa94cccca9c61ecc8137a0d6d3fae36efb`.
All manual work ran on `erics-m5-max2`; no remote machine qualification.
The proof artifact has Sentry DSN removed and disconnected stderr. The app
retains the normal generation contract, Safe Auto level2, Keep Model in
Memory on, no custom RAM fraction/cache cap, and default native dispatch.

An owned directory obstructed `cache_index.db`. The native v0.2 JANGHT1.8
bundle loaded and generated through the fallback. The directory marker was
preserved. After normal exit and moving only that owned obstruction, the
same chat reopened and a valid index appeared. The first blocked-index
code passed4generated tests and2000 independent oracle/input-preservation
cases,986tokens at64.028086tok/s.

Retained negative rows: its second answer has correct arithmetic but invalid
Python (4801tokens,59.3073tok/s). A correction request repeatedly reasoned
about brackets under recovered-index and fresh-cache Adaptive controls and
was cancelled; authoritative completion speed/count is missing for those
cancelled rows. Fresh-cache AR completed14332tokens at38.817357tok/s but
still produced invalid Python. Prior persisted cache and actual speculation
are not necessary. The remaining model/quantization/token-decoding/app cause
is unproven. These rows are failures, not repaired by this ownership change.

A separate native positive lifecycle control then passed:

| Bundle / phase | Tokens | Decode tok/s | Validation |
| --- | ---: | ---: | --- |
| v0.2 fresh stable_unique | 954 | 61.942666 | 4 generated tests +2000 independent cases |
| v0.2 growing list followup | 5575 | 33.487016 | all150 values, sum56625, valid Python |
| v0.2 after process restart | 282 | 44.877618 | correct new list and prior-turn facts |
| v0.1 after cancelled-load retry | 246 | 50.190069 | correct new list; input preservation explanation |
| v0.1 warm followup | 720 | 56.800875 | 4 generated tests +2000 independent cases |

All completed positive rows stop naturally with closed reasoning and an
unlocked composer. v0.2 uses bundle T=.45/top-p=.95/top-k20; v0.1 uses
T=1/.95/20. Samplers are unchanged; actual strategy is adaptive native MTP.
The33.49tok/s row has4026/5575tokens in AR fallback: **speed target fails**,
not a matched speedup. Unseeded outputs are not causal A/B evidence.

The real Server panel after restart reports6961/7093tokens restored
(`native-mtp`), Disk L2 hits/misses/stores3/3/3. Effective topology is fp16,
12KV +36Mamba layers, disk-backed restore, paged RAM off, TurboQuant0.
Separate SSM companion counters are0/3/0; do not label those as companion
hits or TurboQuant layers. Final SQLite integrity is `ok`,13indexed entries.

Activity Monitor captured31.27GB during the v0.2 growing turn and38.22GB
during the v0.1 warm prefill, with matching PIDs. Per-request sampled physical
peaks are33,828,445,992 /37,176,468,008 /33,775,984,256bytes for the three
v0.2 rows and42,895,685,344 /42,792,662,776bytes for v0.1. These measurements
do not establish a model-size-independent low-RAM path. Cancelling first
v0.1 load releases residency and leaves no inference activity; Activity
Monitor settles312.8MB. Normal retry succeeds. Both owned app processes quit
and collectors stop. No build or test overlapped these generation timings.

Private receipts/screenshots are under
`/Users/eric/vmlx-private-evidence/sentry-allosaurus-20261024/`:
`sqlite-index-obstructed/`, `sqlite-index-cold-control/`, and
`sqlite-index-ar-control/`. Key receipts are
`native-positive-validation.receipt.json`, `native-positive-resource.receipt.json`,
`native-v01-lifecycle-validation.receipt.json` and `native-final-index.receipt.json`
in the last root. Raw prompts/history/Sentry data and images are not committed.
