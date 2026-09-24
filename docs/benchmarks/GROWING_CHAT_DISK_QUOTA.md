# Growing-chat SSD quota diagnostic

`BENCH_GROWING_CHAT_CACHE=1` accepts `BENCH_GROWING_DISK_MAX_GB` in GiB.
Omitting it retains the 4 GiB default. Invalid, non-finite, sub-byte and
unrepresentable values fail before loading the model or creating the cache root.
The requested quota is printed; the existing coordinator statistics report the
effective quota, stores, skips and evictions.

Use a task-owned `BENCH_GROWING_CACHE_DIR`, preserve it with
`BENCH_KEEP_GROWING_CACHE=1` when inspecting the index, and keep the existing host
memory guard. For bundle-native sampling use `BENCH_GROWING_BUNDLE_DEFAULTS=1`
and the model's native thinking setting. Do not interpret a length-cap stop as
coherency success.

This input changes only the benchmark coordinator. It does not change production
cache settings or relax the regression's restore/coherency assertions. A quota
that cannot retain a valid boundary can legitimately make the restore assertion
fail: inspect store skips and eviction records before calling that a cache bug.
Compare performance only on identical workloads without competing compiler load.
