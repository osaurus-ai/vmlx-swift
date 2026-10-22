# Process-wide writer payload reservations

Stacks on #643, separate from its append-only KV lifetime change.

DiskCache previously checked only existing per-instance pending bytes, before
adding the incoming payload. Multiple models/cache instances could each retain
RAM/8, and every queue could overshoot by one complete payload.

Reserve the incoming logical payload bytes atomically against one process-wide
RAM/8 ceiling before admitting write-behind. A reservation follows InflightWrite
ownership until every producer/writer/screen reference is released. A full or
oversized reservation uses the established synchronous store path, without waiting
under CombinedDiskCacheQuotaLock or the MLX IO lock. Invalid, duplicate, refused
and synchronous stores release their producer reservation on return. Persistent
quota, entry validation, native dtypes, chain metadata and async publication rules
are unchanged. The number is writer-owned payload bytes, not total process RAM;
shared buffer/strided-owner capacity and live KV/allocator pool remain separate.

Tests: four reservation/integration cases pass, including two real DiskCache
instances with 8 KiB payloads under one 12 KiB cap: the second uses synchronous
publication, the first stays asynchronous, both entries are durable, and reserved
bytes returns to zero after writer closure release. Existing 20 write-behind tests
also pass (nonfinite/corrupt/duplicate/quota/publication/deadlock paths).

Not yet qualified in a newly pinned Osaurus app. Do not call this an all-model
memory/speed fix. #643's fresh app proof is a separate scope. No sampler, kernel,
model weights, adaptive speculation or configured RAM-safety ceiling changes.
