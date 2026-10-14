# Full-attention draft prefill capture: one concatenation

Status: component proof only; actual Osaurus after-change text/media, cache and performance proof pending. Leave PR draft/open.

The full-attention DFlash/DSpark prefill path retained captured target hidden rows by concatenating all earlier rows on every chunk. Text and media paths now evaluate each new chunk, retain those arrays, and concatenate once at the end. Target prefill chunk boundaries, snapshots, progress, captured order/dtype and the final drafter projection shape remain unchanged. Sliding drafters retain the existing bounded trailing-window operation.

For the observed MiniCPM8M growing-chat suffix of27,219 rows and width10,240, nominal512-row chunks imply about15.6GB cumulative concat output at16bits. This is an operation-byte estimate, not measured hardware traffic. Exact full-prefix restore with one uncached row does not pay this growing-copy cost, so this change does not diagnose the warm decode slowdown.

## Component evidence

`DFlash2PrefillHiddenCaptureTests`:3tests pass. Uneven chunks, terminal one-row chunk, empty/single input, no limit and0/5/64-row limits preserve exact values and F32/F16/BF16 dtype. Timing probe is opt-in via `VMLX_PREFILL_CAPTURE_PROBE=1`.

Materialized, contiguous BF16 synthetic input:54chunks,27,219rows,10,240width; input active allocation557,449,216bytes. All three old/new results exactly equal. Timings old/new milliseconds:537.233/19.212(first),75.145/2.783,76.073/2.817. Other source builds were active; these are component observations, not matched-clock model throughput or app gains. Initial constant/broadcast fixtures were replaced before this result.

MLX allocation counters were also logged, but new incremental peak readings were smaller than the final tensor size. Do not interpret those counters as a validated app/RAM reduction; actual app physical footprint remains a required gate.

Command: Xcode developer directory, `swift test --build-system swiftbuild --enable-xctest --filter DFlash2PrefillHiddenCaptureTests`, with the existing built Metal library supplied by `VMLX_TEST_METALLIB_PATH` and the timing flag above. Full log kept privately as `hardening-128k/dspark-prefill-once-tests.log`.

No sampler/seed, cache precision, attention kernel, quota policy, draft-context restore or model unload change. This does not solve the missing historical DSpark companion state. Remaining: rebuilt isolated app with substantial uncached suffix; natural answer and held-out recall; repeat image path; SSD restore/ownership; physical footprint and TTFT. Do not merge on component timing alone.
