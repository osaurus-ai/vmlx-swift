# Hybrid checkpoint memory lifecycle

Canonical hybrid prompts replace their generation suffix on the next turn. The captured suffix-stripped checkpoint is reusable; the exact prompt is not. Non-trimmable recurrent caches cannot produce older boundaries by trimming that exact prompt snapshot either.

The AR and native-MTP iterators therefore keep the captured boundary instead of retaining a redundant exact prompt copy. For disk-only coordinators they publish that owned checkpoint after prefill, and release their references after the coordinator accepts the complete boundary. The asynchronous writer owns its arrays until publication finishes. A rejected store leaves the checkpoint available for the terminal store. Paged caching keeps its existing finalization route.

Terminal stores reuse already-owned snapshots. Live post-answer state is still copied before serialization so subsequent cache mutations cannot change the writer's payload. A non-trimmable cache is checked before attempting a boundary copy. Dtypes, cache salts, chain ownership, quotas, sampler settings and speculative controller behavior retain their existing contracts.

## Verification — 2026-10-20

Runtime implementation `a1e673426`, focused test correction `c1e52b316`, Osaurus `11044899` pinned to that runtime. The unsigned Release app binary SHA256 was `78147654962c30806965db756ca3d58e84f531339aff6295f1038280fde3d4ea`.

- 42 XCTest cases and four Swift Testing cases passed. These cover exact KV/recurrent state after reopening disk, delayed writer ownership, failed synchronous publication and retry, disabled disk/paged/offset mismatch, native/AR boundary capture, stable-seed logits and media salt isolation.
- Real Osaurus Helper chat with Ornith 1.5 122B JANG_4M grew through 43k, 62k, 80k, 98k, 116k and 128,301 tokens. All six ledger answers matched the selected record and depot; every turn restored its preceding SSD checkpoint.
- At 128,301 tokens, MLX active allocation was 80.34 GB after prefill and 80.34 GB at the prompt-snapshot stage (no retained exact copy), 78.58 GB at decode end, 81.88 GB after final store, 78.55 GB once the writer settled. Process physical footprint peaked 12.48 GB. Writer-owned transient buffers still exist.
- A pasted image at 129,083 tokens restored 128,413 tokens and processed 670 new tokens, with 6.656 s runtime TTFT. The visible answer correctly identified ORCHID 47 and a blue square. The accepted stop-token row was saved at 129144 and subsequently restored by AR.
- Off was selected in the actual model picker. At 130k, the AR calculator turn returned 98,926; its follow-up restored the newly spilled 130158-token checkpoint and processed 124 new tokens (1.808 s TTFT). Three distinct~3.35 GB writes took 361/393/319 ms; final synchronous store return 72.6 ms. Peak process footprint 14.93 GB.
- A fresh text-only native-MTP chat exercised early spill itself, then calculated 245 correctly. Its tool follow-up restored 13,523 tokens and processed 116 new tokens (0.895 s TTFT). ~0.49 GB writes took 58/58/43 ms; footprint peaked 6.36 GB during the tool turn.

Runtime decode telemetry is retained in the evidence: native ledger 28.9→15.9 tok/s as context grew 43k→128k; image 9.1 tok/s; AR tool 23.9/47.7 tok/s; text-only native tool 48.8/59.7 tok/s. These are different short outputs and clocks, **not a matched speedup comparison**. Sampling, governor and kernels were unchanged. No claim that native MTP is always faster.

Private raw logs, source identity, screenshot receipts, process memory samples, bounded GPU captures and remaining-path notes are in `cache-ram-ssd-20261020/takeover/` and `ornith122b-claude-20261019/turns/RAM1-*` / `RAM2-*` under the private evidence root. The inherited GPU parser mixed identical times of day across several days; the new harness records only the log bytes appended during each run.

Scope limits:
- DFlash2/DSpark and BatchEngine have separate snapshot ownership paths. This change does not claim to eliminate their retained checkpoints.
- Transient writer-owned buffers are still required. Early enqueue does not mean the bytes are already durable or no longer in RAM.
- This is not a change to long-context attention kernels or adaptive speculation policy.

- A pre-existing native-MTP limitation remains for tool steps in chats with historical media: boundary capture excludes media history, so the tool follow-up may reuse the prior conversation checkpoint rather than the immediately preceding tool step. Text-only native and media-history AR per-step reuse were observed above.
- JANGH2 and other model families have not received a new 128k qualification in this change. Issue #624 remains open for those paths and the separate schedulers.
