# Hybrid checkpoint memory lifecycle

Canonical hybrid prompts replace their generation suffix on the next turn. The captured suffix-stripped checkpoint is reusable; the exact prompt is not. Non-trimmable recurrent caches cannot produce older boundaries by trimming that exact prompt snapshot either.

The AR and native-MTP iterators therefore keep the captured boundary instead of retaining a redundant exact prompt copy. For disk-only coordinators they publish that owned checkpoint after prefill, and release their references after the coordinator accepts the complete boundary. The asynchronous writer owns its arrays until publication finishes. A rejected store leaves the checkpoint available for the terminal store. Paged caching keeps its existing finalization route.

Terminal stores reuse already-owned snapshots. Live post-answer state is still copied before serialization so subsequent cache mutations cannot change the writer's payload. A non-trimmable cache is checked before attempting a boundary copy. Dtypes, cache salts, chain ownership, quotas, sampler settings and speculative controller behavior retain their existing contracts.

## Verification — 2026-10-20

Runtime implementation `a1e673426`, focused test correction `c1e52b316`, Osaurus `11044899` pinned to that runtime. The unsigned Release app binary SHA256 was `78147654962c30806965db756ca3d58e84f531339aff6295f1038280fde3d4ea`.

- 42 XCTest cases and four Swift Testing cases passed. These cover exact KV/recurrent state after reopening disk, delayed writer ownership, failed synchronous publication and retry, disabled disk/paged/offset mismatch, native/AR boundary capture, stable-seed logits and media salt isolation.
- Real Osaurus Helper chat with Ornith 1.5 122B JANG_4M grew through 43k, 62k, 80k, 98k, 116k and 128,301 tokens. All six ledger answers matched the selected record and depot; every turn restored its preceding SSD checkpoint.
- At128,301 tokens, MLX active allocation was80.34GB after prefill and80.34GB at the prompt-snapshot stage (no retained exact copy),78.58GB at decode end,81.88GB after final store,78.55GB once the writer settled. Process physical footprint peaked12.48GB. Writer-owned transient buffers still exist.
- A pasted image at129,083 tokens restored128,413 tokens and processed670 new tokens, with6.656s runtime TTFT. The visible answer correctly identified ORCHID47 and a blue square. The accepted stop-token row was saved at129144 and subsequently restored by AR.
- Off was selected in the actual model picker. At130k, the AR calculator turn returned98,926; its follow-up restored the newly spilled130158-token checkpoint and processed124 new tokens (1.808s TTFT). Three distinct~3.35GB writes took361/393/319ms; final synchronous store return72.6ms. Peak process footprint14.93GB.
- A fresh text-only native-MTP chat exercised early spill itself, then calculated245 correctly. Its tool follow-up restored13523 tokens and processed116 new tokens (0.895s TTFT).~0.49GB writes took58/58/43ms; footprint peaked6.36GB during the tool turn.

Runtime decode telemetry is retained in the evidence: native ledger28.9→15.9tok/s as context grew43k→128k; image9.1tok/s; AR tool23.9/47.7tok/s; text-only native tool48.8/59.7tok/s. These are different short outputs and clocks, **not a matched speedup comparison**. Sampling, governor and kernels were unchanged. No claim that native MTP is always faster.

Private raw logs, source identity, screenshot receipts, process memory samples, bounded GPU captures and remaining-path notes are in `cache-ram-ssd-20261020/takeover/` and `ornith122b-claude-20261019/turns/RAM1-*` / `RAM2-*` under the private evidence root. The inherited GPU parser mixed identical times of day across several days; the new harness records only the log bytes appended during each run.

Scope limits:
- DFlash2/DSpark and BatchEngine have separate snapshot ownership paths. This change does not claim to eliminate their retained checkpoints.
- Transient writer-owned buffers are still required. Early enqueue does not mean the bytes are already durable or no longer in RAM.
- This is not a change to long-context attention kernels or adaptive speculation policy.

- A pre-existing native-MTP limitation remains for tool steps in chats with historical media: boundary capture excludes media history, so the tool follow-up may reuse the prior conversation checkpoint rather than the immediately preceding tool step. Text-only native and media-history AR per-step reuse were observed above.
- JANGH2 and other model families have not received a new128k qualification in this change. Issue624 remains open for those paths and the separate schedulers.
