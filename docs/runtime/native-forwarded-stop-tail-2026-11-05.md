# Native MTP forwarded stop-tail checkpoint

Status: scoped component and actual-app cache gates passed; normal PR/CI landing
pending. No release/tag or AgentLoop/tool edits.

An actual Flash-Next native-MTP GUI request on runtime `6f450646` (production
identical to merged `6b0e7ce9`) had 4,675 prompt tokens, 142 visible generated
tokens, 143 returned tokens and all cache leaves at 4,819. The verified queue
contained an accepted stop token and newline before the unforwarded bonus.
The existing one-stop path could not key that two-row recurrent overhang, so
it safely skipped the final entry. This is lost reuse, not cache corruption.

The candidate derives the exact forwarded tail from the actual accepted queue
only at final persistence. It requires homogeneous leaf offsets, exactly one
returned token absent from generated IDs, the last returned ID at the current
queue position, the already-returned queue prefix matching generated history,
and the entire remaining accepted queue matching the overhang. The final
bonus/correction is excluded. Exact, one-stop and trimmable paths are retained;
uncertain alignment still skips. No decode math, sampler, depth, prompt or
model configuration changes.

Component gates cover queue positions/depths, invalid alignment, and actual
nonaligned nine-token KV/Mamba disk store/reopen/restore. Both old short keys
must remain rejected, persistent recurrent slots and KV values must survive,
and scratch slots must remain absent. The first test run passed 52 functions
but did not include the new CPU test file in Package.swift's explicit list;
that harness omission is retained and corrected, with the corrected affected run passing 54 functions in four suites.
No substitute pass is claimed.

The final full focused target records 1,253 total / 1,207 passed / 9 failed /
37 skipped, with the same nine failing methods as the prior baseline and no
new failing identifiers. Existing failures remain open.

Actual source-bound Release40 (runtime `42e77371`, consumer `b96359b6`) ran
five GUI-only text requests with bundle defaults and Adaptive native MTP.
The first three identical candidate requests produced correct visible answers
and expanded reasoning at 56.6, 75.7 and 66.2 tok/s. The third committed a
1,734-token checkpoint from 1,591 prompt + 141 visible tokens + two exact
forwarded tokens. The next GUI turn restored all 1,734 tokens from disk and
correctly calculated 271830 × 3 = 815490 at 69.8 tok/s. It produced another
two-token-tail entry at 1,962 tokens. After clean process restart, a new GUI
turn restored all 1,962 tokens and correctly calculated 815490 − 490 = 815000
at 63.3 tok/s, including correct expanded reasoning. All end-states retained;
both app/physical-sampler pairs closed zero/zero.

The two new disk payloads are 163,606,660 and 169,910,412 bytes. Each has 36
recurrent offsets exactly matching its key; twelve attention KV layers plus
36 recurrent layers persist. The model key labels `kv=fp16`, but actual
serialized attention tensors are BF16 and recurrent tensors BF16/F32. Paged
RAM is off; prefix and block disk caches are on. No TurboQuant topology claim.
Coordinator-store return traces are 1.418/1.484 ms and background writes each
21 ms (zero queued ms). These are component timings, excluding other host,
snapshot, load and UI work; no paired speedup or whole-request latency claim.

The new fixture exposes four tools and a shorter system message (1,591 input
tokens) versus the older ten-tool 4,675-token diagnostic. Its first three
requests match each other exactly, not that old input. Do not use these rows
to claim the prior reasoning failure was fixed. No tool/AgentLoop code was
changed to force a match. Peak process physical footprint was 54,765,394,296
bytes, then 53,626,853,080 after restart: low RAM remains unqualified/FAIL.
Owned loading is retained. No general family, media, batching or settings
qualification. No new terminal command was executed in this candidate run;
the earlier real 45-second command proof remains separate.

Separate quality finding: the same nonempty request sometimes generates
empty-user/audit reasoning with native MTP, solo AR and an empty KV cache.
An existing prompt-dump diagnostic captured the real nonempty question with
no audit directive; verified token fragments contain the bad reasoning before
the stop token. Root cause remains unexplained. This cache candidate does not
claim to fix it. Owned model loading has high physical footprint; low RAM and
family-wide quality/performance remain unqualified.

SOURCE EVIDENCE: `Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift`,
`Tests/MLXLMCommonFocusedTests/NativeForwardedStopTailTests.swift`,
`Tests/MLXLMCommonFocusedTests/NativeDiskDurabilityTests.swift`, `Package.swift`.

LIVE EVIDENCE: local `goal-storage-cache-resume-20261104/`
`native-mtp-terminal-current/cold-native39-1` raw log, expanded UI, activity,
verified queue fragments and retained cache; `prompt-observed-native39/`
rendered prompt/readback and actual UI; `native-forwarded-stop-tail/`
component commands/logs/xcresults. Candidate GUI receipt: `native-forwarded-stop-tail/ACTUAL-GUI-RECEIPT.json`;
full focused comparison in `native-forwarded-stop-tail/full-focused/`.
The final source differs from the tested app only by log-block indentation
with exact nonwhitespace bytes and quoted literals verified.
