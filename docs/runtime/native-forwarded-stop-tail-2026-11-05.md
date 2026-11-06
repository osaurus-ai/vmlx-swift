# Native MTP forwarded stop-tail checkpoint

Status: local candidate, not merge-ready. No release/tag or AgentLoop/tool edits.

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

Before landing: finish affected/full-suite comparison, build a source-bound
Release app, prove an actual longer-tail entry and subsequent useful restore,
inspect visible answer and reasoning, preserve any failure state before quota
tests, and assess final-store cost. Additional valid keys can still cause
unhelpful writes if future rendered prompts do not match them.

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
component commands/logs/xcresults. Candidate GUI proof remains pending.
