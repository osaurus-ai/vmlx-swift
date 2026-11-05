# Current benchmark guard inventory

The focused reasoning-visibility guard still read
`RunBench/JangPressRegressionBench.swift`, removed with the cold-tier APIs in
`1af9a11c`. The test threw file-not-found before completing its validation.
No production inference failure is established by that obsolete source path.

Inventory every current recursive benchmark Swift source, matching the
RunBench target's source directory. Require a nonempty inventory, the four
still-existing original benchmark subjects, and nonempty source text. Retain
all forbidden reasoning-fallback assertions and apply the former single-driver
implicit-temperature check to every current source. This covers35 current
files instead of dropping the guard or recreating a dead benchmark.

Normal macOS Xcode passes21test functions in three suites, including all
NoHiddenReasoningCloseBias, TopP shape and JANG default controls. The previous
same-source run failed only the missing legacy driver check. The final style
wrap is whitespace-only; no new changed-line style diagnostic. No runtime,
benchmark implementation, AgentLoop or app-tool changes. App39/source-qualified
runtime remains byte-identical; no new family/model qualification is claimed.
The wider focused target remains red for independently registered failures.

SOURCE EVIDENCE:
`Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift`.
LIVE EVIDENCE: local `goal-storage-cache-resume-20261104/full-focused-xcode/`
`current-benchmark-guard-inventory/Focused.xcresult`, frozen source binding,
35-file SHA inventory and whitespace-only style binding. Earlier missing-file
failures remain in the full sampler candidate run. No release/tag.
