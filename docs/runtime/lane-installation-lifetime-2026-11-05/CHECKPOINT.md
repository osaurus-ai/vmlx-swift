# DFlash lane installation follows model lifetime

The installer remembered target and drafter ObjectIdentifier values for the
entire process. After a model unload, Swift can allocate a new model at the
same address. The old set then reports it as already installed, leaving the new
model on ordinary projections. The separate lane-flat dictionary could also
report policy from an unrelated previous owner.

A real-device regression creates and releases 64 target/drafter pairs. Against
the prior Common build it produced 120 missing-acceleration assertions in
3.158 seconds. This is an installation correctness check, not a model speed
benchmark. Existing source and failed logs are retained.

The candidate stores a weak owner with each installation record, prunes dead
records and binds lane-flat policy to that same live owner. It does not retain
model weights. Kernel math/layout, sampler, prompts, draft-width policy and
cache payloads are unchanged. AgentLoop, orchestration and in-app tools are
untouched.

The broader focused run also found two test-harness problems: a source-contract
check opened Module.swift relative to Xcode's temporary working directory, and
a standalone child xctest inherited the parent IDE session and hung before
executing its test. Source-relative lookup preserves the exact assertion;
stripping inherited XCTest/XCInject controller environment lets the child use
its explicit test filter. A controlled child run additionally reproduced
SIGPIPE only with Xcode OS_ACTIVITY_DT_MODE log redirection; the identical
production-default logger run passed. The child now removes that debugger
override too. Raw failures, control logs and child stack sample remain.

The fresh 287-source Common build passed. The identical strengthened regression
failed 124 acceleration assertions against the old build and passed all 64
lifetimes against the candidate, including weak-owner release checks. No
throughput conclusion follows from this small installation test.

The normal full focused run completed: 246 XCTest cases (8 skipped) and
995 Swift Testing cases. Lane lifetime tests pass. The run remains failed due
to other working-directory-sensitive source checks and existing parser/config
contracts; all original failures are retained. The focused Xcode retry passed all 23 selected tests, including the
source-path contract, lane classification/lifetimes and all three child logger
modes. The original full-suite failure is not relabeled as green.

A normal Release Osaurus dev app on production implementation9842ca9e then
completed three actual Qwen3.8-27B load/unload cycles. Each fresh instance
installed305 target projections (65 tiled) and47 drafter projections. Actual
UI observations confirm unloaded state before cycles2 and3. All three growing
image-history turns returned the correct invoice/tool values, stopping naturally
at122/94/143 tokens with native rates79.95/92.92/94.80tokens/s. Restores consumed
11,163/11,545/11,942 tokens. The first entry came from AR without optional
features, so authentic conditioning grew389→786→1,215 rows, below the declared
2,047 cap. This is not a full-window restore or paired speed benchmark.

Sampled physical footprint peaked at23,265,997,304 bytes and fails the low-RAM
gate under retained owned loading. Earlier quality/numerical failures remain.
Current production source is identical to the compiled app; subsequent commits
only change focused tests and this documentation. No release/tag.

Status: the bounded lifetime/cache change has source, Metal-regression,
focused-Xcode and actual-dev-app proof. Broader family/quality/performance and
remaining full-suite findings stay open.
No merged-fix, family-quality, RAM or speed claim. Original DFlash arithmetic,
long-answer and numerical-parity failures remain separate open items.
