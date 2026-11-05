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
stripping only inherited XCTest/XCInject controller environment lets the child
use its explicit test filter. Raw failure and child stack sample remain.

The fresh 287-source Common build passed. The identical strengthened regression
failed 124 acceleration assertions against the old build and passed all 64
lifetimes against the candidate, including weak-owner release checks. No
throughput conclusion follows from this small installation test.

Status: full focused test retry and actual dev-app load/unload proof are pending.
No merged-fix, family-quality, RAM or speed claim. Original DFlash arithmetic,
long-answer and numerical-parity failures remain separate open items.
