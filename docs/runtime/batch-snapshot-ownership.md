# Batched prompt snapshot ownership

## Problem

The batched prefill path retained an evaluated prompt snapshot even when there was no cache coordinator, or when the request was auxiliary and the terminal path never stored it. This duplicates cache state through decode for no consumer.

The terminal boundary helper also copied cache state before checking whether the media identity was preserved and every layer supported trimming. Non-trimmable recurrent cache stacks could therefore allocate a throwaway copy before taking the recurrent reconstruction path.

## Change

- Capture the exact prompt snapshot only for a coordinator-backed, non-auxiliary request. Existing DSV4 topology rules remain in the shared helper.
- Check trim eligibility on the source snapshot before copying. Valid trims still work on a private copy and evaluate it before publication.

These changes do not remove the necessary live KV cache, recurrent rollback state, or the payload retained by an in-flight SSD write. They do not change paged cache defaults, disk formats, sampling, speculative depth, kernels, or the writer's scheduling.

## Validation status

Source parsing and diff checks pass. Allocation and snapshot-independence tests pass 2/2; existing BatchEngineIntegrationTests pass 30/30. An additional source-inspection suite has an obsolete assertion against unchanged Evaluate.swift and its Swift Testing runner also lacked the default metallib; that suite is not counted as passing. Actual Osaurus app proof is pending. This is not an all-family RAM qualification and does not close issue #624.
