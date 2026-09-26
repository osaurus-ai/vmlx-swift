# Composite cache coordinator store boundaries

A real Osaurus Falcon-H1-0.5B-Instruct-4bit chat at engine2f973dc6 exposed a gap in the earlier restore fix: the coordinator checked `CacheList.offset`, which stays zero while its children advance. Valid snapshots were refused, so every subsequent request cold-prefilled.

Store and restore now share recursive leaf-offset extraction. Empty composites and mismatched children remain invalid. Post-answer key alignment uses the same leaf boundaries, preserving the consumed-stop-token rule for composites.

Validation: Release build and29 tests passed, including coordinator store/fetch/typed restore for the shipped Falcon Mamba+KV layout, inconsistent-child and empty-subtree rejection, ordinary rotating caches, damaged-payload matrices and tiny Falcon continuation parity. Deeper CacheList-within-CacheList serialization remains unsupported and is explicitly tested to refuse restore; this change does not claim support for that format.

Evidence: /Users/eric/vmlx-private-evidence/required-followups-2026-09-25/coordinator-store/{source-r3.json,build-r3-receipt.json,test-r3.log,test-r3-receipt.json}. Original failed app proof is retained under cache-pin/ui-proof-r1. Two plain-chat turns ended normally at186.9/188.6tok/s, but one had inconsistent intermediate wording; the initial tool-heavy row was cancelled. Those are diagnostic rows, not model-family qualification. New Osaurus pin/live restore proof is required separately. No release authorization.
