# Owned packed-bank regression contract

The shipped loader change 1747b2bdb replaced production shard-backed arrays
with direct reads into owned packed compute allocations. Two mapped-bank
tests still asserted that the retired mmap tracked-byte counter increased;
both fail with the current loader because that counter correctly stays flat.

The composition test now checks that the mmap counter stays unchanged and
the loaded packed bank does not share a mapped shard, while retaining its
existing correct-output proof after the original owner and source are gone.
The owner-release test measures live MLX allocation growth, drops the owner,
and checks that live allocation returns to its starting value. It also checks
weak-owner release and an unchanged mmap counter. Reusable allocator cache
occupancy is distinct from live packed allocations. No loader policy changes.

The original failure is retained. The corrected five bank tests and eight
resident-reader tests pass locally on Max2 alongside routed-kernel checks.
No claim is made that these small fixtures prove a full model's RAM footprint
or close a production crash. Native Activity Monitor/resource proof remains
separate from this test-contract correction.
