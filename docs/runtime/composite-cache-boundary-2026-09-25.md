# Composite and recurrent cache boundaries

A CacheList wrapper does not advance its own offset. Restore validation must inspect every child, reject empty composites, and require all leaf offsets to equal the matched prompt boundary. Typed nested caches must also select disk-backed restoration. Pure recurrent snapshots recover their token count from the saved recurrent offset rather than returning zero solely because no attention layer exists.

Falcon-H1 additionally failed to advance its MambaCache offset during forward execution. The mixer now advances it by the actual input sequence length, alongside updating its recurrent state.

## Evidence

Private evidence root: `/Users/eric/vmlx-private-evidence/required-followups-2026-09-25/composite`.

- `test-r1.log`: 22/23 tests passed; real tiny Falcon-H1 forward produced inconsistent child offsets after six tokens.
- `test-r2.log`: after the offset fix, offsets passed but continuation parity failed with an in-memory tensor dictionary shared by both live caches. This was not a disk round trip.
- `test-r3.log`: 23/23 tests passed, including a real safetensors save/load of the tiny production Falcon-H1 model's cache and continuation-logit equality at unchanged rtol=1e-5/atol=1e-6. The round trip isolates cache storage before either continuation mutates it.
- `build-r4-receipt.json` and `source-r4.json`: exact tested source manifest, successful Release build, no source drift or guard stop.
- `test-r3-metallib.json`: kernel library identity from engine 6827ef1153efa434f04ebeff801d9d6b0ce3e6bd; Metal sources/submodules unchanged by this fix.

Coverage includes intact and damaged cache topology records, nested/empty composite boundaries, all-recurrent token counts and state, invalid metadata, and attention-only legacy rejection for hybrid state. Positive controls invoke the production boundary validator.

This is cache-state correctness proof, not pretrained Falcon language-quality proof or a throughput benchmark. No pretrained Falcon bundle was loaded. Raptor's separate reasoning-only failure remains unresolved; these changes do not claim to fix it. No quantization, sampler, memory-limit, or release changes.
