# Auxiliary hybrid prefill disk isolation

Internal utility requests carry `LMInput.CachePromptIntent.auxiliary`: they may
restore existing prefixes but must not publish prompt boundaries. The batched
hybrid-pool path captured and stored an N-1 disk seed before the final-generation
auxiliary guard. A description or title utility could therefore write a boundary
even though finalization correctly skipped persistence.

The prefill capture condition now excludes auxiliary requests. The store entry
point also checks intent. Ordinary generation and reusable-prefix warmups retain
their N-1 capture behavior. Cache lookup, generation defaults, quantization and
disk format are unchanged.

## Regression evidence

`AuxiliaryPrefillSeedTests` executes `BatchEngine.submit` with a tiny hybrid-pool
model and a real disk coordinator. It checks three intents:

- Ordinary generation and warmup each split an eight-token prompt into seven
  and one, and actually write a safetensors boundary.
- A cold auxiliary request prepares all eight tokens without publishing a file.
- An auxiliary request following either positive control restores the seven-token
  disk boundary, prepares only the remaining token, and leaves payload bytes and
  the set of payload files unchanged.

The unmodified engine failed both auxiliary no-capture and no-write assertions.
After the fix, all three parameterized cases passed, including both restore
checks. Raw receipts are in the local private evidence directory
`/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23`:
`auxiliary-isolated-red-r5.log`, `auxiliary-green-r2.log`,
`auxiliary-green-r2-receipt.json`, and `auxiliary-green-r2-source.json`.

The test is registered in `MLXLMCommonFocusedTests`. The root Release test build
hit an existing unrelated `MLXArrayEmptyHostReadTests` reference to the debug-only
`withEvalLockForTesting` helper. The recorded Release run used an isolated test
package depending on the real engine products, with the same test/support files
symlinked and the same-pin app metallib supplied at its colocated lookup path.
Earlier fixture/build/resource failures remain in the evidence directory.

This is a bounded cache regression, not real-model quality, speed, app UI or
all-family cache qualification. The fixture ends on its first EOS sample and
does not establish a decode-throughput result. MiMo audio qualification and the
Raptor performance/eviction campaign remain separate open work. No release or
tag is authorized.
