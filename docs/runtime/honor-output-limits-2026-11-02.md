# Honor requested output-token limits

A DeepSeek-V4.1 request with `max_tokens=1000000` could stop with `length`
after exactly 11,387 generated tokens. Agent controls and the request were
correct: a lower-level Metal buffer projection silently replaced the limit.
The historical formula was `(499000 * 0.9 - 16384) / 38`. It also assumed
499,000 resources when the OS did not expose `iogpu.rsrc_limit`.

BatchSlot and TokenIterator now keep the requested limit exactly, including
an omitted limit. Buffer projections are advisory only and require an actual
OS resource-limit value. The historical `clampedMaxTokens` symbol remains a
source-compatible passthrough. DeepSeek-V4.1's growable contiguous pool slabs
also no longer declare the older per-token-buffer retention estimate.

This does not remove real allocation failures, guarantee a million-token
answer, force the model to continue, or change its sampler or stop tokens.
An explicit short output budget still stops generation at that budget.

## Validation

The focused suite passed 22 tests across five suites. The regressions exercise real
BatchSlot initialization for omitted, million-token and eight-token limits;
run 12,001 constant-logit CPU TokenIterator steps across the old cutoff;
check the explicit eight-token stop; and distinguish contiguous V4.1 pools
from the historical retention estimate. Existing long-pool disk round trips
cover non-aligned lengths above 16K rows.

The constant-logit fixture tests admission and stopping. It is not a
DeepSeek model-quality, DSpark throughput, hardware resource-capacity or
million-token generation qualification.
