# Cold-tier removal — staged draft

Requested removal of JangPress / MLXPress is in progress. This change removes router-observation and after-layer cold-page-sweep calls, their coordinator implementations, loader configuration/page-advice calls and the end-of-turn advisor wait. It does not alter routers, quantized matmul, Hadamard/trellis transforms, canonical safetensors mmap, or KV/SSD caches.

Do not merge this draft as full removal. Loader activation, policy/status APIs, probe/embed tiers, experimental prestacking and Mach storage, CLI products, downstream app settings and legacy tests still need removal or disentangling. AffineStreamingSwitchGLU uses the shard reader for actual weights; preserve that canonical reader separately. Safetensors alignment repair must retain its explicit authorization and byte checks.

No speed or RAM improvement is claimed. Source parsing is separate from compilation and actual-app load/restore proof; those are pending on this draft. Production previously disabled the advisor by default, so the quota defect and unused KV checkpoints are separate fixes (#630/#631).
