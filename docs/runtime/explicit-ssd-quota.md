# Explicit SSD cache quotas

Explicit percentages through 100% and legacy GB limits are honored. Automatic remains 30% of free space plus the cache's owned payloads. A quota is a ceiling, not a reservation; the app shows an advisory when the requested ceiling exceeds available space.

## Validation

The production policy executable passed 32 assertions covering explicit high shares, legacy limits, precedence, stable quotas as owned bytes grow, unknown readings, full volumes, and overflow handling.

Built Osaurus #3154 at15c41c01 with engine08e3097ad. In the actual isolated Release app, saved60% and80% on the external SSD; the loaded coordinator adopted80% without a quota-induced unload. Relaunch preserved the setting and restored6777 of7998 tokens from that SSD. Three connected tool/recall rows returned correct391 and483 totals and all four recalled inventory fields. Physical footprint peaks were6.99/6.46/6.09GB, cache-store returns3.0–6.4ms, and background writes42–93ms. These short rows are not a cross-family memory or speculative-speed qualification.

Private receipts: `~/vmlx-private-evidence/ssd-quota-20261021/live-results.json`, per-turn captures, settings screenshots and launch manifests. GitHub proof comments on engine#630 and app#3154 summarize the same evidence. No writer, eviction-order, kernel, sampling, or speculative-depth changes.
