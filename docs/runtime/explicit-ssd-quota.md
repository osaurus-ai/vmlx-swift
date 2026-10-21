# Explicit SSD cache quotas

An explicit disk percentage was accepted up to100% but then limited to25% of free space plus indexed cache payloads. Automatic independently used30%, so editing the setting could reduce the effective limit below Automatic.

Explicit percentages now use their requested share of total volume capacity without the host ceiling. Legacy GB choices are also honored. Automatic still uses30% of free plus owned payloads; filling the cache does not shrink that budget. A quota is a maximum, not a reservation. Low free space and a request above available space are advisory; real filesystem write errors remain errors. Unknown measurements preserve the existing fallback/previous-cap behavior. No allocation, cache format, SSD writer or eviction-order changes.

Validation in progress: exact production-policy reproduction and32assertions pass, including high percentages, legacy sizes, growth stability, unknown measurements and overflow. Focused suites and built-app save/relaunch/runtime proof remain pending. See the paired Osaurus PR for UI behavior and subsequent evidence. This does not claim to fix duplicate RAM ownership across model families or remove MLXPress/JangPress.
