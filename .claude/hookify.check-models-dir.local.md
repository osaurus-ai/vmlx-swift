---
name: check-models-dir
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: (find|ls|grep|rg|test|\[)\s+.*(MLXModels|\.osaurus/models)
  - field: command
    operator: not_contains
    pattern: /models
---

🧭 **Model search — check `~/models` FIRST.**

Eric's MASTER model library is **`~/models`** (ALL variants live there:
`~/models/JANGQ-AI/` = Ornith 1.0 9B+35B MXFP4/MXFP8/JANG; `~/models/OsaurusAI/` =
OsaurusAgent-9b MXFP4/MXFP8; `~/models/dealign.ai/` = CRACK packs). You are searching
`~/MLXModels` or `~/.osaurus/models` WITHOUT `~/models` — this is the recurring mistake
that keeps making you falsely conclude a model is "not on disk."

Before concluding anything is missing, run:
`find ~/models -maxdepth 3 -iname '*<name>*'`

`~/MLXModels` = osaurus's scan dir only (symlink from `~/models`, same volume).
`~/.osaurus/models` = legacy orphan trap (osaurus does NOT scan it).
