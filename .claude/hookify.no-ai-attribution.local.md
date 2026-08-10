---
name: no-ai-attribution
enabled: true
event: bash
pattern: (git\s+commit|gh\s+pr\s+(create|edit)|git\s+tag)[\s\S]*(Co-Authored-By|Generated with|Claude Code|Claude Opus|Claude Sonnet|AI-assisted|🤖)
---

⛔ **ABSOLUTE RULE: no AI attribution in anything GitHub-visible.**

Never add to commits, PR titles/bodies, changelogs, release notes, READMEs, or any GitHub-visible content:
- `Co-Authored-By: Claude` (or any AI co-author trailer)
- "Generated with Claude Code" / "🤖 Generated with…" footers
- Any mention of Claude, Codex, or AI assistance

**No exceptions.** All commits and PRs are Eric's work.

Rewrite the message and try again.
