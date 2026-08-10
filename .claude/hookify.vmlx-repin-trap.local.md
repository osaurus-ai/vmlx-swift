---
name: vmlx-repin-trap
enabled: true
event: bash
pattern: gh\s+pr\s+merge.*vmlx|git\s+(merge|push).*vmlx-origin
---

🔗 **You just changed vmlx. osaurus pins it by SHA — the fix does NOTHING until you repin.**

This has silently failed before: a PR titled "+ vmlx repin" merged with **no repin**, and the live gate then
"passed" against a binary that did **not** contain the change. That is false confidence, and it is worse than
a red test.

**The SHA lives in SIX places. A partial repin leaves the app building against the OLD engine:**

1. `Packages/OsaurusCore/Package.swift` (`revision:`)
2. `Packages/OsaurusCore/Package.resolved`  ← easy to miss; SwiftPM rewrites it on build
3. `osaurus.xcworkspace/xcshareddata/swiftpm/Package.resolved`
4. `App/osaurus.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`
5. `Packages/OsaurusCore/Tests/Service/RuntimePolicySourceTests.swift` (tripwire test)
6. `Packages/OsaurusCore/Tests/Service/ImageGenerationBridgeContractTests.swift` (tripwire test)

**Verify, don't assume:**

```bash
grep -rl "<OLD_SHA>" . | grep -vE '\.build|build-|SourcePackages|\.git/'   # must be EMPTY
```

**Then PROVE the new engine is really in the app:**

```bash
strings build-release/.../osaurus.app/Contents/MacOS/osaurus | grep -c "<new symbol from the fix>"
```

If that count is 0, the app is running the **old** engine and any gate you run is meaningless.
Rebuild the app after every repin — a stale binary is the #1 way we've fooled ourselves.
