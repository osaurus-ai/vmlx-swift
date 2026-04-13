# vmlx-swift — local development fork

This is a local working copy of `osaurus-ai/mlx-swift` (branch
`osaurus-0.31.3`) set up as the upstream for `vmlx-swift-lm` optimization
work. **Not pushed to any remote.** Edits here flow directly into
`../vmlx-swift-lm` on the next `swift build` because `vmlx-swift-lm`'s
`Package.swift` uses a local path dependency.

## Layout

| Path | Purpose |
|------|---------|
| `/Users/eric/vmlx-swift` | this fork |
| `/Users/eric/vmlx-swift-lm` | model library consuming it |

## Branches

| Branch | Purpose |
|--------|---------|
| `osaurus-0.31.3` | pristine `osaurus-ai/mlx-swift` tracking branch, don't edit |
| `vmlx-0.31.3` (HEAD) | working branch for optimizations |

Submodule `Source/Cmlx/mlx` also has a local branch:

| Branch | Purpose |
|--------|---------|
| (detached at upstream) | default `ml-explore/mlx` state |
| `vmlx-patches-0.31.3` (HEAD) | committed patches for `GatherQMM::output_shapes` + `CustomKernel::set_output_shapes` that `vmlx-swift-lm`'s compiled-kernel path depends on |

## Remotes

| Name | URL | Purpose |
|------|-----|---------|
| `osaurus-upstream` | `https://github.com/osaurus-ai/mlx-swift.git` | pull latest osaurus work into `osaurus-0.31.3` |

**No `origin` remote.** A plain `git push` will fail. This is intentional —
the fork is local-only until we decide to publish.

## Iteration workflow

1. Edit a file under `Source/MLX/…` or `Source/Cmlx/mlx/mlx/…`.
2. In `../vmlx-swift-lm`, run `swift build -c release`. SPM picks up the
   local path dep automatically and rebuilds the Cmlx C++ + MLX Swift
   modules against the new source.
3. Run a bench (`swift run -c release CompileBench` for op-level,
   `swift run -c release RunBench` for full Qwen 3.5-35B multi-turn).
4. Iterate.

## Runtime quirk: `default.metallib`

`swift build` from the terminal does NOT compile the `.metal` kernel
files in `Source/Cmlx/mlx-generated/metal/` into a `default.metallib`
(only Xcode's build phase does that). As a workaround:

```sh
cp ~/Library/Developer/Xcode/DerivedData/vmlx-swift-lm-*/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib \
   /Users/eric/vmlx-swift-lm/default.metallib
```

The C++ loader falls through four paths before finally trying
`default_mtllib_path` (= literal `"default.metallib"` resolved against
cwd). Running `swift run` from `vmlx-swift-lm/` with a metallib sitting
in that directory makes the load succeed.

**TODO:** add an SPM build plugin or prebuild command to `Package.swift`
that compiles `mlx-generated/metal/*.metal` into
`Sources/Cmlx/Resources/default.metallib` and declares it as a resource.
Then the Xcode DerivedData workaround goes away. This is blocked on
deciding whether to patch Package.swift directly or submit the change
upstream to `osaurus-ai/mlx-swift`.

## Switching back to URL dependency

When this fork is not needed, edit `vmlx-swift-lm/Package.swift`:

```swift
// From:
.package(name: "mlx-swift", path: "../vmlx-swift"),
// Back to:
.package(url: "https://github.com/osaurus-ai/mlx-swift", branch: "osaurus-0.31.3"),
```

Then `swift package clean && swift build -c release`.

## Current optimization targets

Per `vmlx-swift-lm/docs/research/2026-04-13-decode-speed-to-120.md`, the
Swift-vs-Python gap on Qwen 3.5-35B-A3B-4bit is ~10-12 tok/s end-to-end,
and microbench attribution puts it at:

1. **~5 tok/s** — Swift eager per-op overhead (~0.5 us/op extra vs Python)
2. **~3 tok/s** — compile-site coverage differences (Python compiles
   more than Swift captures)
3. **~3 tok/s** — structural: ARC on `MLXArray` class, module-boundary
   dispatch, heap alloc on every op return

Concrete things to try in this fork:

- [ ] `@inlinable` on hot-path `MLXArray+Ops.swift` methods (`exp`,
      `add`, `multiply`, `matmul`, `reshape`) **plus** `MLXArray.init(_:)`
      so cross-module inlining actually works end-to-end
- [ ] `@inline(__always)` on `MLXArray.ctx` getter to avoid dispatch on
      every bridge call
- [ ] Pool `mlx_array_new()` allocations to avoid repeated C-side
      alloc/free on hot paths (requires mlx-c change)
- [ ] Investigate whether `MLXArray` can become a struct (with a
      `ManagedBuffer` holding `mlx_array`) instead of a `final class`.
      High impact but breaks every caller.
- [ ] Build plugin for Metal kernel compilation — removes the Xcode
      DerivedData metallib workaround and enables clean CI builds

Measurements go in the parent repo's research doc.
