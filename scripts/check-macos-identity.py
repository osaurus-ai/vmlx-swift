#!/usr/bin/env python3
# Copyright © 2026 Osaurus AI. All rights reserved.
# SPDX-License-Identifier: MIT
"""check-macos-identity.py: show that macOS compiles exactly the same code at two commits.

    python3 scripts/check-macos-identity.py <base-ref> [<new-ref>] [--mlx-new <ref>]

new-ref defaults to HEAD. Linux-only work in vmlx-swift must not change what Apple platforms compile.
For each commit this script rebuilds the tree from git alone (the superproject, and its submodules at
their recorded commits; no network), then compares:

  1. the sources SwiftPM compiles for the Cmlx target on macOS (`swift package describe`);
  2. each of those sources preprocessed by clang (-E -P) under the Cmlx settings that
     `swift package dump-package` reports for a macOS debug build, whitespace-normalized, with
     __LINE__, __DATE__ and __TIME__ fixed, so that a guard which only adds lines is no difference;
  3. the files the Xcode project compiles from its synchronized mlx and mlx-c folders (tracked files
     minus membershipExceptions, the lists tools/update-xcode-membership.swift writes). The project's
     other synchronized folders (mlx-conditional, mlx-generated, fmt, metal-cpp, framework) are not
     compared.

--mlx-new takes the mlx submodule's commit on the new side from a ref of the submodule itself, so a
fork commit can be checked before vmlx records it. Both trees are materialized in turn at the same
path, so __FILE__ expands identically. Only committed state is compared. Prints `macOS identity holds`
and exits 0; names every difference and exits 1; exits 2 when the check itself cannot run. macOS only.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN = ROOT / ".build" / "check-macos-identity" / f"run-{os.getpid()}"
TREE = RUN / "tree"
TARGET = "Source/Cmlx"
MLX = "Source/Cmlx/mlx"
# The Xcode project's exception sets for its synchronized folders, by object id, and the submodule
# each folder shows.
XCODE_EXCEPTION_SETS = {
    "C3AE9EA62EAAABFC000BD280": MLX,
    "C3AE9E162EAAAB37000BD280": "Source/Cmlx/mlx-c",
}
LANGUAGES = {
    ".c": "c", ".m": "objective-c", ".mm": "objective-c++", ".cc": "c++", ".cpp": "c++",
    ".cxx": "c++", ".CPP": "c++", ".c++": "c++", ".s": "assembler-with-cpp",
    ".S": "assembler-with-cpp",
}
# Environment variables Package.swift reads. The check evaluates the manifest's defaults.
MANIFEST_VARIABLES = ("VMLX_", "MLX_SWIFT_BUILD_DOC", "SPI_GENERATE_DOCS")
# Macros that would make identical code preprocess differently: the line number (assert expands it,
# and a guard above an assert shifts it) and the build time.
FIXED_MACROS = ["-Wno-builtin-macro-redefined", "-D__LINE__=0", '-D__DATE__="Jan  1 2000"',
                '-D__TIME__="00:00:00"']
# What a SwiftPM debug build of Cmlx adds on macOS (Swift Build's response files).
DEBUG_DEFINES = ["-DSWIFT_PACKAGE=1", "-DDEBUG=1",
                 "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG"]


def run(args, **kwargs):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if result.returncode != 0:
        command = " ".join(str(a) for a in args)
        raise RuntimeError(f"{command} failed:\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def submodule_commits(ref, mlx_override):
    """Every submodule `ref` records, as path -> commit, with mlx's replaced by `mlx_override`."""
    commits = {}
    for line in run(["git", "-C", str(ROOT), "ls-tree", "-r", ref]).decode().splitlines():
        meta, path = line.split("\t", 1)
        _, kind, sha = meta.split()
        if kind == "commit":
            commits[path] = sha
    if mlx_override:
        commits[MLX] = run(["git", "-C", str(ROOT / MLX), "rev-parse", "--verify",
                            mlx_override + "^{commit}"]).decode().strip()
    return commits


def materialize(ref, commits):
    """Writes the commit's tree, with its submodules at `commits`, to TREE."""
    if TREE.exists():
        shutil.rmtree(TREE)
    TREE.mkdir(parents=True)
    run(["tar", "-x", "-C", str(TREE)], input=run(["git", "-C", str(ROOT), "archive", ref]))
    for path, sha in commits.items():
        (TREE / path).mkdir(parents=True, exist_ok=True)
        run(["tar", "-x", "-C", str(TREE / path)],
            input=run(["git", "-C", str(ROOT / path), "archive", sha]))


def swiftpm(*arguments):
    environment = {k: v for k, v in os.environ.items() if not k.startswith(MANIFEST_VARIABLES)}
    return json.loads(run(["swift", "package", "--package-path", str(TREE), *arguments],
                          env=environment))


def cmlx_sources():
    description = swiftpm("describe", "--type", "json")
    return sorted(next(t for t in description["targets"] if t["name"] == "Cmlx")["sources"])


def cmlx_flags():
    """(C flags, C++ flags) for Cmlx in a macOS debug build, from the manifest's own settings."""
    manifest = swiftpm("dump-package")
    cmlx = next(t for t in manifest["targets"] if t["name"] == "Cmlx")
    by_tool = {"c": [], "cxx": []}
    for setting in cmlx.get("settings", []):
        condition = setting.get("condition") or {}
        platforms = condition.get("platformNames")
        if platforms and "macos" not in platforms:
            continue  # an empty list means "every platform"
        if condition.get("config") not in (None, "debug"):
            continue
        if setting["tool"] not in by_tool:
            continue  # linker settings do not change what compiles
        kind = setting["kind"]
        if "define" in kind:
            by_tool[setting["tool"]].append("-D" + kind["define"]["_0"])
        elif "headerSearchPath" in kind:
            by_tool[setting["tool"]].append("-I" + str(TREE / TARGET / kind["headerSearchPath"]["_0"]))
        else:
            raise RuntimeError(f"unhandled Cmlx setting {setting}: teach check-macos-identity.py")
    sdk = run(["xcrun", "--show-sdk-path"]).decode().strip()
    common = ["-E", "-P", "-target", "arm64-apple-macos14.0", "-isysroot", sdk,
              "-I" + str(TREE / TARGET / "include"), *DEBUG_DEFINES, *FIXED_MACROS]
    standard = "-std=" + (manifest.get("cxxLanguageStandard") or "gnu++20")
    return common + by_tool["c"], common + [standard] + by_tool["c"] + by_tool["cxx"]


def preprocess(source, c_flags, cxx_flags):
    """(sha256, compressed text) of the source's whitespace-normalized preprocessed form, or
    (None, clang's error) when it does not preprocess."""
    language = LANGUAGES[pathlib.PurePosixPath(source).suffix]
    cxx = language in ("c++", "objective-c++")
    try:
        output = run(["xcrun", "clang++" if cxx else "clang", *(cxx_flags if cxx else c_flags),
                      "-x", language, str(TREE / TARGET / source)])
    except RuntimeError as error:
        return None, str(error)
    lines = (" ".join(line.split()) for line in output.decode(errors="replace").splitlines())
    text = ("\n".join(line for line in lines if line) + "\n").encode()
    return hashlib.sha256(text).hexdigest(), zlib.compress(text)


def exception_list(pbxproj, object_id):
    start = pbxproj.index(f"\t\t{object_id} /*")
    head = "membershipExceptions = (\n"
    begin = pbxproj.index(head, start) + len(head)
    end = pbxproj.index("\n\t\t\t);", begin)
    items = set()
    for line in pbxproj[begin:end].splitlines():
        item = line.strip().rstrip(",")
        if item.startswith('"') and item.endswith('"'):
            item = item[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if item:
            items.add(item)
    return items


def xcode_members(ref, commits):
    pbxproj = run(["git", "-C", str(ROOT), "show",
                   f"{ref}:xcode/MLX.xcodeproj/project.pbxproj"]).decode()
    members = set()
    for object_id, path in XCODE_EXCEPTION_SETS.items():
        if path not in commits:
            continue
        tracked = run(["git", "-C", str(ROOT / path), "ls-tree", "-r", "--name-only", commits[path]])
        excepted = exception_list(pbxproj, object_id)
        members |= {f"{path}/{f}" for f in tracked.decode().splitlines() if f not in excepted}
    return members


def snapshot(ref, mlx_override=None):
    commits = submodule_commits(ref, mlx_override)
    materialize(ref, commits)
    sources = cmlx_sources()
    c_flags, cxx_flags = cmlx_flags()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        texts = dict(zip(sources, pool.map(lambda s: preprocess(s, c_flags, cxx_flags), sources)))
    return sources, texts, xcode_members(ref, commits)


def main():
    if sys.platform != "darwin":
        print("check-macos-identity.py runs on macOS only", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base")
    parser.add_argument("new", nargs="?", default="HEAD")
    parser.add_argument("--mlx-new", help="a ref of the mlx submodule to use on the new side")
    args = parser.parse_args()
    try:
        base_sources, base_texts, base_members = snapshot(args.base)
        new_sources, new_texts, new_members = snapshot(args.new, args.mlx_new)
    except Exception as error:  # the check could not run: that is no verdict
        print(f"check-macos-identity.py: {error}", file=sys.stderr)
        shutil.rmtree(RUN, ignore_errors=True)
        return 2
    shutil.rmtree(TREE, ignore_errors=True)
    new_label = args.new + (f" (mlx {args.mlx_new})" if args.mlx_new else "")
    # A source compiled at both commits must preprocess at both, or nothing was compared. One compiled
    # at only one commit is a difference either way, and is reported as one.
    broken = [(label, source, texts[source][1])
              for label, texts in ((args.base, base_texts), (new_label, new_texts))
              for source in sorted(set(base_sources) & set(new_sources)) if texts[source][0] is None]
    for label, source, error in broken:
        print(f"check-macos-identity.py: {source} does not preprocess at {label}: {error}",
              file=sys.stderr)
    if broken:
        shutil.rmtree(RUN, ignore_errors=True)
        return 2
    problems = []
    for source in sorted(set(base_sources) ^ set(new_sources)):
        where, texts = (args.base, base_texts) if source in base_sources else (new_label, new_texts)
        note = " (it does not preprocess on macOS)" if texts[source][0] is None else ""
        problems.append(f"Cmlx source list: {source} is compiled only at {where}{note}")
    for source in sorted(set(base_sources) & set(new_sources)):
        if base_texts[source][0] != new_texts[source][0]:
            stem = RUN / "diff" / source
            stem.parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(f"{stem}.base.i").write_bytes(zlib.decompress(base_texts[source][1]))
            pathlib.Path(f"{stem}.new.i").write_bytes(zlib.decompress(new_texts[source][1]))
            problems.append(f"preprocessed source differs: {source} "
                            f"(diff -u {stem}.base.i {stem}.new.i)")
    for member in sorted(base_members ^ new_members):
        where = args.base if member in base_members else new_label
        problems.append(f"Xcode project member: {member} is compiled only at {where}")
    for problem in problems:
        print(problem)
    if problems:
        print(f"macOS identity FAILS: {len(problems)} difference(s) between {args.base} and {new_label}")
        return 1
    shutil.rmtree(RUN, ignore_errors=True)
    print(f"macOS identity holds: {len(new_sources)} Cmlx sources and {len(new_members)} Xcode "
          f"members compile the same at {args.base} and {new_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
