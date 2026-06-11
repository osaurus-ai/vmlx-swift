#!/usr/bin/env python3
"""Download Gemma 4 QAT model repos into ~/models.

This is a deterministic Hugging Face Hub download helper, not an agent runner.
It materializes each repo sequentially so disk and network pressure stay bounded.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


REPOS = [
    "OsaurusAI/gemma-4-12B-it-qat-JANG_4M",
    "OsaurusAI/gemma-4-31B-it-qat-JANG_4M",
    "OsaurusAI/gemma-4-26B-A4B-it-qat-JANG_4M",
    "OsaurusAI/gemma-4-E4B-it-qat-JANG_4M",
    "OsaurusAI/gemma-4-E2B-it-qat-JANG_4M",
    "OsaurusAI/gemma-4-31B-it-qat-MXFP4",
    "OsaurusAI/gemma-4-26B-A4B-it-qat-MXFP4",
    "OsaurusAI/gemma-4-12B-it-qat-MXFP4",
    "OsaurusAI/gemma-4-E4B-it-qat-MXFP4",
    "OsaurusAI/gemma-4-E2B-it-qat-MXFP4",
]


def repo_slug(repo: str) -> str:
    return repo.replace("/", "--")


def log(line: str, log_file: Path) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = f"[{stamp}] {line}"
    print(text, flush=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def main() -> int:
    models_dir = Path(os.environ.get("MODELS_DIR", str(Path.home() / "models"))).expanduser()
    log_dir = Path(os.environ["DOWNLOAD_LOG_DIR"]).expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download-all.log"
    status_file = log_dir / "status.jsonl"

    for repo in REPOS:
        dest = models_dir / repo_slug(repo)
        dest.mkdir(parents=True, exist_ok=True)
        started = time.time()
        log(f"START {repo} -> {dest}", log_file)
        record = {
            "repo": repo,
            "dest": str(dest),
            "started_at": started,
            "status": "started",
        }
        with status_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                max_workers=8,
            )
        except Exception as exc:
            record = {
                "repo": repo,
                "dest": str(dest),
                "finished_at": time.time(),
                "status": "failed",
                "error": repr(exc),
            }
            with status_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            log(f"FAIL {repo}: {exc!r}", log_file)
            continue

        record = {
            "repo": repo,
            "dest": str(dest),
            "finished_at": time.time(),
            "elapsed_seconds": round(time.time() - started, 3),
            "status": "ok",
        }
        with status_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        log(f"OK {repo}", log_file)

    log("DONE", log_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
