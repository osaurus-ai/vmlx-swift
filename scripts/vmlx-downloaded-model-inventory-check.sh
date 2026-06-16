#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${VMLINUX_MODEL_INVENTORY_AUDIT_DIR:-/tmp/vmlx-downloaded-model-inventory-$(date +%Y%m%d-%H%M%S)}"
MODELS_ROOTS_RAW="${VMLINUX_MODEL_INVENTORY_ROOTS:-/Users/eric/models:/Users/eric/osaurus_models/finished}"
mkdir -p "$OUT_DIR"

fail=0
pass() { echo "PASS $*"; }
fail_msg() { echo "FAIL $*" >&2; fail=1; }

RELEASE_LEDGER="$ROOT/.agents/vmlx-osaurus/codex/RELEASE-READINESS.md"
PR_CHECKLIST="$ROOT/.agents/vmlx-osaurus/codex/PR-PROOF-CHECKLIST.md"
MATRIX="$ROOT/scripts/vmlx-live-model-matrix.sh"
NO_HIDDEN_TESTS="$ROOT/Tests/MLXLMCommonFocusedTests/NoHiddenReasoningCloseBiasFocusedTests.swift"
TOPOLOGY_TESTS="$ROOT/Tests/MLXLMCommonFocusedTests/CacheCoordinatorTopologyFocusedTests.swift"
RUNBENCH="$ROOT/RunBench/Bench.swift"
OMNIBENCH="$ROOT/RunBench/OmniBench.swift"

require_file() {
  local file="$1" label="$2"
  if [[ -f "$file" ]]; then
    pass "$label exists"
  else
    fail_msg "missing $label: $file"
  fi
}

require_text() {
  local file="$1" pattern="$2" label="$3"
  if rg -q "$pattern" "$file"; then
    pass "$label"
  else
    fail_msg "missing $label in ${file#$ROOT/}"
  fi
}

require_inventory_path() {
  local pattern="$1" label="$2"
  if rg -q "$pattern" "$OUT_DIR/models.tsv"; then
    pass "downloaded inventory includes $label"
  else
    fail_msg "downloaded inventory missing $label"
  fi
}

family_doc_pattern() {
  case "$1" in
    qwen) printf 'Qwen' ;;
    qwen3.5) printf 'Qwen3\.5|Qwen' ;;
    gemma) printf 'Gemma' ;;
    zaya) printf 'ZAYA|CCA' ;;
    zaya-vl) printf 'ZAYA|VL|media' ;;
    dsv4) printf 'DSV4|DeepSeek-V4|CSA/HSA/SWA' ;;
    ling-bailing) printf 'Ling|Bailing' ;;
    minimax) printf 'MiniMax' ;;
    nemotron-omni) printf 'Nemotron|Omni' ;;
    laguna) printf 'Laguna' ;;
    hy3) printf 'Hy3|Hunyuan' ;;
    mistral) printf 'Mistral|Ministral' ;;
    *) printf '%s' "$1" ;;
  esac
}

family_label() {
  case "$1" in
    qwen) printf 'Qwen3.6/Qwen family' ;;
    qwen3.5) printf 'Qwen3.5 family' ;;
    gemma) printf 'Gemma family' ;;
    zaya) printf 'ZAYA text family' ;;
    zaya-vl) printf 'ZAYA VL family' ;;
    dsv4) printf 'DeepSeek-V4/DSV4 family' ;;
    ling-bailing) printf 'Ling/Bailing hybrid family' ;;
    minimax) printf 'MiniMax family' ;;
    nemotron-omni) printf 'Nemotron Omni family' ;;
    laguna) printf 'Laguna family' ;;
    hy3) printf 'Hy3/Hunyuan family' ;;
    mistral) printf 'Mistral/Ministral family' ;;
    *) printf '%s family' "$1" ;;
  esac
}

IFS=':' read -r -a MODEL_ROOTS <<<"$MODELS_ROOTS_RAW"
python3 - "$OUT_DIR" "${MODEL_ROOTS[@]}" <<'PY'
import csv
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
roots = [pathlib.Path(p).expanduser() for p in sys.argv[2:]]

def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def first_arch(config):
    arch = config.get("architectures")
    if isinstance(arch, list) and arch:
        return str(arch[0])
    text = config.get("text_config")
    if isinstance(text, dict):
        arch = text.get("architectures")
        if isinstance(arch, list) and arch:
            return str(arch[0])
    return "unknown"

def model_type(config):
    value = config.get("model_type")
    if value:
        return str(value)
    text = config.get("text_config")
    if isinstance(text, dict) and text.get("model_type"):
        return str(text["model_type"])
    return "unknown"

def family_for(path, config):
    lowered = f"{path.name} {path} {model_type(config)} {first_arch(config)}".lower()
    path_lowered = f"{path.name} {path}".lower()
    has_preprocessor = (path / "preprocessor_config.json").exists()
    if "deepseek-v4" in lowered or "_dsv4" in lowered or "deepseek_v4" in lowered:
        return "dsv4"
    if "zaya" in lowered and (has_preprocessor or "-vl" in lowered or "vl-" in lowered):
        return "zaya-vl"
    if "zaya" in lowered:
        return "zaya"
    if "gemma" in lowered:
        return "gemma"
    if "qwen3.6" in path_lowered or "qwen3_6" in path_lowered:
        return "qwen"
    if "qwen3.5" in lowered or "qwen3_5" in lowered:
        return "qwen3.5"
    if "qwen" in lowered:
        return "qwen"
    if "ling" in lowered or "bailing" in lowered:
        return "ling-bailing"
    if "minimax" in lowered:
        return "minimax"
    if "nemotron" in lowered or "omni" in lowered:
        return "nemotron-omni"
    if "laguna" in lowered:
        return "laguna"
    if "hy3" in lowered or "hunyuan" in lowered:
        return "hy3"
    if "mistral" in lowered or "ministral" in lowered:
        return "mistral"
    return "other"

def topology_for(family, path, config):
    has_preprocessor = (path / "preprocessor_config.json").exists()
    if family in {"qwen", "qwen3.5"}:
        return "hybrid-ssm-kv"
    if family == "gemma":
        return "rotating-swa-kv"
    if family in {"zaya", "zaya-vl"}:
        return "cca-companion" + ("+media" if has_preprocessor else "")
    if family == "dsv4":
        return "csa-hsa-swa-pool"
    if family == "ling-bailing":
        return "linear-attention-arrays+kv"
    if family == "minimax":
        return "routed-moe-kv"
    if family == "nemotron-omni":
        return "omni-audio-video-kv"
    if family == "laguna":
        return "glm-thinking-kv"
    if family == "hy3":
        return "hunyuan-companion"
    if family == "mistral":
        return "mistral-kv"
    return "unknown"

def gen_summary(path):
    gen = read_json(path / "generation_config.json")
    if not gen:
        return "missing"
    keys = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "do_sample"]
    present = [key for key in keys if key in gen]
    return ",".join(present) if present else "present-no-sampler-keys"

rows = []
seen = set()
for root in roots:
    if not root.exists():
        continue
    for config_path in root.rglob("config.json"):
        path = config_path.parent
        if path in seen:
            continue
        seen.add(path)
        config = read_json(config_path)
        family = family_for(path, config)
        if family == "other":
            continue
        rows.append({
            "family": family,
            "topology": topology_for(family, path, config),
            "architecture": first_arch(config),
            "model_type": model_type(config),
            "generation_defaults": gen_summary(path),
            "path": str(path),
        })

rows.sort(key=lambda row: (row["family"], row["path"]))
with (out_dir / "models.tsv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["family", "topology", "architecture", "model_type", "generation_defaults", "path"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

counts = {}
examples = {}
for row in rows:
    counts[row["family"]] = counts.get(row["family"], 0) + 1
    examples.setdefault(row["family"], row["path"])

with (out_dir / "family-counts.tsv").open("w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["family", "count", "example"])
    for family in sorted(counts):
        writer.writerow([family, counts[family], examples[family]])
PY

require_file "$OUT_DIR/models.tsv" "downloaded model inventory"
require_file "$OUT_DIR/family-counts.tsv" "downloaded model family counts"
require_file "$RELEASE_LEDGER" "release readiness ledger"
require_file "$PR_CHECKLIST" "PR proof checklist"
require_file "$MATRIX" "live model matrix"
require_file "$NO_HIDDEN_TESTS" "no-hidden parser/default focused tests"
require_file "$TOPOLOGY_TESTS" "cache topology focused tests"
require_file "$RUNBENCH" "RunBench harness"
require_file "$OMNIBENCH" "OmniBench harness"

echo "--- discovered families ---"
cat "$OUT_DIR/family-counts.tsv"

if awk -F '\t' 'NR > 1 { found=1 } END { exit found ? 1 : 0 }' "$OUT_DIR/family-counts.tsv"; then
  fail_msg "downloaded inventory discovered no supported model families"
fi

require_inventory_path 'dealign.ai/Qwen3.6-35B-A3B-MXFP4-CRACK-MTP' \
  "reported dealignai Qwen3.6 35B MXFP4 MTP slug"

while IFS=$'\t' read -r family count example; do
  [[ "$family" == "family" || -z "$family" ]] && continue
  pass "downloaded inventory includes $(family_label "$family") ($count)"
done <"$OUT_DIR/family-counts.tsv"

echo "--- matrix/source coverage ---"
require_text "$MATRIX" 'qwen_multiturn_tool' "matrix has Qwen multi-turn tool lane"
require_text "$MATRIX" 'vl_mixed_text_image_video' "matrix has VL/video media lane"
require_text "$MATRIX" 'omni' "matrix has Omni lane"
require_text "$MATRIX" 'n-a:deepseek-v4-uses-swa-csa-hsa-hybrid-pool-cache-not-turboquant-kv' \
  "matrix refuses DSV4 generic TurboQuant substitution"
require_text "$MATRIX" 'BENCH_GROWING_CHAT_CACHE=1' "matrix has growing-chat prefix cache lane"
require_text "$MATRIX" 'BENCH_BATCH_DISK_RESTORE=1' "matrix has disk L2 restore lane"
require_text "$MATRIX" 'BENCH_BATCH_TQ_B2=1' "matrix has TurboQuant KV B=2 lane"
require_text "$NO_HIDDEN_TESTS" 'Ling/Bailing|Laguna|MiniMax|Nemotron|Hy3|Mistral' \
  "focused parser/default tests name non-Qwen/Gemma parser families"
require_text "$TOPOLOGY_TESTS" 'Bailing/Ling|ZAYA|DSV4|Gemma4|SSM' \
  "topology tests name hybrid/CCA/DSV4/Gemma cache families"
require_text "$RUNBENCH" 'Laguna|MiniMax|Nemotron|Qwen3.5|Bailing|Ling' \
  "RunBench names downloaded non-Qwen/Gemma runtime families"
require_text "$OMNIBENCH" 'Nemotron' "OmniBench names Nemotron Omni runtime"

echo "--- release/checklist coverage ---"
require_text "$RELEASE_LEDGER" 'Downloaded model family inventory' \
  "release ledger has downloaded model inventory row"
require_text "$PR_CHECKLIST" 'Downloaded model inventory' \
  "PR checklist has downloaded model inventory section"
while IFS=$'\t' read -r family count example; do
  [[ "$family" == "family" || -z "$family" ]] && continue
  pattern="$(family_doc_pattern "$family")"
  label="$(family_label "$family")"
  require_text "$RELEASE_LEDGER" "$pattern" "release ledger names discovered $label"
  require_text "$PR_CHECKLIST" "$pattern" "PR checklist names discovered $label"
done <"$OUT_DIR/family-counts.tsv"
require_text "$RELEASE_LEDGER" 'not live-promoted|not release-promoted|live proof.*required|partial/failed|not fully proven' \
  "release ledger keeps non-promoted families bounded"

echo "inventory_dir=$OUT_DIR"

if [[ "$fail" -ne 0 ]]; then
  echo "Downloaded model inventory coverage guard failed." >&2
  exit 1
fi

echo "Downloaded model inventory coverage guard passed."
