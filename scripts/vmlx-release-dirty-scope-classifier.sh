#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/vmlx-release-dirty-scope-classifier-$(date +%Y%m%d-%H%M%S).md}"
mkdir -p "$(dirname "$OUT")"

all_dirty="$({
  git -C "$ROOT" diff --name-only
  git -C "$ROOT" diff --cached --name-only
  git -C "$ROOT" ls-files --others --exclude-standard
} | sort -u)"

classify() {
  local path="$1"
  case "$path" in
    .gitignore)
      echo "release-guard" ;;
    scripts/vmlx-osaurus-release-readiness-audit.sh|scripts/vmlx-release-dirty-scope-classifier.sh|scripts/vmlx-push-readiness-scope-check.sh|scripts/vmlx-architecture-cache-proof-check.sh|scripts/vmlx-qwen-gemma-proof-check.sh|scripts/vmlx-family-matrix-coverage-check.sh|scripts/vmlx-downloaded-model-inventory-check.sh|scripts/vmlx-ram-footprint-proof-boundary-check.sh|scripts/vmlx-reasoning-visible-boundary-check.sh)
      echo "release-guard" ;;
    .agents/vmlx-osaurus/codex/*)
      echo "coordination-doc" ;;
    .logs/*|codex/logs/*)
      echo "evidence-log-artifact" ;;
    Libraries/vMLXFlux/*|Libraries/vMLXFluxKit/*|Libraries/vMLXFluxModels/*|Libraries/vMLXFluxVideo/*|Tests/vMLXFluxTests/*|tools/vMLXFluxProbe/*)
      echo "flux-generated-or-unrelated" ;;
    Package.swift|Libraries/MLXLMCommon/*|Libraries/MLXHuggingFaceMacros/*|RunBench/*|Tests/MLXLMCommonFocusedTests/*|Tests/MLXLMTests/*|Tests/MLXPressPolicyTests/*|scripts/vmlx-live-model-matrix.sh|AGENTS.md)
      echo "runtime-source-or-test" ;;
    docs/*|goalnew.md)
      echo "docs-or-planning" ;;
    *)
      echo "unknown" ;;
  esac
}

{
  echo "# vMLX release dirty-scope classification"
  echo
  echo "Repo: $ROOT"
  echo "HEAD: $(git -C "$ROOT" rev-parse HEAD)"
  echo
  if [[ -z "$all_dirty" ]]; then
    echo "No dirty paths."
    exit 0
  fi

  printf '%s\t%s\n' "category" "path" >"$OUT.tsv"
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    printf '%s\t%s\n' "$(classify "$path")" "$path" >>"$OUT.tsv"
  done <<<"$all_dirty"

  echo "## Counts"
  echo
  awk -F '\t' 'NR>1 { count[$1]++ } END { for (c in count) print "- " c ": " count[c] }' "$OUT.tsv" | sort
  echo
  echo "## Release interpretation"
  echo
  echo "- release-guard and coordination-doc paths are expected release-readiness work."
  echo "- evidence-log-artifact paths should not be published unless explicitly curated."
  echo "- runtime-source-or-test paths require scoped ownership, review, and proof before release."
  echo "- flux-generated-or-unrelated paths must be isolated from the vMLX/Osaurus release unless Eric explicitly expands scope."
  echo "- unknown paths are blockers until manually classified."
  echo
  for category in release-guard coordination-doc runtime-source-or-test evidence-log-artifact flux-generated-or-unrelated docs-or-planning unknown; do
    if awk -F '\t' -v c="$category" 'NR>1 && $1 == c { found=1 } END { exit found ? 0 : 1 }' "$OUT.tsv"; then
      echo "## $category"
      echo
      awk -F '\t' -v c="$category" 'NR>1 && $1 == c { print "- `" $2 "`" }' "$OUT.tsv"
      echo
    fi
  done
} >"$OUT"

echo "$OUT"
