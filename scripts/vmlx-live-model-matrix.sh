#!/usr/bin/env bash
set -u

usage() {
  cat <<'EOF'
Usage:
  scripts/vmlx-live-model-matrix.sh [options]

Options:
  --models-root PATH      Model root. Default: ~/models
  --model PATH            Add one model directory. Repeatable. Defaults to all
                          discovered config.json bundles under --models-root.
  --run-dir PATH          Artifact directory. Default:
                          docs/local/live-model-matrix/<timestamp>
  --profile NAME          inventory|metadata|text|batch|vl|omni|mtp|all. Default: inventory
  --max-size-gb N         Skip live load above N GB unless --allow-huge. Default: 20
  --allow-huge            Permit live loads above --max-size-gb.
  --no-build              Reuse existing .build/debug/RunBench.
  --dry-run               Write planned commands but do not execute live loads.
  -h, --help              Show this help.

Profiles:
  inventory   Write models.tsv only.
  metadata    Run no/low-load config and template smokes.
  text        Run BENCH_PROD with an explicit cache coordinator.
  batch       Run B=1, multi-turn, cache-hit, B=2, per-slot sampler, and TQ B=2.
  vl          Run VL BatchEngine chat and media-salt cache probes.
  omni        Run Nemotron Omni probe with BatchEngine stress enabled.
  mtp         Run focused MTP metadata tests for MTP-looking bundles.
  all         metadata plus the model-family live profile.

This is a proof harness, not a pass generator. Skipped or failed rows remain
blocked/failed until the artifact says otherwise.
EOF
}

MODELS_ROOT="${HOME}/models"
RUN_DIR=""
PROFILE="inventory"
MAX_SIZE_GB=20
ALLOW_HUGE=0
BUILD=1
DRY_RUN=0
MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-root)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MODELS_ROOT="$2"; shift 2 ;;
    --model)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MODELS+=("$2"); shift 2 ;;
    --run-dir)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      RUN_DIR="$2"; shift 2 ;;
    --profile)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      PROFILE="$2"; shift 2 ;;
    --max-size-gb)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MAX_SIZE_GB="$2"; shift 2 ;;
    --allow-huge)
      ALLOW_HUGE=1; shift ;;
    --no-build)
      BUILD=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

case "$PROFILE" in
  inventory|metadata|text|batch|vl|omni|mtp|all) ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="docs/local/live-model-matrix/$(date -u +"%Y%m%dT%H%M%SZ")"
fi
mkdir -p "$RUN_DIR"
: >"${RUN_DIR}/status.tsv"
: >"${RUN_DIR}/commands.sh"

json_value() {
  local file="$1" query="$2" fallback="$3"
  jq -r "$query // \"$fallback\"" "$file" 2>/dev/null || printf "%s\n" "$fallback"
}

model_size_bytes() {
  du -sk "$1" 2>/dev/null | awk '{print $1 * 1024}'
}

model_size_gb() {
  awk -v bytes="$1" 'BEGIN { printf "%.2f", bytes / 1024 / 1024 / 1024 }'
}

is_gt_gb() {
  awk -v bytes="$1" -v gb="$2" 'BEGIN { exit !(bytes > gb * 1024 * 1024 * 1024) }'
}

has_file_named() {
  find "$1" -maxdepth 2 -name "$2" -print -quit 2>/dev/null | grep -q .
}

contains_mtp_evidence() {
  local dir="$1"
  [[ "$(basename "$dir" | tr '[:upper:]' '[:lower:]')" == *mtp* ]] && return 0
  if [[ -f "$dir/config.json" ]] &&
     jq -e '.. | objects | to_entries[]? | select((.key|ascii_downcase|contains("mtp")) and (.value != null))' \
       "$dir/config.json" >/dev/null 2>&1
  then
    return 0
  fi
  if [[ -f "$dir/jang_config.json" ]] &&
     jq -e '.. | objects | to_entries[]? | select((.key|ascii_downcase|contains("mtp")) and (.value != null))' \
       "$dir/jang_config.json" >/dev/null 2>&1
  then
    return 0
  fi
  return 1
}

classify_profile() {
  local dir="$1" arch="$2"
  if [[ "$arch" == NemotronHForCausalLM* ]]; then
    printf "omni"; return
  fi
  if [[ "$arch" == *VL* ]] || has_file_named "$dir" preprocessor_config.json; then
    printf "vl"; return
  fi
  printf "text"
}

discover_models() {
  if [[ ${#MODELS[@]} -gt 0 ]]; then
    printf "%s\n" "${MODELS[@]}"
  else
    find "$MODELS_ROOT" -maxdepth 3 -name config.json -print |
      sed 's#/config.json$##' |
      sort -u
  fi
}

write_inventory() {
  printf "status\tsize_gb\tbytes\tprofile\tmtp\tarchitecture\tmodel_type\tgen_max_new_tokens\tgen_temperature\tgen_top_p\tgen_top_k\tgen_min_p\tgen_repetition_penalty\tgen_do_sample\tpath\n" \
    >"${RUN_DIR}/models.tsv"
  while IFS= read -r dir; do
    [[ -n "$dir" && -f "$dir/config.json" ]] || continue
    local bytes size arch model_type profile mtp gen_config gen_max gen_temp gen_top_p gen_top_k gen_min_p gen_rep gen_do_sample
    bytes="$(model_size_bytes "$dir")"
    size="$(model_size_gb "$bytes")"
    arch="$(json_value "$dir/config.json" '.architectures?[0]' unknown)"
    model_type="$(json_value "$dir/config.json" '.model_type // .text_config.model_type' unknown)"
    profile="$(classify_profile "$dir" "$arch")"
    mtp="no"
    if contains_mtp_evidence "$dir"; then mtp="yes"; fi
    gen_config="$dir/generation_config.json"
    if [[ -f "$gen_config" ]]; then
      gen_max="$(json_value "$gen_config" '.max_new_tokens' nil)"
      gen_temp="$(json_value "$gen_config" '.temperature' nil)"
      gen_top_p="$(json_value "$gen_config" '.top_p' nil)"
      gen_top_k="$(json_value "$gen_config" '.top_k' nil)"
      gen_min_p="$(json_value "$gen_config" '.min_p' nil)"
      gen_rep="$(json_value "$gen_config" '.repetition_penalty' nil)"
      gen_do_sample="$(json_value "$gen_config" '.do_sample' nil)"
    else
      gen_max="missing"; gen_temp="missing"; gen_top_p="missing"
      gen_top_k="missing"; gen_min_p="missing"; gen_rep="missing"
      gen_do_sample="missing"
    fi
    printf "discovered\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$size" "$bytes" "$profile" "$mtp" "$arch" "$model_type" \
      "$gen_max" "$gen_temp" "$gen_top_p" "$gen_top_k" "$gen_min_p" \
      "$gen_rep" "$gen_do_sample" "$dir" \
      >>"${RUN_DIR}/models.tsv"
  done < <(discover_models)
}

run_logged() {
  local name="$1"; shift
  local out="${RUN_DIR}/${name}.out"
  local err="${RUN_DIR}/${name}.err"
  printf "%q " "$@" >>"${RUN_DIR}/commands.sh"
  printf "\n" >>"${RUN_DIR}/commands.sh"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf "%s\tdry-run\n" "$name" >>"${RUN_DIR}/status.tsv"
    return 0
  fi
  "$@" >"$out" 2>"$err"
  local code=$?
  if [[ "$code" -eq 0 ]]; then
    if grep -qi "not applicable" "$out"; then
      printf "%s\tn-a\n" "$name" >>"${RUN_DIR}/status.tsv"
    else
      printf "%s\tpass\n" "$name" >>"${RUN_DIR}/status.tsv"
    fi
    return 0
  fi
  printf "%s\tfail:%s\n" "$name" "$code" >>"${RUN_DIR}/status.tsv"
  return "$code"
}

run_runbench() {
  local name="$1"; shift
  run_logged "$name" env "$@" .build/debug/RunBench
}

matrix_max_tokens() {
  printf "%s" "${VMLX_MATRIX_MAX_TOKENS:-${VMLINUX_MATRIX_MAX_TOKENS:-64}}"
}

run_batch_stack() {
  local name="$1" dir="$2" max_tokens="$3"
  run_runbench "${name}.batch_single" \
    BENCH_MODEL="$dir" BENCH_BATCH=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.batch_chat" \
    BENCH_MODEL="$dir" BENCH_BATCH_CHAT=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.batch_cache_hit" \
    BENCH_MODEL="$dir" BENCH_BATCH_CACHE_HIT=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.batch_disk_restore" \
    BENCH_MODEL="$dir" BENCH_BATCH_DISK_RESTORE=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.batch_concurrent_b2" \
    BENCH_MODEL="$dir" BENCH_BATCH_CONCURRENT=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.batch_perslot_sampler_b2" \
    BENCH_MODEL="$dir" BENCH_BATCH_PERSLOT_SAMPLER=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.batch_tq_b2" \
    BENCH_MODEL="$dir" BENCH_BATCH_TQ_B2=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
}

safe_name() {
  basename "$1" | tr -c 'A-Za-z0-9._-' '_'
}

maybe_build() {
  [[ "$BUILD" -eq 0 || "$PROFILE" == "inventory" ]] && return 0
  run_logged build_runbench swift build --jobs "${VMLINUX_SWIFT_BUILD_JOBS:-2}" --product RunBench
}

write_inventory
maybe_build

if [[ "$PROFILE" == "inventory" ]]; then
  {
    printf "# vMLX Live Model Matrix\n\n"
    printf -- "- run dir: %s\n" "$RUN_DIR"
    printf -- "- profile: inventory\n"
    printf -- "- inventory: models.tsv\n"
  } >"${RUN_DIR}/REPORT.md"
  echo "inventory: ${RUN_DIR}/models.tsv" >&2
  exit 0
fi

while IFS=$'\t' read -r status size_gb bytes family_profile mtp arch model_type gen_max gen_temp gen_top_p gen_top_k gen_min_p gen_rep gen_do_sample dir; do
  [[ "$status" == "discovered" ]] || continue
  name="$(safe_name "$dir")"
  if [[ "$ALLOW_HUGE" -eq 0 ]] && is_gt_gb "$bytes" "$MAX_SIZE_GB"; then
    printf "%s\tskipped:size>%sGB\n" "$name" "$MAX_SIZE_GB" >>"${RUN_DIR}/status.tsv"
    continue
  fi

  if [[ "$PROFILE" == "metadata" || "$PROFILE" == "all" ]]; then
    run_runbench "${name}.config" BENCH_MODEL="$dir" BENCH_CONFIG_SMOKE=1 BENCH_MAX_TOKENS=8 || true
    run_runbench "${name}.template" BENCH_MODEL="$dir" BENCH_TEMPLATE_SMOKE=1 BENCH_MAX_TOKENS=8 || true
  fi

  if [[ "$PROFILE" == "mtp" || ( "$PROFILE" == "all" && "$mtp" == "yes" ) ]]; then
    expects_vl=0
    [[ "$family_profile" == "vl" ]] && expects_vl=1
    run_logged "${name}.mtp" env \
      VMLX_MTP_REAL_BUNDLE="$dir" \
      VMLX_MTP_REAL_BUNDLE_EXPECTS_VL="$expects_vl" \
      swift test --filter MTPRuntimeFocusedTests --jobs 2 || true
  fi

  live_profile="$PROFILE"
  [[ "$PROFILE" == "all" ]] && live_profile="$family_profile"

  case "$live_profile" in
    text)
      run_runbench "${name}.prod" \
        BENCH_MODEL="$dir" BENCH_PROD=1 BENCH_PROD_COORD=1 \
        BENCH_MAX_TOKENS="$(matrix_max_tokens)" || true
      ;;
    batch)
      run_batch_stack "$name" "$dir" "$(matrix_max_tokens)"
      ;;
    vl)
      run_runbench "${name}.vl_batch_chat" \
        BENCH_MODEL="$dir" BENCH_VL_BATCH_CHAT=1 \
        BENCH_MAX_TOKENS="$(matrix_max_tokens)" || true
      run_runbench "${name}.vl_media_salt" \
        BENCH_MODEL="$dir" BENCH_VL_BATCH_MEDIASALT=1 \
        BENCH_MAX_TOKENS="$(matrix_max_tokens)" || true
      ;;
    omni)
      run_runbench "${name}.omni" \
        BENCH_MODEL="$dir" BENCH_OMNI=1 BENCH_OMNI_BATCH=1 \
        BENCH_MAX_TOKENS="$(matrix_max_tokens)" || true
      ;;
    metadata|mtp)
      ;;
  esac
done <"${RUN_DIR}/models.tsv"

{
  printf "# vMLX Live Model Matrix\n\n"
  printf -- "- run dir: %s\n" "$RUN_DIR"
  printf -- "- profile: %s\n" "$PROFILE"
  printf -- "- max size GB: %s\n" "$MAX_SIZE_GB"
  printf -- "- allow huge: %s\n" "$ALLOW_HUGE"
  printf -- "- dry run: %s\n\n" "$DRY_RUN"
  printf "## Status\n\n"
  printf "| Row | Status |\n|---|---|\n"
  while IFS=$'\t' read -r row row_status; do
    [[ -n "$row" ]] || continue
    printf "| %s | %s |\n" "$row" "$row_status"
  done <"${RUN_DIR}/status.tsv"
} >"${RUN_DIR}/REPORT.md"

echo "report: ${RUN_DIR}/REPORT.md" >&2
