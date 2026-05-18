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
  --profile NAME          inventory|metadata|infer|text|batch|vl|omni|mtp|turnmatrix|all.
                          Default: inventory
  --max-size-gb N         Skip live load above N GB unless --allow-huge. Default: 20
  --allow-huge            Permit live loads above --max-size-gb.
  --exclude-regex REGEX   Skip discovered/model paths matching REGEX. Repeatable
                          patterns are ORed. Example: 'Kimi|DeepSeek-V4|DSV4'.
  --release               Build/run .build/release/RunBench. Required for speed gates.
  --no-build              Reuse existing RunBench for the selected configuration.
  --dry-run               Write planned commands but do not execute live loads.
  -h, --help              Show this help.

Profiles:
  inventory   Write models.tsv only, including separate mtp_tensors and
              mtp_auto columns so tensor evidence is not confused with a
              supported native-MTP launch policy.
  metadata    Run no/low-load config and template smokes.
  infer       Run plain inference rows only: metadata, BENCH_PROD without a
              cache coordinator, plus direct media inference for VL/Omni.
  text        Run BENCH_PROD with an explicit cache coordinator.
  batch       Run B=1, multi-turn, cache-hit, B=2, per-slot sampler, and TQ B=2.
  vl          Run VL BatchEngine chat and media-salt cache probes.
  omni        Run Nemotron Omni probe with BatchEngine stress enabled.
  mtp         Run focused MTP metadata tests for MTP-looking bundles.
  turnmatrix  Run the production turn matrix for the detected family:
              metadata, MTP metadata when present, tiered cache OFF/ON,
              reasoning ON/OFF, batch cache stack, and media rows for VL/Omni.
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
EXCLUDE_REGEX=""
BUILD_CONFIGURATION="debug"
SWIFT_DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

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
    --exclude-regex)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      if [[ -z "$EXCLUDE_REGEX" ]]; then
        EXCLUDE_REGEX="$2"
      else
        EXCLUDE_REGEX="${EXCLUDE_REGEX}|${2}"
      fi
      shift 2 ;;
    --release)
      BUILD_CONFIGURATION="release"; shift ;;
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
  inventory|metadata|infer|text|batch|vl|omni|mtp|turnmatrix|all) ;;
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
  python3 - "$dir" <<'PY'
import json
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def int_value(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None

def safetensors_header_names(path):
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return []
            header_len = struct.unpack("<Q", raw)[0]
            if header_len <= 0 or header_len > 64 * 1024 * 1024:
                return []
            header = json.loads(handle.read(header_len))
    except Exception:
        return []
    return [key for key in header if key != "__metadata__"]

def tensor_names():
    index = load_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if isinstance(weight_map, dict):
        return list(weight_map)
    names = []
    for path in sorted(root.glob("*.safetensors")):
        names.extend(safetensors_header_names(path))
    return names

config = load_json(root / "config.json")
text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
prefixes = []
for base_layer in (
    int_value(config.get("num_hidden_layers")),
    int_value(text_config.get("num_hidden_layers")),
):
    if base_layer is not None and base_layer > 0:
        prefixes.append(f"model.layers.{base_layer}.")
        prefixes.append(f"language_model.model.layers.{base_layer}.")

def is_mtp_tensor(name):
    lower = name.lower()
    if lower.startswith("mtp.") or lower.startswith("model.mtp_layers."):
        return True
    if ".mtp." in lower or ".mtp_layers." in lower:
        return True
    if "nextn" in lower or "next_n" in lower:
        return True
    return any(name.startswith(prefix) for prefix in prefixes)

sys.exit(0 if any(is_mtp_tensor(name) for name in tensor_names()) else 1)
PY
}

supports_mtp_auto_launch() {
  local dir="$1"
  python3 - "$dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def normalize(value):
    return str(value or "").strip().lower().replace("-", "_")

def int_value(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None

config = load_json(root / "config.json")
text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
jang = load_json(root / "jang_config.json")

model_types = set()
for value in (
    config.get("model_type"),
    text_config.get("model_type"),
    (jang.get("source_model") or {}).get("architecture") if isinstance(jang.get("source_model"), dict) else None,
):
    value = normalize(value)
    if value:
        model_types.add(value)

supported = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_5_vl",
    "qwen3_vl",
}
if not any(value in supported for value in model_types):
    sys.exit(1)

quant = jang.get("quantization") if isinstance(jang.get("quantization"), dict) else {}
arch = jang.get("architecture") if isinstance(jang.get("architecture"), dict) else {}
config_quant = config.get("quantization") if isinstance(config.get("quantization"), dict) else {}

profile = normalize(quant.get("profile"))
bits = int_value(config_quant.get("bits"))
if bits is None:
    bits = int_value(quant.get("target_bits"))
is_moe = any("moe" in value for value in model_types) or bool(arch.get("has_moe"))

if profile == "jang_2k":
    sys.exit(1)
if is_moe and bits is not None and bits <= 2:
    sys.exit(1)

sys.exit(0)
PY
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
  fi | {
    if [[ -n "$EXCLUDE_REGEX" ]]; then
      grep -E -v "$EXCLUDE_REGEX" || true
    else
      cat
    fi
  }
}

write_inventory() {
  printf "status\tsize_gb\tbytes\tprofile\tmtp_tensors\tmtp_auto\tarchitecture\tmodel_type\tgen_max_new_tokens\tgen_temperature\tgen_top_p\tgen_top_k\tgen_min_p\tgen_repetition_penalty\tgen_do_sample\tpath\n" \
    >"${RUN_DIR}/models.tsv"
  while IFS= read -r dir; do
    [[ -n "$dir" && -f "$dir/config.json" ]] || continue
    local bytes size arch model_type profile mtp_tensors mtp_auto gen_config gen_max gen_temp gen_top_p gen_top_k gen_min_p gen_rep gen_do_sample
    bytes="$(model_size_bytes "$dir")"
    size="$(model_size_gb "$bytes")"
    arch="$(json_value "$dir/config.json" '.architectures?[0]' unknown)"
    model_type="$(json_value "$dir/config.json" '.model_type // .text_config.model_type' unknown)"
    profile="$(classify_profile "$dir" "$arch")"
    mtp_tensors="no"
    mtp_auto="no"
    if contains_mtp_evidence "$dir"; then
      mtp_tensors="yes"
      if supports_mtp_auto_launch "$dir"; then mtp_auto="yes"; fi
    fi
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
    printf "discovered\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$size" "$bytes" "$profile" "$mtp_tensors" "$mtp_auto" "$arch" "$model_type" \
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

mark_status() {
  local name="$1" status="$2"
  printf "%s\t%s\n" "$name" "$status" >>"${RUN_DIR}/status.tsv"
}

run_runbench() {
  local name="$1"; shift
  run_logged "$name" env "$@" ".build/${BUILD_CONFIGURATION}/RunBench"
}

matrix_max_tokens() {
  printf "%s" "${VMLX_MATRIX_MAX_TOKENS:-${VMLINUX_MATRIX_MAX_TOKENS:-192}}"
}

matrix_prod_max_tokens() {
  printf "%s" "${VMLX_MATRIX_PROD_MAX_TOKENS:-${VMLINUX_MATRIX_PROD_MAX_TOKENS:-2048}}"
}

matrix_prod_seed() {
  printf "%s" "${VMLX_MATRIX_PROD_SEED:-${VMLINUX_MATRIX_PROD_SEED:-0}}"
}

run_text_turn_matrix() {
  local name="$1" dir="$2" max_tokens="$3"
  local cache_dir="${RUN_DIR}/${name}.prod-cache"
  local prod_max_tokens prod_seed
  prod_max_tokens="$(matrix_prod_max_tokens)"
  prod_seed="$(matrix_prod_seed)"

  # Tiered cache OFF: no CacheCoordinator, so no cross-request prefix, paged,
  # disk-L2, or SSM companion state. Per-request KV remains required for real
  # autoregressive decode.
  run_runbench "${name}.prod_defaults_tiered_cache_off" \
    BENCH_MODEL="$dir" BENCH_PROD=1 BENCH_PROD_SEED="$prod_seed" \
    BENCH_MAX_TOKENS="$prod_max_tokens" || true

  run_runbench "${name}.prod_defaults_tiered_cache_on" \
    BENCH_MODEL="$dir" BENCH_PROD=1 BENCH_PROD_SEED="$prod_seed" BENCH_PROD_COORD=1 \
    BENCH_PROD_CACHE_DIR="$cache_dir" \
    BENCH_MAX_TOKENS="$prod_max_tokens" || true

  if [[ "${VMLX_MATRIX_INCLUDE_GREEDY:-${VMLINUX_MATRIX_INCLUDE_GREEDY:-0}}" == "1" ]]; then
    run_runbench "${name}.prod_greedy_tiered_cache_off" \
      BENCH_MODEL="$dir" BENCH_PROD=1 BENCH_PROD_GREEDY=1 BENCH_PROD_SEED="$prod_seed" \
      BENCH_MAX_TOKENS="$prod_max_tokens" || true
    run_runbench "${name}.prod_greedy_tiered_cache_on" \
      BENCH_MODEL="$dir" BENCH_PROD=1 BENCH_PROD_GREEDY=1 BENCH_PROD_SEED="$prod_seed" BENCH_PROD_COORD=1 \
      BENCH_PROD_CACHE_DIR="${RUN_DIR}/${name}.prod-greedy-cache" \
      BENCH_MAX_TOKENS="$prod_max_tokens" || true
  fi

  run_batch_stack "$name" "$dir" "$max_tokens"
}

run_plain_infer_matrix() {
  local name="$1" dir="$2" family_profile="$3" max_tokens="$4"
  local prod_max_tokens prod_seed
  prod_max_tokens="$(matrix_prod_max_tokens)"
  prod_seed="$(matrix_prod_seed)"

  run_runbench "${name}.infer_prod_defaults_cache_off" \
    BENCH_MODEL="$dir" BENCH_PROD=1 BENCH_PROD_SEED="$prod_seed" \
    BENCH_MAX_TOKENS="$prod_max_tokens" || true

  case "$family_profile" in
    vl)
      run_runbench "${name}.infer_vl_batch_chat" \
        BENCH_MODEL="$dir" BENCH_VL_BATCH_CHAT=1 \
        BENCH_MAX_TOKENS="$max_tokens" || true
      ;;
    omni)
      run_runbench "${name}.infer_omni" \
        BENCH_MODEL="$dir" BENCH_OMNI=1 \
        BENCH_MAX_TOKENS="$max_tokens" || true
      ;;
  esac
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

run_vl_turn_matrix() {
  local name="$1" dir="$2" max_tokens="$3"
  local video_path="${VMLX_MATRIX_VIDEO:-${VMLINUX_MATRIX_VIDEO:-Tests/MLXLMTests/Resources/1080p_30.mov}}"
  local video_resize="${VMLX_MATRIX_VIDEO_RESIZE:-${VMLINUX_MATRIX_VIDEO_RESIZE:-224}}"

  # Text-only rows on a VL bundle are the "VL payload OFF" proof. Media rows
  # below are the "VL payload ON" proof.
  run_text_turn_matrix "$name" "$dir" "$max_tokens"

  run_runbench "${name}.vl_batch_chat" \
    BENCH_MODEL="$dir" BENCH_VL_BATCH_CHAT=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.vl_chat_cache" \
    BENCH_MODEL="$dir" BENCH_VL_CHAT_CACHE=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
  run_runbench "${name}.vl_media_salt" \
    BENCH_MODEL="$dir" BENCH_VL_BATCH_MEDIASALT=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true

  if [[ -f "$video_path" ]]; then
    run_runbench "${name}.vl_mixed_text_image_video" \
      BENCH_MODEL="$dir" BENCH_VL_MIXED=1 BENCH_VIDEO="$video_path" \
      BENCH_VL_VIDEO_RESIZE="$video_resize" \
      BENCH_MAX_TOKENS="$max_tokens" || true
  else
    mark_status "${name}.vl_mixed_text_image_video" "n-a:no-video-fixture"
  fi
}

run_omni_turn_matrix() {
  local name="$1" dir="$2" max_tokens="$3"

  run_text_turn_matrix "$name" "$dir" "$max_tokens"
  run_runbench "${name}.omni" \
    BENCH_MODEL="$dir" BENCH_OMNI=1 BENCH_OMNI_BATCH=1 \
    BENCH_MAX_TOKENS="$max_tokens" || true
}

safe_name() {
  basename "$1" | tr -c 'A-Za-z0-9._-' '_'
}

maybe_build() {
  [[ "$BUILD" -eq 0 || "$PROFILE" == "inventory" ]] && return 0
  local config_args=()
  if [[ "$BUILD_CONFIGURATION" == "release" ]]; then
    config_args=(-c release)
  fi
  run_logged build_runbench env DEVELOPER_DIR="$SWIFT_DEVELOPER_DIR" \
    swift build "${config_args[@]}" --jobs "${VMLINUX_SWIFT_BUILD_JOBS:-2}" \
      --product RunBench
}

write_inventory
maybe_build

if [[ "$PROFILE" == "inventory" ]]; then
  {
    printf "# vMLX Live Model Matrix\n\n"
    printf -- "- run dir: %s\n" "$RUN_DIR"
    printf -- "- profile: inventory\n"
    printf -- "- build configuration: %s\n" "$BUILD_CONFIGURATION"
    printf -- "- exclude regex: %s\n" "${EXCLUDE_REGEX:-none}"
    printf -- "- inventory: models.tsv\n"
  } >"${RUN_DIR}/REPORT.md"
  echo "inventory: ${RUN_DIR}/models.tsv" >&2
  exit 0
fi

while IFS=$'\t' read -r status size_gb bytes family_profile mtp_tensors mtp_auto arch model_type gen_max gen_temp gen_top_p gen_top_k gen_min_p gen_rep gen_do_sample dir; do
  [[ "$status" == "discovered" ]] || continue
  name="$(safe_name "$dir")"
  if [[ "$ALLOW_HUGE" -eq 0 ]] && is_gt_gb "$bytes" "$MAX_SIZE_GB"; then
    printf "%s\tskipped:size>%sGB\n" "$name" "$MAX_SIZE_GB" >>"${RUN_DIR}/status.tsv"
    continue
  fi

  if [[ "$PROFILE" == "metadata" || "$PROFILE" == "infer" || "$PROFILE" == "all" || "$PROFILE" == "turnmatrix" ]]; then
    run_runbench "${name}.config" BENCH_MODEL="$dir" BENCH_CONFIG_SMOKE=1 BENCH_MAX_TOKENS=8 || true
    run_runbench "${name}.template" BENCH_MODEL="$dir" BENCH_TEMPLATE_SMOKE=1 BENCH_MAX_TOKENS=8 || true
  fi

  if [[ "$PROFILE" == "mtp" && "$mtp_tensors" != "yes" ]]; then
    mark_status "${name}.mtp" "n-a:no-mtp-tensors"
  elif [[ "$PROFILE" == "mtp" || ( ( "$PROFILE" == "all" || "$PROFILE" == "turnmatrix" ) && "$mtp_tensors" == "yes" ) ]]; then
    expects_vl=0
    [[ "$family_profile" == "vl" ]] && expects_vl=1
    expects_blocked=0
    [[ "$mtp_auto" == "yes" ]] || expects_blocked=1
    run_logged "${name}.mtp" env \
      DEVELOPER_DIR="$SWIFT_DEVELOPER_DIR" \
      VMLX_MTP_REAL_BUNDLE="$dir" \
      VMLX_MTP_REAL_BUNDLE_EXPECTS_VL="$expects_vl" \
      VMLX_MTP_REAL_BUNDLE_EXPECTS_BLOCKED="$expects_blocked" \
      swift test --filter MTPRuntimeFocusedTests --jobs 2 || true
  fi

  live_profile="$PROFILE"
  [[ "$PROFILE" == "all" ]] && live_profile="$family_profile"

  case "$live_profile" in
    infer)
      run_plain_infer_matrix "$name" "$dir" "$family_profile" "$(matrix_max_tokens)"
      ;;
    turnmatrix)
      case "$family_profile" in
        text) run_text_turn_matrix "$name" "$dir" "$(matrix_max_tokens)" ;;
        vl) run_vl_turn_matrix "$name" "$dir" "$(matrix_max_tokens)" ;;
        omni) run_omni_turn_matrix "$name" "$dir" "$(matrix_max_tokens)" ;;
      esac
      ;;
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
  printf -- "- build configuration: %s\n" "$BUILD_CONFIGURATION"
  printf -- "- exclude regex: %s\n" "${EXCLUDE_REGEX:-none}"
  printf -- "- max size GB: %s\n" "$MAX_SIZE_GB"
  printf -- "- prod max tokens: %s\n" "$(matrix_prod_max_tokens)"
  printf -- "- prod seed: %s\n" "$(matrix_prod_seed)"
  printf -- "- batch/media max tokens: %s\n" "$(matrix_max_tokens)"
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
