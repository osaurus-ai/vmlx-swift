#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

STAMP="${VMLX_IMAGE_PROOF_STAMP:-$(date -u +%Y-%m-%dT%H%M%SZ)}"
DOC_PREFIX=""
if [[ -f "$ROOT/docs/OSAURUS_IMAGE_UI_MANIFEST.json" ]]; then
  DOC_PREFIX="docs/"
fi

ART_ROOT="${VMLX_IMAGE_PROOF_ARTIFACT_ROOT:-${DOC_PREFIX}local/vmlx-flux-probes/${STAMP}}"
OUT_ROOT="${VMLX_IMAGE_PROOF_OUTPUT_ROOT:-${DOC_PREFIX}local/vmlx-flux-outputs/${STAMP}}"
MODEL_ROOT="${VMLX_IMAGE_MODEL_ROOT:-}"
WIDTH="${VMLX_IMAGE_PROOF_WIDTH:-512}"
HEIGHT="${VMLX_IMAGE_PROOF_HEIGHT:-512}"
SEED="${VMLX_IMAGE_PROOF_SEED:-7}"
IDEOGRAM_SEED="${VMLX_IMAGE_PROOF_IDEOGRAM_SEED:-103437}"
Z_STEPS="${VMLX_IMAGE_PROOF_Z_STEPS:-8}"
FLUX_STEPS="${VMLX_IMAGE_PROOF_FLUX_STEPS:-4}"
QWEN_IMAGE_STEPS="${VMLX_IMAGE_PROOF_QWEN_IMAGE_STEPS:-20}"
QWEN_EDIT_STEPS="${VMLX_IMAGE_PROOF_QWEN_EDIT_STEPS:-20}"
IDEOGRAM_STEPS="${VMLX_IMAGE_PROOF_IDEOGRAM_STEPS:-20}"
SKIP_BUILD="${VMLX_IMAGE_PROOF_SKIP_BUILD:-0}"
RUN_CONTRACT_CHECK="${VMLX_IMAGE_PROOF_CONTRACT_CHECK:-1}"
SUMMARY_ONLY="${VMLX_IMAGE_PROOF_SUMMARY_ONLY:-0}"
MIN_OPEN_FILES="${VMLX_IMAGE_PROOF_MIN_OPEN_FILES:-4096}"

mkdir -p "$ART_ROOT" "$OUT_ROOT"

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  CURRENT_OPEN_FILES="$(ulimit -n)"
  if [[ "$CURRENT_OPEN_FILES" != "unlimited" && "$CURRENT_OPEN_FILES" -lt "$MIN_OPEN_FILES" ]]; then
    ulimit -n "$MIN_OPEN_FILES" 2>/dev/null || true
  fi
fi

if [[ "$SKIP_BUILD" != "1" && "$SUMMARY_ONLY" != "1" ]]; then
  swift build --product vmlxflux-probe
fi

resolve_probe() {
  if [[ -n "${VMLX_FLUX_PROBE:-}" ]]; then
    printf '%s\n' "$VMLX_FLUX_PROBE"
    return
  fi

  local candidate
  for candidate in \
    ".build/arm64-apple-macosx/debug/vmlxflux-probe" \
    ".build/debug/vmlxflux-probe"
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  printf 'vmlx-image-current-proof: vmlxflux-probe executable not found after build\n' >&2
  return 1
}

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  PROBE="$(resolve_probe)"
fi
run_probe() {
  local label="$1"
  shift
  local artifacts="${ART_ROOT}/${label}"
  local outputs="${OUT_ROOT}/${label}"
  mkdir -p "$artifacts" "$outputs"
  printf '\n== %s ==\n' "$label"
  if [[ -n "$MODEL_ROOT" ]]; then
    "$PROBE" --root "$MODEL_ROOT" "$@" --artifacts "$artifacts" --output-dir "$outputs"
  else
    "$PROBE" "$@" --artifacts "$artifacts" --output-dir "$outputs"
  fi
}

APPLE_PROMPT="a red apple on a plain white background, centered, clean product photo"
MOUNTAIN_PROMPT="a blue mountain landscape under a golden sun, watercolor"
EDIT_APPLE_PROMPT="turn the apple blue while keeping it centered on a plain white background"
EDIT_PEAR_PROMPT="turn the apple into a green pear on a plain white background"
IDEOGRAM_APPLE_PROMPT='{"high_level_description":"A clean studio photograph of a green glass apple on a white ceramic plate.","style_description":{"aesthetics":"clean, crisp, minimal","lighting":"soft diffuse studio lighting","photo":"eye-level product photography with shallow depth of field","medium":"photograph","color_palette":["#31A354","#FFFFFF","#D9D9D9"]},"compositional_deconstruction":{"background":"A neutral pale studio backdrop and white tabletop.","elements":[{"type":"obj","bbox":[260,300,650,700],"desc":"A translucent green glass apple centered on a white ceramic plate with bright highlights."},{"type":"obj","bbox":[580,260,760,740],"desc":"A simple round white ceramic plate under the apple."}]}}'
IDEOGRAM_MOUNTAIN_PROMPT='{"high_level_description":"A clean stylized landscape poster of blue mountains under a warm yellow sun on a white background.","style_description":{"aesthetics":"clean, balanced, graphic","lighting":"bright even poster lighting","medium":"graphic_design","art_style":"crisp vector-style illustration with soft print texture","color_palette":["#2F6FAE","#F5C542","#FFFFFF","#D6E7F5"]},"compositional_deconstruction":{"background":"A pure white poster background with no text or lettering.","elements":[{"type":"obj","bbox":[350,160,760,840],"desc":"Layered blue mountain peaks centered in the frame."},{"type":"obj","bbox":[170,420,340,580],"desc":"A warm yellow sun above the mountains."}]}}'

if [[ "$SUMMARY_ONLY" != "1" ]]; then
  run_probe "status-load-matrix" \
    --matrix --no-generate

  run_probe "zimage-4bit-gen" \
    --model Z-Image-Turbo-mflux-4bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$Z_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "zimage-8bit-gen" \
    --model Z-Image-Turbo-mflux-8bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$Z_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "flux-schnell-4bit-gen" \
    --model FLUX.1-schnell-mflux-4bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$FLUX_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "flux-schnell-8bit-gen" \
    --model FLUX.1-schnell-mflux-8bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$FLUX_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "qwen-image-4bit-gen" \
    --model qwen-image-mflux-4bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_IMAGE_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "qwen-image-6bit-gen" \
    --model Qwen-Image-mflux-6bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_IMAGE_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  run_probe "qwen-image-8bit-gen" \
    --model qwen-image-mflux-8bit --generate \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_IMAGE_STEPS" \
    --turn "$APPLE_PROMPT" --turn "$MOUNTAIN_PROMPT" --turn "$APPLE_PROMPT"

  QWEN_SOURCE_IMAGE="${VMLX_IMAGE_PROOF_SOURCE_IMAGE:-}"
  if [[ -z "$QWEN_SOURCE_IMAGE" ]]; then
    QWEN_SOURCE_IMAGE="$(
      jq -r '.generation_turns[] | select(.turn == 1 and .status == "completed") | (.image_diagnostics.path // .output)' \
        "$ART_ROOT/qwen-image-8bit-gen/qwen-image-mflux-8bit-load.json"
    )"
  fi

  if [[ ! -f "$QWEN_SOURCE_IMAGE" ]]; then
    printf 'vmlx-image-current-proof: qwen edit source image missing: %s\n' "$QWEN_SOURCE_IMAGE" >&2
    exit 1
  fi

  run_probe "qwen-edit-q4-gen" \
    --model Qwen-Image-Edit-mflux-q4 --edit --source-image "$QWEN_SOURCE_IMAGE" \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_PEAR_PROMPT"

  run_probe "qwen-edit-q5-gen" \
    --model Qwen-Image-Edit-mflux-q5 --edit --source-image "$QWEN_SOURCE_IMAGE" \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_PEAR_PROMPT"

  run_probe "qwen-edit-q6-gen" \
    --model Qwen-Image-Edit-mflux-q6 --edit --source-image "$QWEN_SOURCE_IMAGE" \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_PEAR_PROMPT"

  run_probe "qwen-edit-q8-gen" \
    --model Qwen-Image-Edit-mflux-q8 --edit --source-image "$QWEN_SOURCE_IMAGE" \
    --seed "$SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$QWEN_EDIT_STEPS" \
    --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_APPLE_PROMPT" --turn "$EDIT_PEAR_PROMPT"

  run_probe "ideogram-fp8-gen" \
    --model ideogram-4-fp8 --generate \
    --seed "$IDEOGRAM_SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$IDEOGRAM_STEPS" \
    --turn "$IDEOGRAM_APPLE_PROMPT" --turn "$IDEOGRAM_MOUNTAIN_PROMPT" --turn "$IDEOGRAM_APPLE_PROMPT"

  run_probe "ideogram-nf4-gen" \
    --model ideogram-4-nf4 --generate \
    --seed "$IDEOGRAM_SEED" --width "$WIDTH" --height "$HEIGHT" --steps "$IDEOGRAM_STEPS" \
    --turn "$IDEOGRAM_APPLE_PROMPT" --turn "$IDEOGRAM_MOUNTAIN_PROMPT" --turn "$IDEOGRAM_APPLE_PROMPT"
fi

SUMMARY="$ART_ROOT/current-proof-summary.json"
RUNS=(
  "zimage-4bit-gen|Z-Image-Turbo-mflux-4bit-load.json|generation_turns"
  "zimage-8bit-gen|Z-Image-Turbo-mflux-8bit-load.json|generation_turns"
  "flux-schnell-4bit-gen|FLUX.1-schnell-mflux-4bit-load.json|generation_turns"
  "flux-schnell-8bit-gen|FLUX.1-schnell-mflux-8bit-load.json|generation_turns"
  "qwen-image-4bit-gen|qwen-image-mflux-4bit-load.json|generation_turns"
  "qwen-image-6bit-gen|Qwen-Image-mflux-6bit-load.json|generation_turns"
  "qwen-image-8bit-gen|qwen-image-mflux-8bit-load.json|generation_turns"
  "qwen-edit-q4-gen|Qwen-Image-Edit-mflux-q4-load.json|edit_turns"
  "qwen-edit-q5-gen|Qwen-Image-Edit-mflux-q5-load.json|edit_turns"
  "qwen-edit-q6-gen|Qwen-Image-Edit-mflux-q6-load.json|edit_turns"
  "qwen-edit-q8-gen|Qwen-Image-Edit-mflux-q8-load.json|edit_turns"
  "ideogram-fp8-gen|ideogram-4-fp8-load.json|generation_turns"
  "ideogram-nf4-gen|ideogram-4-nf4-load.json|generation_turns"
)

FAILED=0
ROWS_JSON='[]'
for run in "${RUNS[@]}"; do
  IFS='|' read -r label file key <<< "$run"
  file_path="$ART_ROOT/$label/$file"
  statuses="$(jq -c --arg key "$key" '.[$key] // [] | map(.status)' "$file_path")"
  shas="$(jq -c --arg key "$key" '.[$key] // [] | map(.image_diagnostics.sha256 // null)' "$file_path")"
  outputs="$(jq -c --arg key "$key" '.[$key] // [] | map(.image_diagnostics.path // .output // null)' "$file_path")"
  load_status="$(jq -r '.load_status // "unknown"' "$file_path")"
  if [[ "$key" == "edit_turns" ]]; then
    repeat_index=1
    sensitive_index=2
  else
    repeat_index=2
    sensitive_index=1
  fi
  completed="$(jq -r 'length == 3 and all(. == "completed")' <<< "$statuses")"
  deterministic_repeat="$(jq -r --argjson i "$repeat_index" '.[0] != null and .[$i] != null and .[0] == .[$i]' <<< "$shas")"
  prompt_sensitive="$(jq -r --argjson i "$sensitive_index" '.[0] != null and .[$i] != null and .[0] != .[$i]' <<< "$shas")"
  if [[ "$completed" == "true" && "$deterministic_repeat" == "true" && "$prompt_sensitive" == "true" ]]; then
    row_status="passed"
  else
    row_status="failed"
    FAILED=1
  fi
  row_json="$(
    jq -n \
      --arg label "$label" \
      --arg artifact "$file_path" \
      --arg load_status "$load_status" \
      --arg turn_key "$key" \
      --arg status "$row_status" \
      --argjson statuses "$statuses" \
      --argjson shas "$shas" \
      --argjson outputs "$outputs" \
      --argjson repeat_index "$repeat_index" \
      --argjson sensitive_index "$sensitive_index" \
      --argjson deterministic_repeat "$deterministic_repeat" \
      --argjson prompt_sensitive "$prompt_sensitive" \
      '{
        label: $label,
        artifact: $artifact,
        load_status: $load_status,
        turn_key: $turn_key,
        statuses: $statuses,
        shas: $shas,
        outputs: $outputs,
        repeat_turns: [1, ($repeat_index + 1)],
        prompt_sensitive_turns: [1, ($sensitive_index + 1)],
        deterministic_repeat: $deterministic_repeat,
        prompt_sensitive: $prompt_sensitive,
        status: $status
      }'
  )"
  ROWS_JSON="$(jq -c --argjson row "$row_json" '. + [$row]' <<< "$ROWS_JSON")"
done

MATRIX_PATH="$ART_ROOT/status-load-matrix/compatibility-matrix.json"
MODEL_COUNT="$(jq -r '.model_count' "$MATRIX_PATH")"
LOADED_COUNT="$(jq -r '[.rows[] | select(.load_status == "loaded")] | length' "$MATRIX_PATH")"
UNLOADED="$(jq -c '[.rows[] | select(.load_status != "loaded") | .directory_name]' "$MATRIX_PATH")"
if [[ "$(jq -r 'length' <<< "$UNLOADED")" != "0" ]]; then
  FAILED=1
fi

if [[ "$FAILED" == "0" ]]; then
  SUMMARY_STATUS="passed"
else
  SUMMARY_STATUS="failed"
fi

jq -n \
  --arg status "$SUMMARY_STATUS" \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg artifact_root "$ART_ROOT" \
  --arg output_root "$OUT_ROOT" \
  --arg matrix_artifact "$MATRIX_PATH" \
  --argjson model_count "$MODEL_COUNT" \
  --argjson loaded_count "$LOADED_COUNT" \
  --argjson unloaded "$UNLOADED" \
  --argjson rows "$ROWS_JSON" \
  '{
    status: $status,
    generated_at: $generated_at,
    artifact_root: $artifact_root,
    output_root: $output_root,
    matrix: {
      artifact: $matrix_artifact,
      model_count: $model_count,
      loaded_count: $loaded_count,
      unloaded: $unloaded
    },
    rows: $rows,
    visual_gate: "View generated PNGs before claiming visual quality or Osaurus release readiness.",
    osaurus_gate: "Osaurus HTTP/UI bridge proof is outside this CLI runner and remains required before app-side readiness claims."
  }' > "$SUMMARY"

printf 'current proof summary: %s\n' "$SUMMARY"
printf 'status=%s matrix_loaded=%s/%s\n' "$SUMMARY_STATUS" "$LOADED_COUNT" "$MODEL_COUNT"
jq -r '.rows[] | "\(.label): \(.status) repeat=\(.deterministic_repeat) sensitive=\(.prompt_sensitive) sha=\(.shas | join(","))"' "$SUMMARY"
if [[ "$FAILED" != "0" ]]; then
  exit 1
fi

if [[ "$RUN_CONTRACT_CHECK" == "1" && -x "$ROOT/scripts/vmlx-image-openapi-manifest-check.sh" && -x "$(command -v node || true)" ]]; then
  "$ROOT/scripts/vmlx-image-openapi-manifest-check.sh"
elif [[ "$RUN_CONTRACT_CHECK" == "1" && ! -x "$(command -v node || true)" ]]; then
  printf 'Skipping manifest/OpenAPI contract check: node is not available on PATH.\n' >&2
fi

printf '\nProof artifacts: %s\n' "$ART_ROOT"
printf 'Proof outputs:   %s\n' "$OUT_ROOT"
printf 'Summary:         %s\n' "$SUMMARY"
printf 'PARTIAL until generated PNGs are visually inspected and Osaurus HTTP/UI bridge proof exists.\n'
