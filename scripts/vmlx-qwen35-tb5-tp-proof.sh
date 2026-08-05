#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
ARTIFACT_DIR="${QWEN35_TP_ARTIFACT_DIR:-$ROOT/.artifacts/qwen35-tb5-tp-$(date +%Y%m%d-%H%M%S)}"
MODEL_PATH="${QWEN35_TP_MODEL:-}"
BACKEND="${QWEN35_TP_BACKEND:-ring}"
RDMA_BACKEND="${QWEN35_TP_RDMA_BACKEND:-jaccl}"
PROMPT_TOKENS="${QWEN35_TP_PROMPT_TOKEN_IDS:-151644,8948,198,2610,525,264,10950,17847,13,151645,198,151644,872,198,3838,264,1290,3984,624,151645,198,151644,77091,198}"
MAX_NEW_TOKENS="${QWEN35_TP_MAX_NEW_TOKENS:-8}"
TIMEOUT_SECONDS="${QWEN35_TP_TIMEOUT_SECONDS:-900}"
CACHE_DIR="${QWEN35_TP_CACHE_DIR:-$ARTIFACT_DIR/cache}"
SWIFTCMD="${SWIFTCMD:-swift}"

mkdir -p "$ARTIFACT_DIR" "$CACHE_DIR"

log() {
  printf '[qwen35-tb5-tp-proof] %s\n' "$*"
}

write_json() {
  local path="$1"
  shift
  python3 - "$path" "$@" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = dict(arg.split("=", 1) for arg in sys.argv[2:])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_with_timeout() {
  local seconds="$1"
  shift
  python3 - "$seconds" "$@" <<'PY'
import os, signal, subprocess, sys
timeout = int(sys.argv[1])
cmd = sys.argv[2:]
proc = subprocess.Popen(cmd)
try:
    rc = proc.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"timeout after {timeout}s: {' '.join(cmd)}", file=sys.stderr)
    sys.exit(124)
sys.exit(rc)
PY
}

build_products() {
  log "building Qwen35TPProofRunner and DistributedPeerSmoke"
  "$SWIFTCMD" build --package-path "$ROOT" --product Qwen35TPProofRunner --product DistributedPeerSmoke \
    2>&1 | tee "$ARTIFACT_DIR/build.log"
}

prepare_mlx_metal() {
  log "preparing MLX metallib for command-line kernel smoke"
  if "$ROOT/scripts/prepare-mlx-metal.sh" > "$ARTIFACT_DIR/metal-prep.log" 2> "$ARTIFACT_DIR/metal-prep.err"; then
    return 0
  fi
  log "MLX metallib preparation failed; kernel/model rows will be blocked"
  return 1
}

run_peer_smoke() {
  log "running encrypted loopback peer smoke"
  "$SWIFTCMD" run --package-path "$ROOT" DistributedPeerSmoke --self-test --interface loopback \
    > "$ARTIFACT_DIR/peer-smoke.json" 2> "$ARTIFACT_DIR/peer-smoke.stderr"
}

run_collective_smoke() {
  log "running single-rank collective/kernel smoke"
  run_with_timeout "$TIMEOUT_SECONDS" env \
    MLX_RANK=0 \
    MLX_WORLD_SIZE=1 \
    MLX_DIST_BACKEND="$BACKEND" \
    TP_STRICT=0 \
    TP_SMOKE=1 \
    TP_OUTPUT_PATH="$ARTIFACT_DIR/collective-smoke.json" \
    "$SWIFTCMD" run --package-path "$ROOT" Qwen35TPProofRunner \
    > "$ARTIFACT_DIR/collective-smoke.log" 2>&1
}

run_model_baseline() {
  log "running Qwen 3.5 baseline load/decode with prefix + disk L2 + TurboQuant KV"
  run_with_timeout "$TIMEOUT_SECONDS" env \
    MLX_RANK=0 \
    MLX_WORLD_SIZE=1 \
    MLX_DIST_BACKEND="$BACKEND" \
    TP_MODEL_PATH="$MODEL_PATH" \
    TP_OUTPUT_PATH="$ARTIFACT_DIR/baseline-cold.json" \
    TP_SHARDING_PLAN=qwen35 \
    TP_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    TP_PROMPT_TOKEN_IDS="$PROMPT_TOKENS" \
    TP_ENABLE_CACHE_COORDINATOR=1 \
    TP_PREFIX_CACHE=1 \
    TP_L2_DISK_CACHE=1 \
    TP_CACHE_DIR="$CACHE_DIR/baseline" \
    TP_KV_MODE=turboquant \
    TP_TEMPERATURE=0 \
    "$SWIFTCMD" run --package-path "$ROOT" Qwen35TPProofRunner \
    > "$ARTIFACT_DIR/baseline-cold.log" 2>&1

  log "running Qwen 3.5 warm replay for prefix/L2 hit evidence"
  run_with_timeout "$TIMEOUT_SECONDS" env \
    MLX_RANK=0 \
    MLX_WORLD_SIZE=1 \
    MLX_DIST_BACKEND="$BACKEND" \
    TP_MODEL_PATH="$MODEL_PATH" \
    TP_OUTPUT_PATH="$ARTIFACT_DIR/baseline-warm.json" \
    TP_SHARDING_PLAN=qwen35 \
    TP_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    TP_PROMPT_TOKEN_IDS="$PROMPT_TOKENS" \
    TP_ENABLE_CACHE_COORDINATOR=1 \
    TP_PREFIX_CACHE=1 \
    TP_L2_DISK_CACHE=1 \
    TP_CACHE_DIR="$CACHE_DIR/baseline" \
    TP_KV_MODE=turboquant \
    TP_TEMPERATURE=0 \
    "$SWIFTCMD" run --package-path "$ROOT" Qwen35TPProofRunner \
    > "$ARTIFACT_DIR/baseline-warm.log" 2>&1
}

validate_outputs() {
  log "validating JSON outputs and cache evidence"
  python3 - "$ARTIFACT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = {
    "artifact_dir": str(root),
    "status": "PARTIAL",
    "proof_rows": [],
    "blocked": [],
}

peer = root / "peer-smoke.json"
if peer.exists():
    try:
        peer_json = json.loads(peer.read_text())
        summary["proof_rows"].append({
            "name": "encrypted_peer_smoke",
            "ok": bool(peer_json.get("ok")),
            "rdma_ready": peer_json.get("route", {}).get("rdmaReady"),
            "rdma_blocked_reason": peer_json.get("route", {}).get("rdmaBlockedReason"),
        })
    except Exception as exc:
        summary["blocked"].append(f"peer smoke JSON parse failed: {exc}")
else:
    summary["blocked"].append("peer smoke did not produce JSON")

for name in ["baseline-cold", "baseline-warm"]:
    path = root / f"{name}.json"
    if not path.exists():
        summary["blocked"].append(f"{name} output missing")
        continue
    data = json.loads(path.read_text())
    disk = data.get("cache_stats", {}).get("disk", {})
    ssm = data.get("cache_stats", {}).get("ssm", {})
    engine = data.get("engine", {})
    summary["proof_rows"].append({
        "name": name,
        "model_type": data.get("model_type"),
        "generated_token_count": len(data.get("generated_tokens", [])),
        "decoded_preview": data.get("decoded", "")[:160],
        "tokens_per_second": data.get("tokens_per_second"),
        "kv_mode": data.get("kv_mode"),
        "disk_hits": disk.get("hit_count"),
        "disk_misses": disk.get("miss_count"),
        "disk_stores": disk.get("stores"),
        "ssm_hits": ssm.get("hit_count"),
        "ssm_re_derives": ssm.get("re_derives"),
        "turboquant_compressions": engine.get("turboquant_compressions"),
    })
    if not data.get("generated_tokens"):
        summary["blocked"].append(f"{name} generated no tokens")

warm = root / "baseline-warm.json"
if warm.exists():
    warm_data = json.loads(warm.read_text())
    disk_hits = warm_data.get("cache_stats", {}).get("disk", {}).get("hit_count", 0) or 0
    if disk_hits <= 0:
        summary["blocked"].append("warm replay did not report disk L2 hits")

summary["qwen35_tp_boundary"] = (
    "Qwen 3.5 normal attention/MoE projections have a TP plan. "
    "GatedDelta/SSM companion layers are replicated until recurrent-state "
    "and companion-cache parity is proven."
)
summary["rdma_boundary"] = (
    "Loopback/ring proof exercises the Swift distributed path and MLX "
    "collective kernel. Real TB5 RDMA proof requires two Macs with allowed "
    "Thunderbolt data-plane addresses and QWEN35_TP_BACKEND=jaccl."
)
if not summary["blocked"] and len(summary["proof_rows"]) >= 3:
    summary["status"] = "FIXED_FOR_SINGLE_HOST_SIMULATED_TP"
summary_path = root / "SUMMARY.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(summary_path)
if summary["blocked"]:
    for item in summary["blocked"]:
        print(f"BLOCKED: {item}", file=sys.stderr)
    sys.exit(20)
PY
}

main() {
  build_products
  run_peer_smoke
  metal_status="ok"
  if ! prepare_mlx_metal; then
    metal_status="blocked"
  fi
  collective_status="skipped"
  if [[ "$metal_status" == "ok" ]]; then
    if run_collective_smoke; then
      collective_status="ok"
    else
      collective_status="blocked"
    fi
  fi

  if [[ -z "$MODEL_PATH" ]]; then
    write_json "$ARTIFACT_DIR/SUMMARY.json" \
      artifact_dir="$ARTIFACT_DIR" \
      status="PARTIAL_NO_MODEL" \
      metal_status="$metal_status" \
      collective_status="$collective_status" \
      blocked="set QWEN35_TP_MODEL to run model load/decode/cache proof" \
      rdma_backend="$RDMA_BACKEND" \
      rdma_boundary="set QWEN35_TP_BACKEND=jaccl and use real Thunderbolt data-plane addresses for TB5 RDMA proof"
    log "no QWEN35_TP_MODEL set; wrote partial summary to $ARTIFACT_DIR/SUMMARY.json"
    return 0
  fi

  if [[ "$metal_status" != "ok" || "$collective_status" != "ok" ]]; then
    write_json "$ARTIFACT_DIR/SUMMARY.json" \
      artifact_dir="$ARTIFACT_DIR" \
      status="BLOCKED_METAL_OR_COLLECTIVE_SMOKE" \
      metal_status="$metal_status" \
      collective_status="$collective_status" \
      blocked="kernel smoke must pass before model load/decode/cache proof" \
      rdma_backend="$RDMA_BACKEND"
    return 20
  fi

  run_model_baseline
  validate_outputs
  log "summary: $ARTIFACT_DIR/SUMMARY.json"
}

main "$@"
