#!/bin/bash
# Run one GPU job under a watchdog (test harness, 2026-10-28). Kills the job (whole process group) the moment:
#   - ~/mm31f/logs/chain3d.log contains "STEP merge", or gains any line past the baseline taken before the start;
#   - any `mm31t.` python process exists;
#   - swap use grows more than 256 MiB over the value at the start (chain_guard3 must never see growth from us).
# A fired watchdog writes ~/mm31f/swift_t5/STOP; every later call refuses to start (no more GPU work).
#   guarded.sh <log> <command...>
LOG=$1; shift
D=~/mm31f/swift_t5
CHAIN=~/mm31f/logs/chain3d.log
STOP=$D/STOP
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
if [ -e "$STOP" ]; then echo "REFUSED: $STOP exists ($(cat $STOP))" | tee -a "$LOG"; exit 3; fi
if grep -q "STEP merge" "$CHAIN" || pgrep -f "mm31t\." >/dev/null; then
  echo "pipeline active at $(date)" > "$STOP"; echo "REFUSED: pipeline active" | tee -a "$LOG"; exit 3
fi
BASE_LINES=$(wc -l < "$CHAIN"); BASE_SWAP=$(swap_mb)
echo "[guard] start $(date '+%T') chain lines $BASE_LINES swap ${BASE_SWAP}M: $*" >> "$LOG"
set -m
"$@" >> "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  WHY=""
  grep -q "STEP merge" "$CHAIN" && WHY="STEP merge in chain3d.log"
  [ -z "$WHY" ] && [ "$(wc -l < "$CHAIN")" -gt "$BASE_LINES" ] && WHY="chain3d.log advanced: $(tail -1 "$CHAIN")"
  [ -z "$WHY" ] && pgrep -f "mm31t\." >/dev/null && WHY="mm31t process: $(pgrep -af 'mm31t\.' | head -1)"
  [ -z "$WHY" ] && [ "$(swap_mb)" -gt $((BASE_SWAP + 256)) ] && WHY="swap grew to $(swap_mb)M from ${BASE_SWAP}M"
  if [ -n "$WHY" ]; then
    # process group when job control gave the child its own group; the child and its children directly in any case
    kill -TERM -- -$PID 2>/dev/null; pkill -TERM -P $PID 2>/dev/null; kill -TERM $PID 2>/dev/null; sleep 2
    kill -KILL -- -$PID 2>/dev/null; pkill -KILL -P $PID 2>/dev/null; kill -KILL $PID 2>/dev/null
    echo "$(date '+%F %T') $WHY" > "$STOP"
    echo "[guard] KILLED $(date '+%T'): $WHY" | tee -a "$LOG"
    exit 4
  fi
  sleep 1
done
wait $PID; RC=$?
echo "[guard] end $(date '+%T') rc=$RC swap $(swap_mb)M" >> "$LOG"
exit $RC
