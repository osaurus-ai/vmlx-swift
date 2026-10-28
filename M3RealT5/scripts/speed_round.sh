#!/bin/bash
# speed_round.sh <round tag> <bundle tag> <binaries...>: one TokenIterator speed process per binary, in order.
R=$1; T=$2; shift 2
cd ~/mm31f/swift_t5
for b in "$@"; do
  P=dense,auto; [ "$b" = base ] && P=dense
  ./guarded.sh logs/speed_${R}_${T}_$b.log env T5_BUNDLE=$HOME/mm31f/test_bundles/t5_$T T5_OUT=out T5_TAG=${T}_${R}_$b \
      T5_TOKENS=tokens.safetensors T5_PATHS=$P T5_PROBES=${T5_PROBES:-3} ./bin_$b/M3RealT5 speed || exit 1
done
for b in "$@"; do
  echo "== $R $T $b"
  python3 - "out/speed_${T}_${R}_$b.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for n in ("8269", "32845"):
    if n not in d: continue
    for p, v in d[n].items():
        print(n, p, "pp/s", round(v["pp_s"]["median"], 1), [round(x) for x in v["pp_s"]["runs"]],
              "tok/s", round(v["tok_s"]["median"], 1), [round(x, 1) for x in v["tok_s"]["runs"]], v["first16"][:6])
PY
done
