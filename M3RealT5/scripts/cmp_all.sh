#!/bin/bash
# cmp_all.sh <tag>: every comparison pair for one bundle tag (CPU only, numpy).
T=$1; cd ~/mm31f/swift_t5
P=~/mlx/vmlx-release-1680-final-20261018/.venv/bin/python
for pair in "py_${T}_dense swift_${T}_dense" "py_${T}_selref swift_${T}_auto" "py_${T}_auto swift_${T}_auto" \
            "swift_${T}_dense swift_${T}_auto" "py_${T}_dense py_${T}_selref" "py_${T}_selref py_${T}_auto" \
            "py_${T}_perturb swift_${T}_dense" "py_${T}_perturb py_${T}_selref" "py_${T}_shift swift_${T}_dense"; do
  set -- $pair
  [ -f out/$1.safetensors ] && [ -f out/$2.safetensors ] || continue
  $P -I compare.py out/$1.safetensors out/$2.safetensors --name "$1~$2" --json out/cmp_$1~$2.json
done
