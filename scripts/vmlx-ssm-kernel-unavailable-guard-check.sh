#!/usr/bin/env bash
set -euo pipefail

target="Libraries/MLXLLM/Models/SSM.swift"

if rg -n 'fatalError\("SSM kernel not available"\)' "$target" >/dev/null; then
  echo "FAIL: missing SSM custom-kernel fallback still process-aborts"
  exit 1
fi

if ! perl -0ne 'exit(/guard let kernel = SSMKernelManager\.shared\.ssmKernel else \{\s*return ssmAttn\(/s ? 0 : 1)' "$target"; then
  echo "FAIL: SSM kernel-unavailable path must fall back to reference ssmAttn"
  exit 1
fi

echo "PASS: SSM kernel-unavailable path fails over to reference SSM attention."
