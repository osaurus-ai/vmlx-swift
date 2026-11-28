// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstdio>

#include "hwy/targets.h"

// Prints the Highway targets this CPU supports, by Highway's own detection;
// scripts/run-cpu-kernel-tests.sh derives the targets a host must execute from it.
int main() {
  const int64_t supported = hwy::SupportedTargets();
  for (int i = 0; i < 63; ++i) {
    const int64_t bit = int64_t{1} << i;
    if (supported & bit) {
      std::printf("%s ", hwy::TargetName(bit));
    }
  }
  std::printf("\n");
  return 0;
}
