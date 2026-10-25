// SPDX-License-Identifier: MIT
#include <sqlite3.h>

int vmlxProbeEnableSQLiteMemoryTracking(void) {
    return sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 1);
}
