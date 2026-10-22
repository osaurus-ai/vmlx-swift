// Lightweight graph inspection bridge for runtime dispatch and RunBench.

#ifndef VMLX_CMLX_GRAPH_SHIM_H
#define VMLX_CMLX_GRAPH_SHIM_H

#include <stdint.h>
#include "VMLXMappedFile.h"
#include "VMLXResidentFile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* VMLXGraphArray;

// Returns 1 for traced/retained graphs (or an invalid handle), 0 otherwise.
// This is metadata only: it never evaluates, copies or synchronizes the array.
int vmlx_graph_array_is_tracer(VMLXGraphArray array);

int vmlx_graph_stats(VMLXGraphArray array, int32_t* node_count, int32_t* astype_count);

#ifdef __cplusplus
}
#endif

#endif
