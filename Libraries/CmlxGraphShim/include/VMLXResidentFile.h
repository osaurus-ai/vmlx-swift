#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Read directly into one allocator-owned tensor. No file mapping or tensor-sized staging copy.
int vmlx_resident_file_array(void **result, int32_t fd, uint64_t offset,
                             size_t length, const int32_t *dims, int32_t rank,
                             int32_t dtype);
#ifdef __cplusplus
}
#endif
