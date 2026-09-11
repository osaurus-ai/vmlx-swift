// Non-Apple stub: the Neural Engine bridge reports unavailable and every
// create fails by name. Keeps the package building on Linux.
#include "vmlx_ane_bridge.h"
#include <stdio.h>

int vmlx_ane_available(void) { return 0; }
vmlx_ane_plane *vmlx_ane_plane_create(size_t bytes) { (void)bytes; return NULL; }
void vmlx_ane_plane_free(vmlx_ane_plane *plane) { (void)plane; }
void *vmlx_ane_plane_base(vmlx_ane_plane *plane) { (void)plane; return NULL; }
size_t vmlx_ane_plane_bytes(const vmlx_ane_plane *plane) { (void)plane; return 0; }
vmlx_ane_model *vmlx_ane_model_create(
    const char *name, const char *mil_text, const void *weights, size_t weight_bytes,
    vmlx_ane_plane *const *inputs, uint32_t input_count,
    vmlx_ane_plane *const *outputs, uint32_t output_count,
    uint32_t procedure_count, const char *cache_dir, char *error, size_t error_size)
{
    (void)mil_text; (void)weights; (void)weight_bytes; (void)inputs; (void)input_count;
    (void)outputs; (void)output_count; (void)procedure_count; (void)cache_dir;
    if (error && error_size) snprintf(error, error_size, "ANE %s: no Neural Engine on this platform", name);
    return NULL;
}
void vmlx_ane_model_free(vmlx_ane_model *model) { (void)model; }
int vmlx_ane_model_eval(vmlx_ane_model *model, uint32_t procedure, char *error, size_t error_size) {
    (void)model; (void)procedure;
    if (error && error_size) snprintf(error, error_size, "no Neural Engine on this platform");
    return 0;
}
double vmlx_ane_model_compile_seconds(const vmlx_ane_model *model) { (void)model; return 0; }
bool vmlx_ane_model_cache_hit(const vmlx_ane_model *model) { (void)model; return false; }
