#ifndef VMLX_ANE_BRIDGE_H
#define VMLX_ANE_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The only door to the private AppleNeuralEngine framework.
 *
 * Everything above this header speaks MIL text, a weight blob, and byte
 * planes; everything below it is objc_msgSend into _ANEInMemoryModel and
 * friends. Nothing here touches MLX: the caller packs fp16 planes and reads
 * fp16 planes, and a compiled program is a static graph with fixed shapes.
 *
 * Contract notes (all measured, see SpecDec/ANE-MTP-DRAFTER.md):
 * - Request inputs bind to MIL function parameters in ALPHABETICAL name
 *   order, not declaration order. Name parameters so the sort matches the
 *   plane order passed here (a_..., b_..., ...). Same-shape mismatches swap
 *   data silently.
 * - fp16 planes: the innermost (width) dim times 2 bytes must sit on the
 *   64-byte grid, i.e. width must be a multiple of 32; other widths compile
 *   and then fail every eval with a bare "Program Inference error".
 * - Compile/load runs in aned, which cannot read $HOME; staging lives in
 *   $TMPDIR at exactly $TMPDIR/<identifier>. Compiled artifacts stay for the
 *   life of the handle (aned demand-reads them while serving).
 * - Evals on one handle are strictly serial; callers own that serialization.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vmlx_ane_model vmlx_ane_model;
typedef struct vmlx_ane_plane vmlx_ane_plane;

/* 1 when the private framework is present and its classes resolve. */
int vmlx_ane_available(void);

/* A host-mapped, 16 KB-aligned IOSurface-backed plane of at least `bytes`.
 * Zero-filled. */
vmlx_ane_plane *vmlx_ane_plane_create(size_t bytes);
void vmlx_ane_plane_free(vmlx_ane_plane *plane);
void *vmlx_ane_plane_base(vmlx_ane_plane *plane);
size_t vmlx_ane_plane_bytes(const vmlx_ane_plane *plane);

/* Compile (or restore from the compile cache) and load one MIL program.
 * `weights` is copied. Inputs bind to indices 0..input_count-1, outputs to
 * 0..output_count-1. `procedure_count` = number of `procedureNNN` functions
 * (1 for a plain `main`). On failure returns NULL and fills `error`.
 *
 * `cache_dir`: NULL disables the persistent compile cache; otherwise
 * compiled artifacts are kept at <cache_dir>/<content-hash> and restored on
 * the next create with the same MIL + weights. */
vmlx_ane_model *vmlx_ane_model_create(
    const char *name,
    const char *mil_text,
    const void *weights, size_t weight_bytes,
    vmlx_ane_plane *const *inputs, uint32_t input_count,
    vmlx_ane_plane *const *outputs, uint32_t output_count,
    uint32_t procedure_count,
    const char *cache_dir,
    char *error, size_t error_size);
void vmlx_ane_model_free(vmlx_ane_model *model);

/* Run one procedure synchronously. 1 on success, 0 with `error` filled. */
int vmlx_ane_model_eval(vmlx_ane_model *model, uint32_t procedure,
                        char *error, size_t error_size);

double vmlx_ane_model_compile_seconds(const vmlx_ane_model *model);
bool vmlx_ane_model_cache_hit(const vmlx_ane_model *model);

#ifdef __cplusplus
}
#endif

#endif
