// ANE decode-shape probe: compiles one MIL program on the Apple Neural Engine
// through the private AppleNeuralEngine framework and measures the latency of
// an int8-weight 1x1 conv (== a Linear) at decode-shaped tiles.
//
// This is the make-or-break measurement for running an MTP draft head on the
// ANE: drafting is D sequential 1-token forwards, so what matters is the
// per-eval floor and the effective weight bandwidth at rows <= 32, not the
// prefill-shaped throughput mlx-serve measured at rows >= 256.
//
// Build:
//   clang -fobjc-arc -O2 -framework Foundation -framework IOSurface -framework Metal -framework MetalPerformanceShaders \
//         tools/ane-draft-probe/aneprobe.m -o /tmp/aneprobe
// Run:
//   /tmp/aneprobe --hidden 5120 --out 5120 --rows 32 --iters 50
//   /tmp/aneprobe --layout batch --rows 1          # tokens in the batch dim
//   /tmp/aneprobe --hidden 5120 --out 17408 --rows 32 --chunks 4
//   /tmp/aneprobe --gpu-load          # ANE evals while a Metal GEMM loop runs
//   /tmp/aneprobe --gpu-only          # the GEMM loop alone (control)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

// ── blob (64-byte header + 64-byte aligned chunks; mirrors coremltools) ──
typedef struct { uint8_t *data; size_t cursor; size_t cap; } blob_t;

static void blob_reserve(blob_t *b, size_t extra) {
    size_t need = b->cursor + extra;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap : 4096;
    while (cap < need) cap *= 2;
    b->data = realloc(b->data, cap);
    memset(b->data + b->cap, 0, cap - b->cap);
    b->cap = cap;
}

// MILBlob BlobDataType: Float16=1 Float32=2 UInt8=3 Int8=4 ... Int4=8 UInt4=11.
enum { BLOB_FP16 = 1, BLOB_FP32 = 2, BLOB_UINT8 = 3, BLOB_INT8 = 4, BLOB_INT4 = 8, BLOB_UINT4 = 11 };
static uint32_t g_blob_dtype = BLOB_FP16;
static size_t blob_add(blob_t *b, const void *payload, size_t bytes) {
    size_t padded = 64 + ((bytes + 63) & ~(size_t)63);
    blob_reserve(b, padded);
    size_t header = b->cursor;
    uint8_t *chunk = b->data + header;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    memcpy(chunk + 4, &g_blob_dtype, 4);
    uint32_t size32 = (uint32_t)bytes;
    uint32_t offset32 = (uint32_t)(header + 64);
    memcpy(chunk + 8, &size32, sizeof(size32));
    memcpy(chunk + 16, &offset32, sizeof(offset32));
    if (payload) memcpy(chunk + 64, payload, bytes);
    b->cursor = header + padded;
    return header;
}

static blob_t blob_new(void) {
    blob_t b = {0};
    blob_reserve(&b, 4096);
    b.cursor = 64;
    b.data[0] = 0x01;
    b.data[4] = 0x02;
    return b;
}

// ── MIL text ──
static NSString *mil_header(void) {
    return @"program(1.3)\n[buildInfo = dict<string, string>({{"
            "\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
            "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n{\n";
}

static void emit_int8_weight(NSMutableString *t, const char *name, uint32_t n,
                             uint32_t k, size_t q_off, size_t s_off) {
    [t appendFormat:@"        tensor<fp16, [%u,%u,1,1]> %s = "
        "constexpr_affine_dequantize()[axis=int32(0), name=string(\"%s\"), "
        "quantized_data=tensor<int8, [%u,%u,1,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), "
        "offset=uint64(%zu))), scale=tensor<fp16, [%u]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), "
        "offset=uint64(%zu))), zero_point=int8(0)];\n",
        n, k, name, name, n, k, q_off, n, s_off];
}

typedef enum { LAYOUT_WIDTH = 0, LAYOUT_BATCH = 1 } layout_t;
typedef enum { W_INT8 = 0, W_LUT4 = 1, W_BLOCK4 = 2 } wmode_t;
static uint32_t g_group = 64;  // block size for W_BLOCK4 (JANG affine gs64)

// iOS18 constexpr_lut_to_dense: 4-bit indices [n,k,1,1] + one fp16 LUT of 16
// entries shaped [1,1,1,1,16,1] (rank+2: per-group dims, NUM_PALETTES, VECTOR).
static void emit_lut4_weight(NSMutableString *t, const char *name, uint32_t n,
                             uint32_t k, size_t idx_off, size_t lut_off) {
    [t appendFormat:@"        tensor<fp16, [%u,%u,1,1]> %s = "
        "constexpr_lut_to_dense(indices=tensor<uint4, [%u,%u,1,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), "
        "lut=tensor<fp16, [1,1,1,1,16,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))))"
        "[name=string(\"%s\")];\n",
        n, k, name, n, k, idx_off, lut_off, name];
}

// iOS18 constexpr_blockwise_shift_scale: uint4 data [n,k,1,1], fp16 scale
// [n, k/group, 1, 1], uint4 offset (zero point) same shape as scale.
static void emit_block4_weight(NSMutableString *t, const char *name, uint32_t n,
                               uint32_t k, size_t data_off, size_t scale_off,
                               size_t zp_off) {
    [t appendFormat:@"        tensor<fp16, [%u,%u,1,1]> %s = "
        "constexpr_blockwise_shift_scale(data=tensor<uint4, [%u,%u,1,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), "
        "offset=tensor<uint4, [%u,%u,1,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))), "
        "scale=tensor<fp16, [%u,%u,1,1]>(BLOBFILE("
        "path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%zu))))"
        "[name=string(\"%s\")];\n",
        n, k, name, n, k, data_off, n, k / g_group, zp_off, n, k / g_group, scale_off, name];
}

// x: [B, hidden, 1, W]; W = rows (width layout) or B = rows (batch layout).
static NSString *shape4(layout_t layout, uint32_t rows, uint32_t ch) {
    if (layout == LAYOUT_BATCH)
        return [NSString stringWithFormat:@"[%u, %u, 1, 1]", rows, ch];
    return [NSString stringWithFormat:@"[1, %u, 1, %u]", ch, rows];
}

// Emit a K-chunked linear: y = W x with W [out, hidden] split into `chunks`
// K-slabs (a single K > ~4608 conv is a measured 2.6x cliff on the ANE per
// mlx-serve; chunk to stay under it).
static NSString *build_mil(layout_t layout, wmode_t wmode, uint32_t hidden, uint32_t out,
                           uint32_t rows, uint32_t chunks, const size_t *q_offs,
                           size_t s_off, const size_t *zp_offs) {
    NSMutableString *t = [NSMutableString stringWithString:mil_header()];
    NSString *xs = shape4(layout, rows, hidden);
    NSString *ys = shape4(layout, rows, out);
    uint32_t kc = hidden / chunks;
    [t appendFormat:@"    func main<ios18>(tensor<fp16, %@> x0) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n", xs];
    for (uint32_t c = 0; c < chunks; c++) {
        NSString *xin = @"x0";
        if (chunks > 1) {
            [t appendFormat:
                @"        tensor<int32, [4]> b%u = const()[name=string(\"b%u\"), val=tensor<int32, [4]>([0,%u,0,0])];\n"
                "        tensor<int32, [4]> z%u = const()[name=string(\"z%u\"), val=tensor<int32, [4]>(%@)];\n"
                "        tensor<fp16, %@> a%u = slice_by_size(x=x0, begin=b%u, size=z%u)[name=string(\"a%u\")];\n",
                c, c, c * kc, c, c,
                (layout == LAYOUT_BATCH
                    ? [NSString stringWithFormat:@"[%u,%u,1,1]", rows, kc]
                    : [NSString stringWithFormat:@"[1,%u,1,%u]", kc, rows]),
                shape4(layout, rows, kc), c, c, c, c];
            xin = [NSString stringWithFormat:@"a%u", c];
        }
        char wname[16];
        snprintf(wname, sizeof wname, "W%u", c);
        if (wmode == W_INT8) emit_int8_weight(t, wname, out, kc, q_offs[c], s_off);
        else if (wmode == W_LUT4) emit_lut4_weight(t, wname, out, kc, q_offs[c], s_off);
        else emit_block4_weight(t, wname, out, kc, q_offs[c], s_off + c * 0, zp_offs[c]);
        [t appendFormat:
            @"        tensor<fp16, %@> p%u = conv(dilations=dl, groups=gr, pad=pd, "
            "pad_type=pt, strides=st, weight=W%u, x=%@)[name=string(\"p%u\")];\n",
            ys, c, c, xin, c];
    }
    NSString *prev = @"p0";
    for (uint32_t c = 1; c < chunks; c++) {
        NSString *sum = [NSString stringWithFormat:@"s%u", c];
        [t appendFormat:@"        tensor<fp16, %@> %@ = add(x=%@, y=p%u)[name=string(\"%@\")];\n",
            ys, sum, prev, c, sum];
        prev = sum;
    }
    [t appendFormat:@"    } -> (%@);\n}\n", prev];
    return t;
}

static IOSurfaceRef make_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

// GPU load generator: back-to-back fp16 GEMMs on Metal (MPS) from a second
// thread. Reports its own achieved TFLOPS so contention shows on BOTH sides.
static volatile int gpu_stop = 0;
static double gpu_seconds = 0;
static uint64_t gpu_gemms = 0;
static uint32_t gpu_n = 2048;
static int gpu_blit = 0;       // 1 = bandwidth-bound blit copies instead of GEMMs
static double gpu_blit_bytes = 0;
static void *gpu_load_thread(void *arg) {
    (void)arg;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        uint32_t n = gpu_n;
        MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n * 2 dataType:MPSDataTypeFloat16];
        id<MTLBuffer> a = [dev newBufferWithLength:(NSUInteger)n * n * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [dev newBufferWithLength:(NSUInteger)n * n * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> c = [dev newBufferWithLength:(NSUInteger)n * n * 2 options:MTLResourceStorageModeShared];
        MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:a descriptor:d];
        MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:b descriptor:d];
        MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:c descriptor:d];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:n resultColumns:n interiorColumns:n alpha:1 beta:0];
        id<MTLBuffer> big_a = nil, big_b = nil;
        const NSUInteger big = 1u << 30;  // 1 GiB per side: far past any cache
        if (gpu_blit) {
            big_a = [dev newBufferWithLength:big options:MTLResourceStorageModePrivate];
            big_b = [dev newBufferWithLength:big options:MTLResourceStorageModePrivate];
        }
        double t0 = now_s();
        while (!gpu_stop) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            if (gpu_blit) {
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                for (int i = 0; i < 4; i++)
                    [blit copyFromBuffer:(i & 1) ? big_b : big_a sourceOffset:0
                                toBuffer:(i & 1) ? big_a : big_b destinationOffset:0 size:big];
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                gpu_blit_bytes += 4.0 * 2.0 * (double)big;  // read + write per copy
                continue;
            }
            for (int i = 0; i < 8; i++) [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];
            [cb commit];
            [cb waitUntilCompleted];
            gpu_gemms += 8;
        }
        gpu_seconds = now_s() - t0;
    }
    return NULL;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

// Generic runner: load a canonical model.mil + weights/weight.bin (e.g. from
// a coremlc-compiled .mlmodelc built by tools/ane-draft-probe/mtp_head_mil.py) and
// time evals with N input surfaces and one output. Inputs bind to MIL
// function parameters in ALPHABETICAL name order (mlx-serve's finding), so
// the byte sizes here must be listed in that order.
static int run_generic(const char *mil_path, const char *weights_path,
                       const char *in_spec, size_t out_bytes, uint32_t iters,
                       int gpu_load) {
    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        NSData *program = [NSData dataWithContentsOfFile:@(mil_path)];
        NSData *weights = [NSData dataWithContentsOfFile:@(weights_path)];
        if (!program || !weights) { fprintf(stderr, "cannot read mil/weights\n"); return 1; }
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) { fprintf(stderr, "descriptor rejected\n"); return 1; }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) { fprintf(stderr, "model rejected\n"); return 1; }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[directory stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [program writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weights writeToFile:[directory stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        NSError *err = nil;
        double t0 = now_s();
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &err)) {
            fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String ?: "?");
            [fm removeItemAtPath:directory error:nil];
            return 1;
        }
        double t_compile = now_s() - t0;
        t0 = now_s();
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &err)) {
            fprintf(stderr, "load failed: %s\n", err.localizedDescription.UTF8String ?: "?");
            [fm removeItemAtPath:directory error:nil];
            return 1;
        }
        double t_load = now_s() - t0;

        NSMutableArray *inputs = [NSMutableArray array], *in_idx = [NSMutableArray array];
        NSMutableArray<NSValue *> *surfaces = [NSMutableArray array];
        char *spec = strdup(in_spec);
        uint32_t idx = 0;
        for (char *tok = strtok(spec, ","); tok; tok = strtok(NULL, ",")) {
            size_t bytes = (size_t)strtoull(tok, NULL, 10);
            IOSurfaceRef surf = make_surface(bytes);
            uint8_t *base = IOSurfaceGetBaseAddress(surf);
            for (size_t i = 0; i < bytes; i++) base[i] = (uint8_t)(rand() & 0x3f);  // small fp16 values
            [inputs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), surf)];
            [in_idx addObject:@(idx++)];
            [surfaces addObject:[NSValue valueWithPointer:surf]];
        }
        free(spec);
        IOSurfaceRef out_surf = make_surface(out_bytes);
        memset(IOSurfaceGetBaseAddress(out_surf), 0, out_bytes);
        id out_obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), out_surf);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inputs, in_idx, @[out_obj], @[@0], nil, nil, @0);
        if (!request) { fprintf(stderr, "request rejected\n"); return 1; }

        pthread_t gpu_thread;
        if (gpu_load) { pthread_create(&gpu_thread, NULL, gpu_load_thread, NULL); usleep(300000); }
        double *samples = malloc(sizeof(double) * iters);
        uint32_t ok = 0;
        for (uint32_t i = 0; i < iters + 3; i++) {
            double s = now_s();
            BOOL r = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &err);
            double e = now_s() - s;
            if (!r) {
                fprintf(stderr, "eval %u failed: %s\n", i, err.localizedDescription.UTF8String ?: "?");
                [fm removeItemAtPath:directory error:nil];
                return 1;
            }
            if (i >= 3) samples[ok++] = e;
        }
        if (gpu_load) {
            gpu_stop = 1; pthread_join(gpu_thread, NULL);
            if (gpu_blit) printf("  GPU load: blit %.1f GB/s (with ANE evals)\n", gpu_blit_bytes / gpu_seconds / 1e9);
            else printf("  GPU load: %.1f TFLOPS (with ANE evals)\n", 2.0 * (double)gpu_n * gpu_n * gpu_n * (double)gpu_gemms / gpu_seconds / 1e12);
        }
        qsort(samples, ok, sizeof(double), cmp_double);
        printf("ANE generic: %s (weights %.1f MB)\n", mil_path, (double)weights.length / 1e6);
        printf("  compile %.2fs load %.3fs\n", t_compile, t_load);
        printf("  eval: median %.3f ms  p10 %.3f  p90 %.3f  (n=%u)\n",
               samples[ok / 2] * 1e3, samples[ok / 10] * 1e3, samples[(ok * 9) / 10] * 1e3, ok);
        uint16_t *ob = IOSurfaceGetBaseAddress(out_surf);
        printf("  output[0..4] raw u16: %04x %04x %04x %04x\n", ob[0], ob[1], ob[2], ob[3]);
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &err);
        [fm removeItemAtPath:directory error:nil];
        for (NSValue *v in surfaces) CFRelease((IOSurfaceRef)v.pointerValue);
        CFRelease(out_surf);
        free(samples);
    }
    return 0;
}

int main(int argc, char **argv) {
    uint32_t hidden = 5120, out = 5120, rows = 32, iters = 50, chunks = 1;
    layout_t layout = LAYOUT_WIDTH;
    int quiet = 0, gpu_load = 0, gpu_only = 0;
    wmode_t wmode = W_INT8;
    const char *mil_path = NULL, *weights_path = NULL, *in_spec = "0";
    size_t out_bytes = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hidden")) hidden = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")) out = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows")) rows = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunks")) chunks = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layout")) layout = !strcmp(argv[++i], "batch") ? LAYOUT_BATCH : LAYOUT_WIDTH;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--gpu-load")) gpu_load = 1;
        else if (!strcmp(argv[i], "--gpu-only")) { gpu_load = 1; gpu_only = 1; }
        else if (!strcmp(argv[i], "--gpu-n")) gpu_n = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-blit")) gpu_blit = 1;
        else if (!strcmp(argv[i], "--wmode")) { const char *m = argv[++i]; wmode = !strcmp(m, "lut4") ? W_LUT4 : !strcmp(m, "block4") ? W_BLOCK4 : W_INT8; }
        else if (!strcmp(argv[i], "--group")) g_group = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mil")) mil_path = argv[++i];
        else if (!strcmp(argv[i], "--weights")) weights_path = argv[++i];
        else if (!strcmp(argv[i], "--in")) in_spec = argv[++i];
        else if (!strcmp(argv[i], "--out-bytes")) out_bytes = (size_t)strtoull(argv[++i], NULL, 10);
    }
    if (mil_path) return run_generic(mil_path, weights_path, in_spec, out_bytes, iters, gpu_load);
    if (hidden % chunks) { fprintf(stderr, "hidden %% chunks != 0\n"); return 2; }

    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!descriptorClass || !modelClass || !requestClass || !surfaceClass) {
            fprintf(stderr, "private ANE classes missing\n");
            return 1;
        }

        // Random int8 weights [out, hidden] with per-row fp16 scales; random input.
        srand(1234);
        int8_t *wq = malloc((size_t)out * hidden);
        float *ws = malloc(sizeof(float) * out);
        for (size_t i = 0; i < (size_t)out * hidden; i++) wq[i] = (int8_t)((rand() % 255) - 127);
        for (uint32_t i = 0; i < out; i++) ws[i] = 1.0f / (127.0f * sqrtf((float)hidden));
        __fp16 *x = malloc(sizeof(__fp16) * (size_t)hidden * rows);
        for (size_t i = 0; i < (size_t)hidden * rows; i++) x[i] = (__fp16)(((rand() % 2001) - 1000) / 1000.0f);

        blob_t blob = blob_new();
        uint32_t kc = hidden / chunks;
        size_t q_offs[64], zp_offs[64];
        size_t s_off = 0;
        // Reference dequantized weight (fp32) so parity is checked against
        // what the ANE was actually given, whatever the packing.
        float *wref = malloc(sizeof(float) * (size_t)out * hidden);
        if (wmode == W_INT8) {
            int8_t *slab = malloc((size_t)out * kc);
            g_blob_dtype = BLOB_INT8;
            for (uint32_t c = 0; c < chunks; c++) {
                for (uint32_t n = 0; n < out; n++)
                    memcpy(slab + (size_t)n * kc, wq + (size_t)n * hidden + (size_t)c * kc, kc);
                q_offs[c] = blob_add(&blob, slab, (size_t)out * kc);
            }
            free(slab);
            g_blob_dtype = BLOB_FP16;
            s_off = blob_add(&blob, NULL, (size_t)out * 2);
            __fp16 *dst = (__fp16 *)(void *)(blob.data + s_off + 64);
            for (uint32_t i = 0; i < out; i++) dst[i] = (__fp16)ws[i];
            for (size_t n = 0; n < out; n++)
                for (size_t k = 0; k < hidden; k++)
                    wref[n * hidden + k] = (float)wq[n * hidden + k] * (float)(__fp16)ws[n];
        } else if (wmode == W_LUT4) {
            // 4-bit indices packed two per byte (low nibble first), one LUT.
            __fp16 lut[16];
            for (int i = 0; i < 16; i++) lut[i] = (__fp16)((i - 7.5f) / (7.5f * sqrtf((float)hidden)));
            uint8_t *slab = malloc((size_t)out * kc / 2);
            g_blob_dtype = BLOB_UINT4;
            for (uint32_t c = 0; c < chunks; c++) {
                for (uint32_t n = 0; n < out; n++)
                    for (uint32_t k = 0; k < kc; k += 2) {
                        uint8_t lo = (uint8_t)(wq[(size_t)n * hidden + c * kc + k] & 15);
                        uint8_t hi = (uint8_t)(wq[(size_t)n * hidden + c * kc + k + 1] & 15);
                        slab[((size_t)n * kc + k) / 2] = (uint8_t)(lo | (hi << 4));
                    }
                q_offs[c] = blob_add(&blob, slab, (size_t)out * kc / 2);
            }
            free(slab);
            g_blob_dtype = BLOB_FP16;
            s_off = blob_add(&blob, lut, sizeof lut);
            for (size_t n = 0; n < out; n++)
                for (size_t k = 0; k < hidden; k++)
                    wref[n * hidden + k] = (float)lut[wq[n * hidden + k] & 15];
        } else {
            // blockwise uint4 with per-(row, group) fp16 scale and uint4 zero point 8.
            uint32_t groups = kc / g_group;
            uint8_t *slab = malloc((size_t)out * kc / 2);
            uint8_t *zp = malloc((size_t)out * groups / 2);
            memset(zp, 0x88, (size_t)out * groups / 2);
            g_blob_dtype = BLOB_UINT4;
            for (uint32_t c = 0; c < chunks; c++) {
                for (uint32_t n = 0; n < out; n++)
                    for (uint32_t k = 0; k < kc; k += 2) {
                        uint8_t lo = (uint8_t)(wq[(size_t)n * hidden + c * kc + k] & 15);
                        uint8_t hi = (uint8_t)(wq[(size_t)n * hidden + c * kc + k + 1] & 15);
                        slab[((size_t)n * kc + k) / 2] = (uint8_t)(lo | (hi << 4));
                    }
                q_offs[c] = blob_add(&blob, slab, (size_t)out * kc / 2);
                zp_offs[c] = blob_add(&blob, zp, (size_t)out * groups / 2);
            }
            free(slab); free(zp);
            g_blob_dtype = BLOB_FP16;
            // One scale plane shared by every chunk in this probe (same values).
            s_off = blob_add(&blob, NULL, (size_t)out * groups * 2);
            __fp16 *dst = (__fp16 *)(void *)(blob.data + s_off + 64);
            for (size_t i = 0; i < (size_t)out * groups; i++) dst[i] = (__fp16)(1.0f / (7.0f * sqrtf((float)hidden)));
            for (size_t n = 0; n < out; n++)
                for (size_t k = 0; k < hidden; k++)
                    wref[n * hidden + k] = ((float)(wq[n * hidden + k] & 15) - 8.0f) * (float)(__fp16)(1.0f / (7.0f * sqrtf((float)hidden)));
        }

        NSString *mil = build_mil(layout, wmode, hidden, out, rows, chunks, q_offs, s_off, zp_offs);
        if (getenv("ANEPROBE_DUMP_MIL")) printf("%s\n", mil.UTF8String);

        NSData *weights = [NSData dataWithBytesNoCopy:blob.data length:blob.cursor freeWhenDone:YES];
        NSData *program = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) { fprintf(stderr, "descriptor rejected\n"); return 1; }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) { fprintf(stderr, "model rejected\n"); return 1; }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[directory stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [program writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weights writeToFile:[directory stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *err = nil;
        double t0 = now_s();
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &err)) {
            fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String ?: "?");
            [fm removeItemAtPath:directory error:nil];
            return 1;
        }
        double t_compile = now_s() - t0;
        t0 = now_s();
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &err)) {
            fprintf(stderr, "load failed: %s\n", err.localizedDescription.UTF8String ?: "?");
            [fm removeItemAtPath:directory error:nil];
            return 1;
        }
        double t_load = now_s() - t0;

        size_t in_bytes = (size_t)hidden * rows * 2, out_bytes = (size_t)out * rows * 2;
        IOSurfaceRef in_surf = make_surface(in_bytes), out_surf = make_surface(out_bytes);
        __fp16 *in_base = IOSurfaceGetBaseAddress(in_surf);
        __fp16 *out_base = IOSurfaceGetBaseAddress(out_surf);
        memset(out_base, 0, out_bytes);
        // Plane layout: width layout stores [hidden][rows]; batch layout [rows][hidden].
        memcpy(in_base, x, in_bytes);

        id in_obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), in_surf);
        id out_obj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), out_surf);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[in_obj], @[@0], @[out_obj], @[@0], nil, nil, @0);
        if (!request) { fprintf(stderr, "request rejected\n"); return 1; }

        double *samples = malloc(sizeof(double) * iters);
        uint32_t ok = 0;
        pthread_t gpu_thread;
        if (gpu_load) {
            pthread_create(&gpu_thread, NULL, gpu_load_thread, NULL);
            usleep(300000);  // let the GEMM loop reach steady state
        }
        if (gpu_only) { usleep((useconds_t)(iters * 20000)); iters = 0; }
        for (uint32_t i = 0; i < iters + 3; i++) {
            double s = now_s();
            BOOL r = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &err);
            double e = now_s() - s;
            if (!r) {
                fprintf(stderr, "eval %u failed: %s\n", i, err.localizedDescription.UTF8String ?: "?");
                [fm removeItemAtPath:directory error:nil];
                return 1;
            }
            if (i >= 3) samples[ok++] = e;  // discard warmup
        }

        if (gpu_load) {
            gpu_stop = 1;
            pthread_join(gpu_thread, NULL);
            double flops = 2.0 * (double)gpu_n * gpu_n * gpu_n * (double)gpu_gemms;
            if (gpu_blit)
                printf("  GPU load: blit %.1f GB moved in %.2fs = %.1f GB/s %s\n",
                       gpu_blit_bytes / 1e9, gpu_seconds, gpu_blit_bytes / gpu_seconds / 1e9,
                       gpu_only ? "(alone)" : "(with ANE evals)");
            else
            printf("  GPU load: %llu GEMMs %ux%u fp16 in %.2fs = %.1f TFLOPS %s\n",
                   (unsigned long long)gpu_gemms, gpu_n, gpu_n, gpu_seconds, flops / gpu_seconds / 1e12,
                   gpu_only ? "(alone)" : "(with ANE evals)");
        }
        if (iters == 0) { printf("  (gpu-only run, no ANE samples)\n"); return 0; }

        // CPU reference on row 0 (fp32 accumulate over int8*fp16).
        double max_abs = 0, dot = 0, na = 0, nb = 0;
        for (uint32_t n = 0; n < out; n++) {
            float acc = 0;
            for (uint32_t k = 0; k < hidden; k++) {
                float xv = layout == LAYOUT_BATCH ? (float)x[k] : (float)x[(size_t)k * rows];
                acc += wref[(size_t)n * hidden + k] * xv;
            }
            float got = layout == LAYOUT_BATCH ? (float)out_base[n] : (float)out_base[(size_t)n * rows];
            double d = fabs((double)got - acc);
            if (d > max_abs) max_abs = d;
            dot += (double)got * acc; na += (double)got * got; nb += (double)acc * acc;
        }
        double cosine = dot / (sqrt(na) * sqrt(nb) + 1e-30);

        qsort(samples, ok, sizeof(double), cmp_double);
        double med = samples[ok / 2], p10 = samples[ok / 10], p90 = samples[(ok * 9) / 10];
        double bytes = (double)out * hidden * (wmode == W_INT8 ? 1.0 : 0.5);  // weight bytes per eval
        printf("ANE probe M5 Max: hidden=%u out=%u rows=%u chunks=%u layout=%s weights=%s\n",
               hidden, out, rows, chunks, layout == LAYOUT_BATCH ? "batch" : "width",
               wmode == W_INT8 ? "int8" : wmode == W_LUT4 ? "lut4" : "block4");
        printf("  compile %.2fs load %.3fs (weights %.1f MB)\n", t_compile, t_load, bytes / 1e6);
        printf("  eval: median %.3f ms  p10 %.3f  p90 %.3f  (n=%u)\n", med * 1e3, p10 * 1e3, p90 * 1e3, ok);
        printf("  effective weight BW %.1f GB/s; %.2f GOPS\n", bytes / med / 1e9, 2.0 * bytes * rows / med / 1e9);
        printf("  parity row0: cosine %.6f max_abs %.4g\n", cosine, max_abs);
        if (!quiet && cosine < 0.999) printf("  !! PARITY FAIL\n");

        CFRelease(in_surf); CFRelease(out_surf);
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &err);
        [fm removeItemAtPath:directory error:nil];
        free(samples); free(wq); free(ws); free(x); free(wref);
    }
    return 0;
}
