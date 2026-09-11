// Private AppleNeuralEngine bridge. See include/vmlx_ane_bridge.h for the
// contract; the mechanics follow the publicly documented behaviour of
// _ANEInMemoryModelDescriptor / _ANEInMemoryModel / _ANERequest.

#import "vmlx_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

struct vmlx_ane_plane {
    IOSurfaceRef surface;
    size_t bytes;
};

struct vmlx_ane_model {
    void *model;       /* _ANEInMemoryModel, +1 */
    void *requests;    /* NSArray<_ANERequest *>, one per procedure */
    char *staging_directory;
    double compile_seconds;
    bool cache_hit;
};

static void bridge_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double bridge_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int vmlx_ane_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    state = NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil
        && NSClassFromString(@"_ANEInMemoryModel") != nil
        && NSClassFromString(@"_ANERequest") != nil
        && NSClassFromString(@"_ANEIOSurfaceObject") != nil;
    return state;
}

vmlx_ane_plane *vmlx_ane_plane_create(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
    if (!surface) return NULL;
    vmlx_ane_plane *plane = calloc(1, sizeof *plane);
    if (!plane) { CFRelease(surface); return NULL; }
    plane->surface = surface;
    plane->bytes = aligned;
    memset(IOSurfaceGetBaseAddress(surface), 0, aligned);
    return plane;
}

void vmlx_ane_plane_free(vmlx_ane_plane *plane) {
    if (!plane) return;
    if (plane->surface) CFRelease(plane->surface);
    free(plane);
}

void *vmlx_ane_plane_base(vmlx_ane_plane *plane) {
    return plane ? IOSurfaceGetBaseAddress(plane->surface) : NULL;
}

size_t vmlx_ane_plane_bytes(const vmlx_ane_plane *plane) {
    return plane ? plane->bytes : 0;
}

// ── compile cache ──

static NSString *bridge_content_hash(NSData *program, NSData *weights) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    CC_SHA256_Update(&ctx, program.bytes, (CC_LONG)program.length);
    CC_SHA256_Update(&ctx, weights.bytes, (CC_LONG)weights.length);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);
    NSMutableString *hex = [NSMutableString stringWithCapacity:64];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [hex appendFormat:@"%02x", digest[i]];
    return hex;
}

static void bridge_write_sources(NSString *directory, NSData *program, NSData *weights) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files createDirectoryAtPath:[directory stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [program writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weights writeToFile:[directory stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
}

/* Hardlink a directory tree (free on the same APFS volume); copy otherwise. */
static bool bridge_mirror(NSString *from, NSString *to) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:to error:nil];
    [files createDirectoryAtPath:[to stringByDeletingLastPathComponent]
        withIntermediateDirectories:YES attributes:nil error:nil];
    if ([files linkItemAtPath:from toPath:to error:nil]) return true;
    [files removeItemAtPath:to error:nil];
    return [files copyItemAtPath:from toPath:to error:nil];
}

static bool bridge_cache_restore(NSString *entry, NSString *directory) {
    NSFileManager *files = [NSFileManager defaultManager];
    if (![files fileExistsAtPath:[entry stringByAppendingPathComponent:@"compiled.ok"]]) return false;
    [files setAttributes:@{NSFileModificationDate: [NSDate date]} ofItemAtPath:entry error:nil];
    if (!bridge_mirror(entry, directory)) return false;
    [files removeItemAtPath:[directory stringByAppendingPathComponent:@"compiled.ok"] error:nil];
    return true;
}

/* Only the compiled artifacts are kept: `data` embeds the constants, so the
 * weights copy and the MIL text are dead weight in an entry. */
static void bridge_cache_store(NSString *entry, NSString *directory) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:entry error:nil];
    if (![files createDirectoryAtPath:entry withIntermediateDirectories:YES attributes:nil error:nil]) return;
    for (NSString *name in [files contentsOfDirectoryAtPath:directory error:nil]) {
        if ([name isEqualToString:@"weights"] || [name isEqualToString:@"model.mil"]
            || [name isEqualToString:@"vmlx-ane.pid"]) continue;
        NSString *from = [directory stringByAppendingPathComponent:name];
        NSString *to = [entry stringByAppendingPathComponent:name];
        if (![files linkItemAtPath:from toPath:to error:nil]) {
            [files removeItemAtPath:to error:nil];
            if (![files copyItemAtPath:from toPath:to error:nil]) return;
        }
    }
    [[NSData data] writeToFile:[entry stringByAppendingPathComponent:@"compiled.ok"] atomically:YES];
}

/* A killed process never frees its staging; reap marked dirs whose owner is
 * gone, once per process. */
static void bridge_reap_orphans(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *tmp = NSTemporaryDirectory();
        for (NSString *entry in [fm contentsOfDirectoryAtPath:tmp error:nil]) {
            NSString *dir = [tmp stringByAppendingPathComponent:entry];
            NSString *marker = [dir stringByAppendingPathComponent:@"vmlx-ane.pid"];
            NSString *owner_text = [NSString stringWithContentsOfFile:marker
                                                             encoding:NSUTF8StringEncoding error:nil];
            if (!owner_text) continue;
            pid_t owner = (pid_t)owner_text.integerValue;
            if (owner <= 0 || owner == getpid()) continue;
            if (kill(owner, 0) != 0 && errno == ESRCH) [fm removeItemAtPath:dir error:nil];
        }
    });
}

// ── symbol indices ──

/* A bank's procedures do NOT share one symbol numbering; ask the inner
 * _ANEModel (reachable through -model once loaded) per procedure. */
static id bridge_inner_model(id in_memory_model) {
    if (![in_memory_model respondsToSelector:@selector(model)]) return nil;
    return ((id(*)(id, SEL))objc_msgSend)(in_memory_model, @selector(model));
}

static NSArray *bridge_symbol_indices(id in_memory_model, bool want_input, uint32_t proc, uint32_t count) {
    id model = bridge_inner_model(in_memory_model) ?: in_memory_model;
    NSString *key = want_input ? @"ANEFModelInputSymbolIndexArray" : @"ANEFModelOutputSymbolIndexArray";
    if ([model respondsToSelector:@selector(procedureInfoForProcedureIndex:)]) {
        id info = ((id(*)(id, SEL, unsigned int))objc_msgSend)(model, @selector(procedureInfoForProcedureIndex:), proc);
        if ([info isKindOfClass:[NSDictionary class]]) {
            id indices = ((NSDictionary *)info)[key];
            if ([indices isKindOfClass:[NSArray class]] && [(NSArray *)indices count] == count) return indices;
        }
    }
    NSMutableArray *identity = [NSMutableArray array];
    for (uint32_t i = 0; i < count; i++) [identity addObject:@(i)];
    return identity;
}

static bool bridge_has_symbol_api(id in_memory_model) {
    id model = bridge_inner_model(in_memory_model) ?: in_memory_model;
    return model && [model respondsToSelector:@selector(procedureInfoForProcedureIndex:)];
}

// ── model ──

vmlx_ane_model *vmlx_ane_model_create(
    const char *name, const char *mil_text,
    const void *weight_bytes_in, size_t weight_bytes,
    vmlx_ane_plane *const *inputs, uint32_t input_count,
    vmlx_ane_plane *const *outputs, uint32_t output_count,
    uint32_t procedure_count, const char *cache_dir,
    char *error, size_t error_size)
{
    if (!procedure_count) procedure_count = 1;
    if (!vmlx_ane_available()) {
        bridge_fail(error, error_size, "the Neural Engine bridge is unavailable");
        return NULL;
    }
    if (!output_count) {
        bridge_fail(error, error_size, "ANE %s: a program needs at least one output plane", name);
        return NULL;
    }
    vmlx_ane_model *handle = calloc(1, sizeof *handle);
    if (!handle) {
        bridge_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytes:weight_bytes_in length:weight_bytes];
        NSData *program = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            bridge_fail(error, error_size, "ANE %s descriptor rejected", name);
            free(handle);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            bridge_fail(error, error_size, "ANE %s model rejected", name);
            free(handle);
            return NULL;
        }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        bridge_reap_orphans();
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        NSString *entry = nil;
        if (cache_dir) {
            entry = [[@(cache_dir) stringByAppendingPathComponent:@"entries"]
                stringByAppendingPathComponent:bridge_content_hash(program, weights)];
        }
        bool cached = entry && bridge_cache_restore(entry, directory);
        if (!cached) bridge_write_sources(directory, program, weights);
        [[NSString stringWithFormat:@"%d", getpid()]
            writeToFile:[directory stringByAppendingPathComponent:@"vmlx-ane.pid"]
             atomically:YES encoding:NSUTF8StringEncoding error:nil];
        handle->staging_directory = strdup(directory.UTF8String);

        double started = bridge_seconds();
        bool loaded = cached && ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &failure);
        if (cached && !loaded) {
            failure = nil;
            bridge_write_sources(directory, program, weights);
        }
        if (!loaded) {
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(compileWithQoS:options:error:), 21, @{}, &failure)) {
                bridge_fail(error, error_size, "ANE %s compile failed: %s", name,
                            failure ? failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(loadWithQoS:options:error:), 21, @{}, &failure)) {
                bridge_fail(error, error_size, "ANE %s load failed: %s", name,
                            failure ? failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (entry) bridge_cache_store(entry, directory);
        }
        /* Compile inputs are not serving inputs: drop them once loaded. The
         * compiled artifacts must stay (aned demand-reads them). */
        [files removeItemAtPath:[directory stringByAppendingPathComponent:@"weights"] error:nil];
        [files removeItemAtPath:[directory stringByAppendingPathComponent:@"model.mil"] error:nil];
        handle->cache_hit = loaded;
        handle->compile_seconds = bridge_seconds() - started;

        NSMutableArray *in_objs = [NSMutableArray array], *out_objs = [NSMutableArray array];
        for (uint32_t i = 0; i < input_count; i++)
            [in_objs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), inputs[i]->surface)];
        for (uint32_t i = 0; i < output_count; i++)
            [out_objs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), outputs[i]->surface)];
        if (procedure_count > 1 && !bridge_has_symbol_api(model)) {
            bridge_fail(error, error_size, "ANE %s: no per-procedure symbol indices on this build; "
                        "a bank of %u cannot be dispatched", name, procedure_count);
            handle->model = (__bridge_retained void *)model;
            vmlx_ane_model_free(handle);
            return NULL;
        }
        NSMutableArray *requests = [NSMutableArray array];
        for (uint32_t proc = 0; proc < procedure_count; proc++) {
            NSArray *in_indices = bridge_symbol_indices(model, true, proc, input_count);
            NSArray *out_indices = bridge_symbol_indices(model, false, proc, output_count);
            id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                in_objs, in_indices, out_objs, out_indices, nil, nil, @(proc));
            if (!request) {
                bridge_fail(error, error_size, "ANE %s request rejected for procedure %u", name, proc);
                handle->model = (__bridge_retained void *)model;
                vmlx_ane_model_free(handle);
                return NULL;
            }
            [requests addObject:request];
        }
        handle->model = (__bridge_retained void *)model;
        handle->requests = (__bridge_retained void *)requests;
    }
    return handle;
}

int vmlx_ane_model_eval(vmlx_ane_model *handle, uint32_t procedure, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->requests) {
        bridge_fail(error, error_size, "the ANE model is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSArray *requests = (__bridge NSArray *)handle->requests;
        if (procedure >= requests.count) {
            bridge_fail(error, error_size, "ANE procedure %u is outside the bank's %lu procedures",
                        procedure, (unsigned long)requests.count);
            return 0;
        }
        NSError *failure = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            (__bridge id)handle->model, @selector(evaluateWithQoS:options:request:error:), 21, @{},
            requests[procedure], &failure) ? 1 : 0;
        if (!ok)
            bridge_fail(error, error_size, "ANE evaluation failed: %s",
                        failure ? failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}

void vmlx_ane_model_free(vmlx_ane_model *handle) {
    if (!handle) return;
    @autoreleasepool {
        if (handle->model) {
            id model = (__bridge_transfer id)handle->model;
            NSError *failure = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &failure);
        }
        if (handle->requests) {
            id requests = (__bridge_transfer id)handle->requests;
            (void)requests;
        }
        if (handle->staging_directory) {
            [[NSFileManager defaultManager] removeItemAtPath:@(handle->staging_directory) error:nil];
            free(handle->staging_directory);
        }
    }
    free(handle);
}

double vmlx_ane_model_compile_seconds(const vmlx_ane_model *handle) {
    return handle ? handle->compile_seconds : 0.0;
}

bool vmlx_ane_model_cache_hit(const vmlx_ane_model *handle) {
    return handle ? handle->cache_hit : false;
}
