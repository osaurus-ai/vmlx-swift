// Stream.runWith: run callback with a dedicated default stream.
// Uses only public mlx C API — no internal mlx_stream_get_ needed.

#include <Cmlx/mlx-c-stream.h>
#include <Cmlx/mlx-c-device.h>

extern "C" int mlx_stream_run_with(
    mlx_stream stream,
    void (*callback)(void* context),
    void* context) {

    // Save current default
    mlx_device dev = mlx_default_device_new();
    mlx_stream old_stream = mlx_stream_new();
    int rc = mlx_get_default_stream(&old_stream, dev);
    if (rc != 0) {
        mlx_stream_free(old_stream);
        mlx_device_free(dev);
        return rc;
    }

    // Set new default
    rc = mlx_set_default_stream(stream);
    if (rc != 0) {
        mlx_stream_free(old_stream);
        mlx_device_free(dev);
        return rc;
    }

    // Run callback
    callback(context);

    // Restore original default
    mlx_set_default_stream(old_stream);
    mlx_stream_free(old_stream);
    mlx_device_free(dev);
    return 0;
}
