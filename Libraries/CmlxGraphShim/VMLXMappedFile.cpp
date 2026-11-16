#include "VMLXMappedFile.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/error.h"
#include "mlx/allocator.h"
#include <limits>
#include <memory>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct MappedFile {
    int fd{-1};
    void *base{MAP_FAILED};
    size_t size{0};
    struct stat identity{};
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    explicit MappedFile(int source) {
        fd = fcntl(source, F_DUPFD_CLOEXEC, 0);
        if (fd < 0) throw std::runtime_error("[mapped_file] cannot retain descriptor");
        if (fstat(fd, &identity) != 0 || !S_ISREG(identity.st_mode) || identity.st_size <= 0 ||
            static_cast<uint64_t>(identity.st_size) > std::numeric_limits<size_t>::max()) {
            close(fd); fd = -1;
            throw std::runtime_error("[mapped_file] expected nonempty regular file");
        }
        size = static_cast<size_t>(identity.st_size);
        base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            close(fd); fd = -1;
            throw std::runtime_error("[mapped_file] mmap failed");
        }
    }
    ~MappedFile() {
        if (base != MAP_FAILED) munmap(base, size);
        if (fd >= 0) close(fd);
    }
    void validate() const {
        struct stat now{};
        if (fstat(fd, &now) != 0 || now.st_dev != identity.st_dev ||
            now.st_ino != identity.st_ino || now.st_size != identity.st_size)
            throw std::runtime_error("[mapped_file] source identity changed");
#ifdef __APPLE__
        if (now.st_mtimespec.tv_sec != identity.st_mtimespec.tv_sec ||
            now.st_mtimespec.tv_nsec != identity.st_mtimespec.tv_nsec ||
            now.st_ctimespec.tv_sec != identity.st_ctimespec.tv_sec ||
            now.st_ctimespec.tv_nsec != identity.st_ctimespec.tv_nsec)
#else
        if (now.st_mtim.tv_sec != identity.st_mtim.tv_sec ||
            now.st_mtim.tv_nsec != identity.st_mtim.tv_nsec ||
            now.st_ctim.tv_sec != identity.st_ctim.tv_sec ||
            now.st_ctim.tv_nsec != identity.st_ctim.tv_nsec)
#endif
            throw std::runtime_error("[mapped_file] source contents changed");
    }
};
using Owner = std::shared_ptr<MappedFile>;
}
#endif

extern "C" int vmlx_mapped_file_create(int32_t fd, void **owner) {
    try {
#ifndef _WIN32
        if (!owner || *owner) throw std::invalid_argument("[mapped_file] invalid owner output");
        *owner = new Owner(std::make_shared<MappedFile>(fd));
        return 0;
#else
        throw std::runtime_error("[mapped_file] unsupported on Windows");
#endif
    } catch (const std::exception& e) { mlx_error(e.what()); return 1; }
}

extern "C" void vmlx_mapped_file_release(void **owner) {
#ifndef _WIN32
    if (owner) { delete static_cast<Owner *>(*owner); *owner = nullptr; }
#endif
}

extern "C" int vmlx_mapped_file_array(void **result, void *opaque, uint64_t offset,
                                      size_t length, const int32_t *dims, int32_t rank,
                                      int32_t type) {
    try {
#ifndef _WIN32
        using namespace mlx::core;
        if (!result || !opaque || !dims || rank <= 0 || rank > 32 || !length)
            throw std::invalid_argument("[mapped_file] invalid view arguments");
        auto owner = *static_cast<Owner *>(opaque);
        owner->validate();
        const auto dtype = mlx_dtype_to_cpp(static_cast<mlx_dtype>(type));
        const auto item = size_of(dtype);
        size_t bytes = item;
        Shape shape;
        for (int i = 0; i < rank; ++i) {
            if (dims[i] <= 0 || bytes > std::numeric_limits<size_t>::max() / size_t(dims[i]))
                throw std::invalid_argument("[mapped_file] invalid or overflowing shape");
            bytes *= size_t(dims[i]); shape.push_back(dims[i]);
        }
        if (bytes != length || offset > owner->size || length > owner->size - offset)
            throw std::invalid_argument("[mapped_file] view outside mapped file or shape mismatch");
        const auto page = size_t(getpagesize());
        const auto delta = size_t(offset % page);
        if (offset % item || length > size_t(std::numeric_limits<ShapeElem>::max()) - delta)
            throw std::invalid_argument("[mapped_file] misaligned or oversized view");
        const auto span = length + delta;
        auto buffer = allocator::make_buffer(static_cast<char *>(owner->base) + offset - delta, span);
        if (!buffer.ptr()) throw std::runtime_error("[mapped_file] Metal buffer creation failed");
        auto release = [](void *p) { allocator::release(allocator::Buffer(p)); };
        std::unique_ptr<void, decltype(release)> pending(buffer.ptr(), release);
        // Data with a deleter does not own the memory, so MLX never donates it to an op's output: the mapping is
        // read-only, and a write into it is a SIGBUS on the CPU (ml-explore/mlx#4634).
        allocator::Data data(buffer, [owner](allocator::Buffer b) { allocator::release(b); });
        pending.release();
        array base(std::move(data), Shape{ShapeElem(span)}, uint8);
        array view(allocator::Data(allocator::Buffer(nullptr), [](allocator::Buffer) {}), shape, dtype);
        view.copy_shared_buffer(base, view.strides(), view.flags(), view.size(), int64_t(delta / item));
        owner->validate();
        mlx_array output{*result};
        mlx_array_set_(output, std::move(view));
        *result = output.ctx;
        return 0;
#else
        throw std::runtime_error("[mapped_file] unsupported on Windows");
#endif
    } catch (const std::exception& e) { mlx_error(e.what()); return 1; }
}
