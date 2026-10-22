#include "VMLXResidentFile.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/error.h"
#include "mlx/allocator.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

extern "C" int vmlx_resident_file_array(
    void **result, int32_t fd, uint64_t offset, size_t length,
    const int32_t *dims, int32_t rank, int32_t type) {
  try {
#ifndef _WIN32
    using namespace mlx::core;
    if (!result || !length || fd < 0 || rank < 0 || rank > 32 || (rank && !dims))
      throw std::invalid_argument("[resident_file] invalid arguments");
    const auto dtype = mlx_dtype_to_cpp(static_cast<mlx_dtype>(type));
    size_t bytes = size_of(dtype);
    Shape shape;
    for (int i = 0; i < rank; ++i) {
      if (dims[i] < 0 || (dims[i] && bytes > std::numeric_limits<size_t>::max() / size_t(dims[i])))
        throw std::invalid_argument("[resident_file] invalid or overflowing shape");
      bytes *= size_t(dims[i]);
      shape.push_back(dims[i]);
    }
    struct stat before{};
    if (fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_size < 0 ||
        bytes != length || offset > uint64_t(before.st_size) ||
        length > uint64_t(before.st_size) - offset)
      throw std::invalid_argument("[resident_file] invalid tensor span");
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1))
      throw std::runtime_error("[resident_file] cannot disable filesystem data caching");
#endif
    auto buffer = allocator::malloc(length);
    auto release = [](void *p) { allocator::free(allocator::Buffer(p)); };
    std::unique_ptr<void, decltype(release)> pending(buffer.ptr(), release);
    array tensor(buffer, shape, dtype);
    pending.release();
    auto destination = tensor.data<char>();
    auto read = [&](char *out, size_t count, uint64_t position) {
      while (count) {
        const auto n = pread(fd, out, std::min(count, size_t(8 * 1024 * 1024)), off_t(position));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("[resident_file] short or failed read");
        out += n; position += n; count -= size_t(n);
      }
    };
    // Only the unaligned head/tail need a page-sized temporary. The aligned
    // middle is read straight into MLX storage, without Foundation/Data copies.
    const size_t page = size_t(getpagesize());
    std::vector<char> edge(page);
    uint64_t position = offset;
    size_t remaining = length;
    if (remaining && position % page) {
      const auto start = position - position % page;
      const auto available = std::min(uint64_t(page), uint64_t(before.st_size) - start);
      read(edge.data(), size_t(available), start);
      const auto n = std::min(remaining, size_t(available - position % page));
      std::memcpy(destination, edge.data() + position % page, n);
      destination += n; position += n; remaining -= n;
    }
    const auto middle = remaining - remaining % page;
    read(destination, middle, position);
    destination += middle; position += middle; remaining -= middle;
    if (remaining) {
      const auto available = std::min(uint64_t(page), uint64_t(before.st_size) - position);
      read(edge.data(), size_t(available), position);
      std::memcpy(destination, edge.data(), remaining);
    }
    struct stat after{};
    if (fstat(fd, &after) || before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size)
      throw std::runtime_error("[resident_file] source changed during read");
#ifdef __APPLE__
    if (before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec ||
        before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec ||
        before.st_ctimespec.tv_sec != after.st_ctimespec.tv_sec ||
        before.st_ctimespec.tv_nsec != after.st_ctimespec.tv_nsec)
      throw std::runtime_error("[resident_file] source modified during read");
#endif
    mlx_array output{*result};
    mlx_array_set_(output, std::move(tensor));
    *result = output.ctx;
    return 0;
#else
    throw std::runtime_error("[resident_file] unsupported on Windows");
#endif
  } catch (const std::exception& e) { mlx_error(e.what()); return 1; }
}
