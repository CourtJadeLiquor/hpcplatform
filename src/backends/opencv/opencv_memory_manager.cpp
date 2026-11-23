#include "backends/opencv/opencv_memory_manager.h"

#include <cstring>
#include <limits>
#include <memory>

#include <opencv2/core.hpp>

namespace hpcplatform {

Result<std::shared_ptr<Buffer>> OpenCvMemoryManager::Allocate(
    size_t bytes, size_t alignment) {
  if (bytes == 0 ||
      bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return Status::InvalidArgument("OpenCV allocation size is invalid");
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Status::InvalidArgument(
        "OpenCV allocation alignment must be a power of two");
  }

  auto mat = std::make_unique<cv::Mat>();
  try {
    mat->create(1, static_cast<int>(bytes), CV_8UC1);
  } catch (const cv::Exception& error) {
    return Status::OutOfMemory(std::string("cv::Mat allocation failed: ") +
                               error.what());
  }
  if (mat->data == nullptr)
    return Status::OutOfMemory("cv::Mat allocation returned null");
  if (reinterpret_cast<uintptr_t>(mat->data) % alignment != 0) {
    return Status::Unsupported(
        "cv::Mat allocator cannot satisfy the requested alignment");
  }

  const size_t current = state_->allocated_bytes.fetch_add(bytes) + bytes;
  state_->allocation_count.fetch_add(1);
  size_t peak = state_->peak_bytes.load();
  while (current > peak &&
         !state_->peak_bytes.compare_exchange_weak(peak, current)) {
  }
  auto state = state_;
  cv::Mat* owned_mat = mat.release();
  return std::make_shared<Buffer>(owned_mat->data, bytes, device_,
                                  MemoryKind::kHost,
                                  [state, owned_mat](void*, size_t size) {
                                    delete owned_mat;
                                    state->allocated_bytes.fetch_sub(size);
                                    state->allocation_count.fetch_sub(1);
                                  });
}

Status OpenCvMemoryManager::Copy(Buffer* dst, size_t dst_offset,
                                 const Buffer* src, size_t src_offset,
                                 size_t bytes, Stream* stream) {
  (void)stream;
  if (dst == nullptr || src == nullptr || bytes == 0) {
    return Status::InvalidArgument(
        "OpenCV copy requires buffers and non-zero bytes");
  }
  if (dst->kind() == MemoryKind::kDevice ||
      src->kind() == MemoryKind::kDevice) {
    return Status::Unsupported("OpenCV memory manager copies host memory only");
  }
  if (dst_offset > dst->size() || bytes > dst->size() - dst_offset ||
      src_offset > src->size() || bytes > src->size() - src_offset) {
    return Status::InvalidArgument("OpenCV copy range out of bounds");
  }
  std::memcpy(static_cast<char*>(dst->data()) + dst_offset,
              static_cast<const char*>(src->data()) + src_offset, bytes);
  return Status::Ok();
}

MemoryStats OpenCvMemoryManager::Stats() const {
  MemoryStats stats;
  stats.allocated_bytes = state_->allocated_bytes.load();
  stats.peak_bytes = state_->peak_bytes.load();
  stats.allocation_count = state_->allocation_count.load();
  return stats;
}

}  // namespace hpcplatform
