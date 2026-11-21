#include "backends/cpu/cpu_memory_manager.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace mycv {
namespace {

void* AlignedAlloc(size_t bytes, size_t alignment) {
#if defined(_WIN32)
  return _aligned_malloc(bytes, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, bytes) != 0) return nullptr;
  return ptr;
#endif
}

void AlignedFree(void* ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

bool IsValidAlignment(size_t alignment) {
  return alignment != 0 && (alignment & (alignment - 1)) == 0 &&
         alignment % sizeof(void*) == 0;
}

}  // namespace

CpuAllocatorState::~CpuAllocatorState() {
  for (const CpuBlock& block : free_blocks) {
    AlignedFree(block.data);
  }
}

void CpuAllocatorState::Release(void* data, size_t bytes) {
  if (data != nullptr && bytes != 0) {
    std::lock_guard<std::mutex> lock(mutex);
    free_blocks.push_back(CpuBlock{data, bytes});
  }
  allocated_bytes.fetch_sub(bytes, std::memory_order_relaxed);
  allocation_count.fetch_sub(1, std::memory_order_relaxed);
}

Result<std::shared_ptr<Buffer>> CpuMemoryManager::Allocate(size_t bytes,
                                                           size_t alignment) {
  if (bytes == 0) {
    return Status::InvalidArgument("allocation size must be greater than 0");
  }
  if (!IsValidAlignment(alignment)) {
    return Status::InvalidArgument(
        "alignment must be a power of two and a multiple of sizeof(void*)");
  }

  void* ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (auto block = state_->free_blocks.begin();
         block != state_->free_blocks.end(); ++block) {
      if (block->bytes == bytes &&
          reinterpret_cast<uintptr_t>(block->data) % alignment == 0) {
        ptr = block->data;
        state_->free_blocks.erase(block);
        break;
      }
    }
  }
  if (ptr == nullptr) {
    ptr = AlignedAlloc(bytes, alignment);
    if (ptr == nullptr) {
      return Status::OutOfMemory("host allocation of " + std::to_string(bytes) +
                                 " bytes failed");
    }
  }

  const size_t current = state_->allocated_bytes.fetch_add(bytes) + bytes;
  state_->allocation_count.fetch_add(1);
  size_t peak = state_->peak_bytes.load(std::memory_order_relaxed);
  while (current > peak && !state_->peak_bytes.compare_exchange_weak(
                               peak, current, std::memory_order_relaxed)) {
  }

  auto state = state_;
  return std::make_shared<Buffer>(
      ptr, bytes, device_, MemoryKind::kHost,
      [state](void* data, size_t size) { state->Release(data, size); });
}

Status CpuMemoryManager::Copy(Buffer* dst, size_t dst_offset, const Buffer* src,
                              size_t src_offset, size_t bytes, Stream* stream) {
  (void)stream;  // CPU copies are synchronous; the stream is accepted for API
                 // symmetry.
  if (dst == nullptr || src == nullptr) {
    return Status::InvalidArgument("copy requires non-null buffers");
  }
  if (bytes == 0) {
    return Status::InvalidArgument("copy size must be greater than 0");
  }
  if (dst_offset > dst->size() || bytes > dst->size() - dst_offset) {
    return Status::InvalidArgument("destination copy range out of bounds");
  }
  if (src_offset > src->size() || bytes > src->size() - src_offset) {
    return Status::InvalidArgument("source copy range out of bounds");
  }
  if (dst->kind() == MemoryKind::kDevice ||
      src->kind() == MemoryKind::kDevice) {
    return Status::Unsupported(
        "CpuMemoryManager cannot copy device memory; use Context::Copy once "
        "cross-device support lands");
  }

  std::memcpy(static_cast<char*>(dst->data()) + dst_offset,
              static_cast<const char*>(src->data()) + src_offset, bytes);
  return Status::Ok();
}

MemoryStats CpuMemoryManager::Stats() const {
  MemoryStats stats;
  stats.allocated_bytes = state_->allocated_bytes.load();
  stats.peak_bytes = state_->peak_bytes.load();
  stats.allocation_count = state_->allocation_count.load();
  return stats;
}

}  // namespace mycv
