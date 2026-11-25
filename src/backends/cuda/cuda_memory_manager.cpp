#include "backends/cuda/cuda_memory_manager.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_common.h"

namespace hpcplatform {

CudaAllocatorState::~CudaAllocatorState() {
  cudaSetDevice(device_index);
  for (const CudaBlock& block : free_blocks)
    cudaFree(block.raw);
}

void CudaAllocatorState::Release(void* raw, void* data, size_t bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    free_blocks.push_back(CudaBlock{raw, data, bytes});
  }
  allocated_bytes.fetch_sub(bytes);
  allocation_count.fetch_sub(1);
}

Result<std::shared_ptr<Buffer>> CudaMemoryManager::Allocate(size_t bytes,
                                                            size_t alignment) {
  if (bytes == 0)
    return Status::InvalidArgument("CUDA allocation size must be positive");
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      alignment % sizeof(void*) != 0) {
    return Status::InvalidArgument(
        "CUDA allocation alignment must be a power of two and a multiple of "
        "sizeof(void*)");
  }
  if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return Status::InvalidArgument(
        "CUDA allocation size overflows alignment padding");
  }

  CudaBlock block;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (auto it = state_->free_blocks.begin(); it != state_->free_blocks.end();
         ++it) {
      if (it->bytes == bytes &&
          reinterpret_cast<uintptr_t>(it->data) % alignment == 0) {
        block = *it;
        state_->free_blocks.erase(it);
        found = true;
        break;
      }
    }
  }

  if (!found) {
    hpcplatform_RETURN_IF_ERROR(
        CudaStatus(cudaSetDevice(device_.index), "cudaSetDevice"));
    hpcplatform_RETURN_IF_ERROR(CudaStatus(
        cudaMalloc(&block.raw, bytes + alignment - 1), "cudaMalloc"));
    const uintptr_t raw = reinterpret_cast<uintptr_t>(block.raw);
    block.data =
        reinterpret_cast<void*>((raw + alignment - 1) & ~(alignment - 1));
    block.bytes = bytes;
  }

  const size_t current = state_->allocated_bytes.fetch_add(bytes) + bytes;
  state_->allocation_count.fetch_add(1);
  size_t peak = state_->peak_bytes.load();
  while (current > peak &&
         !state_->peak_bytes.compare_exchange_weak(peak, current)) {
  }
  auto state = state_;
  return std::make_shared<Buffer>(
      block.data, bytes, device_, MemoryKind::kDevice,
      [state, raw = block.raw](void* data, size_t size) {
        state->Release(raw, data, size);
      });
}

MemoryStats CudaMemoryManager::Stats() const {
  MemoryStats stats;
  stats.allocated_bytes = state_->allocated_bytes.load();
  stats.peak_bytes = state_->peak_bytes.load();
  stats.allocation_count = state_->allocation_count.load();
  return stats;
}

}  // namespace hpcplatform
