#include "backends/cuda/cuda_memory_manager.h"

#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_common.h"

namespace hpcplatform
{

    CudaMemoryManager::CudaMemoryManager(Device device) : device_(device) {}

    Result<std::shared_ptr<Buffer>> CudaMemoryManager::Allocate(size_t bytes, size_t alignment)
    {
        if (bytes == 0) return Status::InvalidArgument("CUDA allocation size must be positive");
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment % sizeof(void*) != 0)
        {
            return Status::InvalidArgument(
                "CUDA allocation alignment must be a power of two and a multiple of "
                "sizeof(void*)");
        }
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1))
        {
            return Status::InvalidArgument("CUDA allocation size overflows alignment padding");
        }

        hpcplatform_RETURN_IF_ERROR(CudaStatus(cudaSetDevice(device_.index), "cudaSetDevice"));
        void* raw = nullptr;
        hpcplatform_RETURN_IF_ERROR(
            CudaStatus(cudaMallocManaged(&raw, bytes + alignment - 1), "cudaMallocManaged"));
        const uintptr_t address = reinterpret_cast<uintptr_t>(raw);
        void* data = reinterpret_cast<void*>((address + alignment - 1) & ~(alignment - 1));

        // Prefetch the reserved block (payload plus alignment padding) as one
        // operation. Stream-ordered behind the backend stream when attached, so a
        // kernel enqueued afterwards cannot race the page migration.
        const size_t reserved = bytes + alignment - 1;
        hpcplatform_RETURN_IF_ERROR(
            CudaPrefetch(raw, reserved, device_.index, prefetch_stream_));

        const size_t current = state_->allocated_bytes.fetch_add(bytes) + bytes;
        state_->allocation_count.fetch_add(1);
        size_t peak = state_->peak_bytes.load();
        while (current > peak && !state_->peak_bytes.compare_exchange_weak(peak, current))
        {
        }

        auto state = state_;
        return std::make_shared<Buffer>(
            data, bytes, device_, MemoryKind::kDevice,
            [state, device_index = device_.index, raw](void* /*data*/, size_t size)
            {
                state->allocated_bytes.fetch_sub(size);
                state->allocation_count.fetch_sub(1);
                cudaSetDevice(device_index);
                cudaFree(raw);
            });
    }

    MemoryStats CudaMemoryManager::Stats() const
    {
        MemoryStats stats;
        stats.allocated_bytes = state_->allocated_bytes.load();
        stats.peak_bytes = state_->peak_bytes.load();
        stats.allocation_count = state_->allocation_count.load();
        return stats;
    }

}  // namespace hpcplatform
