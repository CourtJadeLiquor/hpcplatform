#ifndef hpcplatform_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_
#define hpcplatform_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_

#include <atomic>
#include <cstddef>
#include <memory>

#include <cuda_runtime_api.h>

#include "hpcplatform/core/memory.h"

namespace hpcplatform
{

    // Allocation statistics shared between the manager and buffer deleters, so a
    // buffer may outlive the manager that produced it.
    struct CudaMemoryState
    {
        std::atomic<size_t> allocated_bytes{0};
        std::atomic<size_t> peak_bytes{0};
        std::atomic<size_t> allocation_count{0};
    };

    // Allocates unified (managed) memory with cudaMallocManaged. Each block is
    // prefetched to the GPU as a whole with cudaMemPrefetchAsync right after
    // allocation, so device kernels read resident pages instead of faulting per
    // page. The host still dereferences the same pointer directly, which removes
    // every explicit H2D/D2H staging copy from operator code paths.
    class CudaMemoryManager final : public MemoryManager
    {
    public:
        explicit CudaMemoryManager(Device device);

        CudaMemoryManager(const CudaMemoryManager&) = delete;
        CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;

        Device device() const override { return device_; }
        Result<std::shared_ptr<Buffer>> Allocate(size_t bytes, size_t alignment) override;
        MemoryStats Stats() const override;

        // Orders prefetches on the backend stream so a migration cannot race the
        // kernels enqueued afterwards. Called once by the backend during
        // Initialize; before that, prefetches run on the default stream.
        void AttachStream(cudaStream_t stream) { prefetch_stream_ = stream; }

    private:
        Device device_;
        cudaStream_t prefetch_stream_ = nullptr;
        std::shared_ptr<CudaMemoryState> state_ = std::make_shared<CudaMemoryState>();
    };

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_
