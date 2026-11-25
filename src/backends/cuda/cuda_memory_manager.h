#ifndef hpcplatform_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_
#define hpcplatform_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "hpcplatform/core/memory.h"

namespace hpcplatform {

struct CudaBlock {
  void* raw = nullptr;
  void* data = nullptr;
  size_t bytes = 0;
};

struct CudaAllocatorState {
  explicit CudaAllocatorState(int index) : device_index(index) {}
  ~CudaAllocatorState();

  void Release(void* raw, void* data, size_t bytes);

  int device_index;
  std::mutex mutex;
  std::vector<CudaBlock> free_blocks;
  std::atomic<size_t> allocated_bytes{0};
  std::atomic<size_t> peak_bytes{0};
  std::atomic<size_t> allocation_count{0};
};

class CudaMemoryManager final : public MemoryManager {
 public:
  explicit CudaMemoryManager(Device device)
      : device_(device),
        state_(std::make_shared<CudaAllocatorState>(device.index)) {}

  Device device() const override { return device_; }
  Result<std::shared_ptr<Buffer>> Allocate(size_t bytes,
                                           size_t alignment) override;
  MemoryStats Stats() const override;

 private:
  Device device_;
  std::shared_ptr<CudaAllocatorState> state_;
};

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_
