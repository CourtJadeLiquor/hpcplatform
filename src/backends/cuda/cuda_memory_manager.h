#ifndef MYCV_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_
#define MYCV_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "mycv/core/memory.h"

namespace mycv {

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
      : device_(device), state_(std::make_shared<CudaAllocatorState>(device.index)) {}

  Device device() const override { return device_; }
  Result<std::shared_ptr<Buffer>> Allocate(size_t bytes, size_t alignment) override;
  Status Copy(Buffer* dst, size_t dst_offset, const Buffer* src,
              size_t src_offset, size_t bytes, Stream* stream) override;
  MemoryStats Stats() const override;

 private:
  Device device_;
  std::shared_ptr<CudaAllocatorState> state_;
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_CUDA_CUDA_MEMORY_MANAGER_H_
