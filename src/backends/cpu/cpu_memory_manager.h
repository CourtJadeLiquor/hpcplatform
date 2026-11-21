#ifndef MYCV_SRC_BACKENDS_CPU_CPU_MEMORY_MANAGER_H_
#define MYCV_SRC_BACKENDS_CPU_CPU_MEMORY_MANAGER_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "mycv/core/memory.h"

namespace mycv {

// A released block is kept for exact-size reuse so repeated operator calls in a
// pipeline do not hit malloc/free every iteration. Cached blocks are returned
// to the OS when the last Buffer sharing this state is destroyed.
struct CpuBlock {
  void* data = nullptr;
  size_t bytes = 0;
};

// Shared between the manager and every Buffer deleter so a tensor may outlive
// its Context without dangling pointers.
struct CpuAllocatorState {
  ~CpuAllocatorState();

  void Release(void* data, size_t bytes);

  std::mutex mutex;
  std::vector<CpuBlock> free_blocks;
  std::atomic<size_t> allocated_bytes{0};
  std::atomic<size_t> peak_bytes{0};
  std::atomic<size_t> allocation_count{0};
};

class CpuMemoryManager final : public MemoryManager {
 public:
  explicit CpuMemoryManager(Device device) : device_(device) {}

  Device device() const override { return device_; }

  Result<std::shared_ptr<Buffer>> Allocate(size_t bytes,
                                           size_t alignment) override;

  Status Copy(Buffer* dst, size_t dst_offset, const Buffer* src,
              size_t src_offset, size_t bytes, Stream* stream) override;

  MemoryStats Stats() const override;

 private:
  Device device_;
  std::shared_ptr<CpuAllocatorState> state_ =
      std::make_shared<CpuAllocatorState>();
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_CPU_CPU_MEMORY_MANAGER_H_
