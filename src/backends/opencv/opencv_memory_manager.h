#ifndef MYCV_SRC_BACKENDS_OPENCV_OPENCV_MEMORY_MANAGER_H_
#define MYCV_SRC_BACKENDS_OPENCV_OPENCV_MEMORY_MANAGER_H_

#include <atomic>
#include <memory>

#include "mycv/core/memory.h"

namespace mycv {

struct OpenCvAllocatorState {
  std::atomic<size_t> allocated_bytes{0};
  std::atomic<size_t> peak_bytes{0};
  std::atomic<size_t> allocation_count{0};
};

class OpenCvMemoryManager final : public MemoryManager {
 public:
  explicit OpenCvMemoryManager(Device device) : device_(device) {}

  Device device() const override { return device_; }
  Result<std::shared_ptr<Buffer>> Allocate(size_t bytes,
                                           size_t alignment) override;
  Status Copy(Buffer* dst, size_t dst_offset, const Buffer* src,
              size_t src_offset, size_t bytes, Stream* stream) override;
  MemoryStats Stats() const override;

 private:
  Device device_;
  std::shared_ptr<OpenCvAllocatorState> state_ =
      std::make_shared<OpenCvAllocatorState>();
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_OPENCV_OPENCV_MEMORY_MANAGER_H_
