#ifndef hpcplatform_SRC_BACKENDS_OPENCV_OPENCV_MEMORY_MANAGER_H_
#define hpcplatform_SRC_BACKENDS_OPENCV_OPENCV_MEMORY_MANAGER_H_

#include <atomic>
#include <memory>

#include "hpcplatform/core/memory.h"

namespace hpcplatform
{

    struct OpenCvAllocatorState
    {
        std::atomic<size_t> allocated_bytes{0};
        std::atomic<size_t> peak_bytes{0};
        std::atomic<size_t> allocation_count{0};
    };

    class OpenCvMemoryManager final : public MemoryManager
    {
    public:
        explicit OpenCvMemoryManager(Device device) : device_(device) {}

        Device device() const override { return device_; }
        Result<std::shared_ptr<Buffer>> Allocate(size_t bytes, size_t alignment) override;
        MemoryStats Stats() const override;

    private:
        Device device_;
        std::shared_ptr<OpenCvAllocatorState> state_ = std::make_shared<OpenCvAllocatorState>();
    };

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_OPENCV_OPENCV_MEMORY_MANAGER_H_
