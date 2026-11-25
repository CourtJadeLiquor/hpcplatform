#ifndef hpcplatform_CORE_STREAM_H_
#define hpcplatform_CORE_STREAM_H_

#include "hpcplatform/core/common.h"
#include "hpcplatform/core/error.h"

namespace hpcplatform
{

    // Execution stream abstraction. CPU backends are synchronous and report
    // synchronous() == true; CUDA/NPU backends wrap their native queues.
    class Stream
    {
    public:
        virtual ~Stream() = default;

        // Returns the device that owns the stream.
        virtual Device device() const = 0;
        // Waits for work on this stream to complete.
        virtual Status Synchronize() = 0;
        // Returns true for synchronous backends such as CPU.
        virtual bool synchronous() const { return true; }
    };

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_STREAM_H_
