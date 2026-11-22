#ifndef MYCV_CORE_STREAM_H_
#define MYCV_CORE_STREAM_H_

#include "mycv/core/device.h"
#include "mycv/core/error.h"

namespace mycv {

// Execution stream abstraction. CPU backends are synchronous and report
// synchronous() == true; CUDA/NPU backends wrap their native queues.
class Stream {
 public:
  virtual ~Stream() = default;

  // Returns the device that owns the stream.
  virtual Device device() const = 0;
  // Waits for work on this stream to complete.
  virtual Status Synchronize() = 0;
  // Returns true for synchronous backends such as CPU.
  virtual bool synchronous() const { return true; }
};

}  // namespace mycv

#endif  // MYCV_CORE_STREAM_H_
