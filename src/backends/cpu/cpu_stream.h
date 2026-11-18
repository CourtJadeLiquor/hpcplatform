#ifndef MYCV_SRC_BACKENDS_CPU_CPU_STREAM_H_
#define MYCV_SRC_BACKENDS_CPU_CPU_STREAM_H_

#include "mycv/core/stream.h"

namespace mycv {

// CPU execution is synchronous: the stream exists to keep one uniform API
// across backends, not to provide concurrency.
class CpuStream final : public Stream {
 public:
  explicit CpuStream(Device device) : device_(device) {}

  Device device() const override { return device_; }
  Status Synchronize() override { return Status::Ok(); }
  bool synchronous() const override { return true; }

 private:
  Device device_;
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_CPU_CPU_STREAM_H_
