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

  // 返回流所属设备。
  virtual Device device() const = 0;
  // 等待该流上的工作完成。
  virtual Status Synchronize() = 0;
  // CPU 等同步后端返回 true。
  virtual bool synchronous() const { return true; }
};

}  // namespace mycv

#endif  // MYCV_CORE_STREAM_H_
