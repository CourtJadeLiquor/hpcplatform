#ifndef MYCV_SRC_CORE_BACKEND_H_
#define MYCV_SRC_CORE_BACKEND_H_

#include <memory>
#include <string>

#include "mycv/core/device.h"
#include "mycv/core/dtype.h"
#include "mycv/core/error.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/memory.h"
#include "mycv/core/stream.h"

// Core-internal contract: every Context is bound to one Backend, and the
// backends layer supplies the implementations. Keeping the interface in core
// (instead of src/backends) makes the dependency direction one-way:
// backends depend on core, never the reverse. Operators and tests that need
// backend capabilities include this header directly; it is never exposed
// through the public API (docs/architecture.md §6.2).
namespace mycv {

// A Backend owns the per-device memory manager, the default stream and any
// backend-private state (OpenCV allocator, CUDA context, NPU handle).
class Backend {
 public:
  virtual ~Backend() = default;

  // 返回后端类型、设备，以及该后端持有的内存管理器与默认流。
  virtual BackendType type() const = 0;
  virtual Device device() const = 0;
  virtual MemoryManager& memory() = 0;
  virtual Stream& stream() = 0;
  // 等待该后端上的全部工作完成。
  virtual Status Synchronize() = 0;

  // 在后端可见范围内搬运原始字节；无法直接访问设备内存时返回 kUnsupported，
  // 不允许暗中经由主机中转，避免调用方误判同步/性能语义。
  virtual Status CopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                            size_t src_offset, size_t bytes, Stream* stream);

  // 该后端是否支持指定 dtype。
  virtual bool SupportsDType(DataType dtype) const {
    return DTypeSize(dtype) != 0;
  }

  // 该构建/后端能否执行指定 CPU 内核级别；非 CPU 后端只对 kNaive 返回 true。
  virtual bool SupportsKernelLevel(CpuKernelLevel level) const = 0;

  // 返回后端与设备的人类可读描述，用于日志与错误信息。
  virtual std::string Describe() const = 0;
};

// Explicit factory: the caller names the backend, nothing is auto-detected.
Result<std::shared_ptr<Backend>> CreateBackend(BackendType type,
                                               int device_index = 0);

}  // namespace mycv

#endif  // MYCV_SRC_CORE_BACKEND_H_
