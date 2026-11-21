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

  virtual BackendType type() const = 0;
  virtual Device device() const = 0;
  virtual MemoryManager& memory() = 0;
  virtual Stream& stream() = 0;
  virtual Status Synchronize() = 0;

  // Transfers raw bytes into a buffer this backend owns (or can reach).
  // Implementations that cannot touch device memory must return kUnsupported
  // rather than staging through the host behind the caller's back, so that a
  // new backend only needs to override this to participate in copies.
  virtual Status CopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                            size_t src_offset, size_t bytes, Stream* stream);

  virtual bool SupportsDType(DataType dtype) const {
    return DTypeSize(dtype) != 0;
  }

  // Whether this build/backend can execute a given CPU kernel level.
  // Non-CPU backends return true for kNaive only.
  virtual bool SupportsKernelLevel(CpuKernelLevel level) const = 0;

  virtual std::string Describe() const = 0;
};

// Explicit factory: the caller names the backend, nothing is auto-detected.
Result<std::shared_ptr<Backend>> CreateBackend(BackendType type,
                                               int device_index = 0);

}  // namespace mycv

#endif  // MYCV_SRC_CORE_BACKEND_H_
