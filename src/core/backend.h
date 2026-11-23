#ifndef hpcplatform_SRC_CORE_BACKEND_H_
#define hpcplatform_SRC_CORE_BACKEND_H_

#include <memory>
#include <string>

#include "hpcplatform/core/device.h"
#include "hpcplatform/core/dtype.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/core/memory.h"
#include "hpcplatform/core/stream.h"

// Core-internal contract: every Context is bound to one Backend, and the
// backends layer supplies the implementations. Keeping the interface in core
// (instead of src/backends) makes the dependency direction one-way:
// backends depend on core, never the reverse. Operators and tests that need
// backend capabilities include this header directly; it is never exposed
// through the public API (docs/architecture.md §6.2).
namespace hpcplatform {

// A Backend owns the per-device memory manager, the default stream and any
// backend-private state (OpenCV allocator, CUDA context, NPU handle).
class Backend {
 public:
  virtual ~Backend() = default;

  // Returns the backend type and device, plus its memory manager and default
  // stream.
  virtual BackendType type() const = 0;
  virtual Device device() const = 0;
  virtual MemoryManager& memory() = 0;
  virtual Stream& stream() const = 0;
  // Waits for all work on this backend to complete.
  virtual Status Synchronize() = 0;

  // Copies raw bytes within memory visible to the backend; returns kUnsupported
  // when device memory is inaccessible. Never stages via the host implicitly,
  // which would obscure synchronization/performance semantics. `stream` is
  // valid only for the CUDA backend.
  virtual Status CopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                            size_t src_offset, size_t bytes,
                            Stream* stream) const;

  // Returns whether this backend supports the specified dtype.
  virtual bool SupportsDType(DataType dtype) const {
    return DTypeSize(dtype) != 0;
  }

  // Returns whether this build/backend supports the CPU kernel level; non-CPU
  // backends support only kNaive.
  virtual bool SupportsKernelLevel(CpuKernelLevel level) const = 0;

  // Returns a human-readable backend and device description for logs and errors.
  virtual std::string Describe() const = 0;
};

// Explicit factory: the caller names the backend, nothing is auto-detected.
Result<std::shared_ptr<Backend>> CreateBackend(BackendType type,
                                               int device_index = 0);

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_CORE_BACKEND_H_
