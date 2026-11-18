#ifndef MYCV_BACKENDS_BACKEND_H_
#define MYCV_BACKENDS_BACKEND_H_

#include <memory>
#include <string>

#include "mycv/core/device.h"
#include "mycv/core/dtype.h"
#include "mycv/core/error.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/memory.h"
#include "mycv/core/stream.h"

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

#endif  // MYCV_BACKENDS_BACKEND_H_
