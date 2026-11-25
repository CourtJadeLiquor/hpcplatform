#ifndef hpcplatform_CORE_CONTEXT_H_
#define hpcplatform_CORE_CONTEXT_H_

#include <memory>
#include <string>

#include "hpcplatform/core/device.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/core/memory.h"
#include "hpcplatform/core/stream.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {

// The per-device provider interface behind every Context. Internal type: the
// definition lives in src/core/backend.h and is never exposed through the
// public API; Context::Create is the only construction path
// (docs/architecture.md §6.2).
class Backend;

// A handle to one device plus the unified memory manager / stream used to
// allocate and schedule tensors. Deliberately small: allocation lives on
// Tensor::Empty and data movement on the free functions in
// hpcplatform/core/transfer.h, so the public surface describes algorithms rather
// than backend mechanics. memory() / stream() are private: the library owns
// scheduling details and exposes them only to friend types (Tensor, OpContext,
// internal helpers).
class Context {
 public:
  // No fallback: an unavailable backend returns kUnsupported.
  static Result<std::shared_ptr<Context>> Create(BackendType backend,
                                                 int device_index = 0);

  // Returns the bound backend type.
  BackendType backend_type() const;
  // Returns the current device (type + index).
  Device device() const;
  // Returns a human-readable backend and device description for logging.
  std::string Describe() const;

  // Returns this Context's execution stream. Public because stream is a
  // scheduling control — callers that need to run work on a specific stream
  // (CUDA explicit scheduling, profiling, external stream interop) must be
  // able to name it. Memory management (memory()) stays private; callers never
  // operate on raw backend allocators from public code.
  Stream& stream();

  // Waits for all work submitted on this Context to complete.
  Status Synchronize();

  // Returns whether the bound backend accepts the CPU kernel level. Non-CPU
  // backends accept only kNaive; callers query this before picking a level.
  bool SupportsKernelLevel(CpuKernelLevel level) const;

  // Aggregated memory statistics for diagnostics and tests. Only exposes the
  // numbers — not the MemoryManager itself — so callers cannot bypass the
  // public allocation path (Tensor::Empty) or mess with raw backend allocators.
  ::hpcplatform::MemoryStats GetMemoryStats() const;

 private:
  friend class Tensor;
  friend class OpContext;

  explicit Context(std::shared_ptr<Backend> backend)
      : backend_(std::move(backend)) {}

  // Internal accessors; Tensor allocation and lifetime-tests use these.
  const std::shared_ptr<Backend>& backend_ptr() const { return backend_; }
  MemoryManager& memory();
  const MemoryManager& memory() const;

  std::shared_ptr<Backend> backend_;
};

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_CONTEXT_H_
