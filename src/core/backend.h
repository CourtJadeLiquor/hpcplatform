#ifndef hpcplatform_SRC_CORE_BACKEND_H_
#define hpcplatform_SRC_CORE_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>

#include "hpcplatform/core/common.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/memory.h"
#include "hpcplatform/core/stream.h"

// Core-internal contract: every Context is bound to one Backend, and the
// backends layer supplies the implementations. Keeping the interface in core
// (instead of src/backends) makes the dependency direction one-way:
// backends depend on core, never the reverse. Operators and tests that need
// backend capabilities include this header directly; it is never exposed
// through the public API (docs/architecture.md §6.2).
namespace hpcplatform
{

    // A Backend owns the per-device memory manager, the default stream and any
    // backend-private state (OpenCV allocator, CUDA context, NPU handle).
    class Backend
    {
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
        virtual Status CopyBuffer(Buffer* dst,
                                  size_t dst_offset,
                                  const Buffer* src,
                                  size_t src_offset,
                                  size_t bytes,
                                  Stream* stream) const;

        // Copies the logical elements of a strided view into another strided view.
        // `shape` gives the element extents in row-major order; the two stride
        // arrays are per-dimension byte strides. Both operands share the element
        // size. This is the backend's decision point for non-contiguous movement —
        // each backend picks its native strategy (a single strided-copy kernel on
        // CUDA, an element-wise loop on host) and returns kUnsupported for memory it
        // cannot access. The default implementation is a portable element-wise host
        // loop; host-side element-wise copies are not a performance cliff, so host
        // backends are free to inherit it.
        virtual Status CopyStrided(Buffer* dst,
                                   size_t dst_offset,
                                   const int64_t* dst_strides,
                                   const Buffer* src,
                                   size_t src_offset,
                                   const int64_t* src_strides,
                                   const Shape& shape,
                                   size_t element_size,
                                   Stream* stream) const;

        // Returns whether this backend supports the specified dtype.
        virtual bool SupportsDType(DataType dtype) const { return DTypeSize(dtype) != 0; }

        // Returns whether this build/backend supports the CPU kernel level; non-CPU
        // backends support only kNaive.
        virtual bool SupportsKernelLevel(CpuKernelLevel level) const = 0;

        // Returns a human-readable backend and device description for logs and errors.
        virtual std::string Describe() const = 0;
    };

    // Explicit factory: the caller names the backend, nothing is auto-detected.
    Result<std::shared_ptr<Backend>> CreateBackend(BackendType type, int device_index = 0);

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_CORE_BACKEND_H_
