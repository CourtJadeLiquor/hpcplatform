// Core copy semantics: Backend::CopyBuffer / Backend::CopyStrided default
// paths, ResolveOwningBackend, CopyTensorData and the public transfer
// functions (Copy / Upload / Download). This is library implementation (not a
// test helper): Tensor::Clone and the transfer API are its callers.

#include "hpcplatform/core/transfer.h"

#include <cstring>
#include <string>
#include <vector>

#include "core/backend.h"
#include "core/copy.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {

// host-to-host copy
Status Backend::CopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                           size_t src_offset, size_t bytes,
                           Stream* stream) const {
  (void)stream;
  if (dst == nullptr || src == nullptr) {
    return Status::InvalidArgument("copy requires non-null buffers");
  }
  if (dst_offset > dst->size() || bytes > dst->size() - dst_offset ||
      src_offset > src->size() || bytes > src->size() - src_offset) {
    return Status::InvalidArgument("copy range exceeds its buffer");
  }
  if (dst->kind() == MemoryKind::kDevice ||
      src->kind() == MemoryKind::kDevice) {
    return Status::Unsupported(std::string("backend ") +
                               BackendTypeName(type()) +
                               " cannot copy device memory");
  }
  // Host-to-host copy. maybe overlap exists between dst and src.
  std::memmove(static_cast<char*>(dst->data()) + dst_offset,
               static_cast<const char*>(src->data()) + src_offset, bytes);
  return Status::Ok();
}

// Portable default: an element-wise host loop over the two strided views.
// Backends with native strided movement (a CUDA kernel, pitched memcpys, ...)
// override this; host backends inheriting the loop is acceptable because
// element-wise host copies are not a performance cliff.
Status Backend::CopyStrided(Buffer* dst, size_t dst_offset,
                            const int64_t* dst_strides, const Buffer* src,
                            size_t src_offset, const int64_t* src_strides,
                            const Shape& shape, size_t element_size,
                            Stream* stream) const {
  (void)stream;  // Host copies are synchronous; the stream is unused.
  if (dst == nullptr || src == nullptr) {
    return Status::InvalidArgument("CopyStrided requires non-null buffers");
  }
  const int rank = shape.Rank();
  if (rank <= 0 || element_size == 0) {
    return Status::InvalidArgument("CopyStrided requires a non-empty layout");
  }
  if (dst->kind() == MemoryKind::kDevice ||
      src->kind() == MemoryKind::kDevice) {
    return Status::Unsupported(std::string("backend ") +
                               BackendTypeName(type()) +
                               " cannot copy device memory");
  }
  // Both views must stay inside their buffers: the last element starts at the
  // sum of (extent - 1) * stride and spans element_size bytes.
  size_t src_reach = src_offset;
  size_t dst_reach = dst_offset;
  for (int dim = 0; dim < rank; ++dim) {
    if (shape.Dim(dim) <= 0) {
      return Status::InvalidArgument("CopyStrided extents must be positive");
    }
    src_reach += static_cast<size_t>(shape.Dim(dim) - 1) *
                 static_cast<size_t>(src_strides[dim]);
    dst_reach += static_cast<size_t>(shape.Dim(dim) - 1) *
                 static_cast<size_t>(dst_strides[dim]);
  }
  if (src_reach + element_size > src->size() ||
      dst_reach + element_size > dst->size()) {
    return Status::InvalidArgument("strided view exceeds its buffer");
  }

  std::vector<int64_t> coords(static_cast<size_t>(rank), 0);
  char* dst_base = static_cast<char*>(dst->data()) + dst_offset;
  const char* src_base = static_cast<const char*>(src->data()) + src_offset;
  const int64_t numel = shape.Numel();
  for (int64_t linear = 0; linear < numel; ++linear) {
    size_t dst_byte = 0;
    size_t src_byte = 0;
    for (int dim = 0; dim < rank; ++dim) {
      dst_byte += static_cast<size_t>(coords[static_cast<size_t>(dim)]) *
                  static_cast<size_t>(dst_strides[dim]);
      src_byte += static_cast<size_t>(coords[static_cast<size_t>(dim)]) *
                  static_cast<size_t>(src_strides[dim]);
    }
    std::memcpy(dst_base + dst_byte, src_base + src_byte, element_size);
    // Odometer increment: row-major successor of the current coordinates.
    for (int dim = rank - 1; dim >= 0; --dim) {
      const size_t index = static_cast<size_t>(dim);
      if (++coords[index] < shape.Dim(dim)) break;
      coords[index] = 0;
    }
  }
  return Status::Ok();
}

Result<std::shared_ptr<Backend>> ResolveOwningBackend(const Tensor& tensor) {
  auto owner = tensor.owning_backend().lock();
  if (owner != nullptr) return owner;
  return CreateBackend(tensor.backend(), tensor.device().index);
}

// Copies the element data of `src` into a caller-allocated `dst`.
//
// What is copied and what is not:
//   - Only element bytes are copied. `dst` must already be allocated with a
//     shape and dtype that match `src` (validated below); shape / dtype /
//     strides are intentionally NOT copied here — that allocation is the
//     caller's job. Context::Copy expects a pre-built dst, and Tensor::Clone
//     builds the dst (with matching metadata) before calling this function.
//   - Strides may differ between src and dst: non-contiguous operands are
//     delegated to Backend::CopyStrided, so copying from a sliced/transposed
//     view into a contiguous tensor is supported.
//
// Executor selection:
//   The backend that owns the device side drives the transfer; when neither
//   side is device memory the caller's `backend` is used. Core never judges
//   capability — whether a byte range or a strided pair is reachable (e.g.
//   peer-to-peer across devices) is decided by the selected backend.
//
// Execution plan (after validation):
//   1. Both contiguous -> a single CopyBuffer of src.ByteSize() (fast path).
//   2. Anything non-contiguous -> one CopyStrided call; the backend picks the
//      strategy.
Status CopyTensorData(const Backend& backend, Tensor* dst, const Tensor& src,
                      Stream* stream) {
  // --- Input validation -------------------------------------------------
  // Both operands must be bound, and the pre-allocated dst must match src in
  // shape and dtype. Metadata (shape / dtype / strides) is the caller's
  // responsibility; only element bytes are copied below.
  if (dst == nullptr || !dst->defined() || !src.defined()) {
    return Status::InvalidArgument("Copy requires two defined tensors");
  }
  if (dst->shape() != src.shape() || dst->dtype() != src.dtype()) {
    return Status::InvalidArgument("Copy requires matching shape and dtype");
  }

  // --- Executor selection ----------------------------------------------
  // The backend owning a device side drives the transfer; a host-only copy
  // uses the caller's backend. Buffers outlive their Context, so the owning
  // backend is resolved fresh when the original one is gone.
  const bool src_device = src.buffer()->kind() == MemoryKind::kDevice;
  const bool dst_device = dst->buffer()->kind() == MemoryKind::kDevice;

  std::shared_ptr<Backend> src_owner;
  std::shared_ptr<Backend> dst_owner;
  const Backend* executor = &backend;

  if (src_device) {
    const auto resolved = ResolveOwningBackend(src);
    if (!resolved.ok()) return resolved.status();
    src_owner = resolved.value();
    executor = src_owner.get();
  }
  if (dst_device) {
    const auto resolved = ResolveOwningBackend(*dst);
    if (!resolved.ok()) return resolved.status();
    dst_owner = resolved.value();
    executor = dst_owner.get();
  }

  Stream* copy_stream = stream != nullptr ? stream : &executor->stream();

  if (dst->IsContiguous() && src.IsContiguous()) {
    // Both layouts are contiguous, so the whole logical range is a single byte
    // block copied from src.offset() to dst.offset() in backend executor.
    return executor->CopyBuffer(dst->buffer().get(), dst->offset(),
                                src.buffer().get(), src.offset(),
                                src.ByteSize(), copy_stream);
  }

  // At least one operand is non-contiguous. Core only routes: the executor
  // backend owns stride-aware movement via CopyStrided and decides both the
  // strategy (kernel launch, pitched memcpy, host loop) and what it can reach
  // (e.g. peer-to-peer across devices stays a backend-level kUnsupported).
  std::vector<int64_t> dst_strides(static_cast<size_t>(dst->shape().Rank()));
  std::vector<int64_t> src_strides(static_cast<size_t>(src.shape().Rank()));
  const size_t element_size = src.ElementSize();
  for (int dim = 0; dim < dst->shape().Rank(); ++dim) {
    dst_strides[static_cast<size_t>(dim)] =
        dst->Stride(dim) * static_cast<int64_t>(element_size);
  }
  for (int dim = 0; dim < src.shape().Rank(); ++dim) {
    src_strides[static_cast<size_t>(dim)] =
        src.Stride(dim) * static_cast<int64_t>(element_size);
  }
  return executor->CopyStrided(
      dst->buffer().get(), dst->offset(), dst_strides.data(), src.buffer().get(),
      src.offset(), src_strides.data(), src.shape(), element_size,
      copy_stream);
}

Status Copy(const Tensor& src, Tensor* dst, Stream* stream) {
  if (dst == nullptr || !dst->defined() || !src.defined()) {
    return Status::InvalidArgument("Copy requires two defined tensors");
  }
  const auto owner = ResolveOwningBackend(*dst);
  if (!owner.ok()) return owner.status();
  return CopyTensorData(*owner.value(), dst, src, stream);
}

namespace {

bool IsHostBuffer(const Tensor& tensor) {
  return tensor.buffer()->kind() != MemoryKind::kDevice;
}

}  // namespace

Status Upload(const Tensor& host_src, Tensor* device_dst, Stream* stream) {
  if (device_dst == nullptr || !device_dst->defined() || !host_src.defined()) {
    return Status::InvalidArgument("Upload requires two defined tensors");
  }
  if (!IsHostBuffer(host_src)) {
    return Status::InvalidArgument("Upload source must be host memory");
  }
  if (device_dst->buffer()->kind() != MemoryKind::kDevice) {
    return Status::InvalidArgument("Upload destination must be device memory");
  }
  return Copy(host_src, device_dst, stream);
}

Status Download(const Tensor& device_src, Tensor* host_dst, Stream* stream) {
  if (host_dst == nullptr || !host_dst->defined() || !device_src.defined()) {
    return Status::InvalidArgument("Download requires two defined tensors");
  }
  if (device_src.buffer()->kind() != MemoryKind::kDevice) {
    return Status::InvalidArgument("Download source must be device memory");
  }
  if (!IsHostBuffer(*host_dst)) {
    return Status::InvalidArgument("Download destination must be host memory");
  }
  return Copy(device_src, host_dst, stream);
}

}  // namespace hpcplatform
