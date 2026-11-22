// Core copy semantics: Backend::CopyBuffer default path, ResolveOwningBackend,
// CopyTensorData and Context::Copy. This is library implementation (not a test
// helper): Context::Copy and Tensor::Clone are its two callers.

#include "mycv/core/context.h"

#include <cstring>
#include <string>

#include "core/backend.h"
#include "core/copy.h"
#include "core/tensor_iterator.h"

namespace mycv {

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
//   - Strides may differ between src and dst: the slow path reads each logical
//     element from src's layout and writes it to dst's layout, so copying from
//     a sliced/transposed view into a contiguous tensor is supported.
//
// Executor selection:
//   The backend that owns the device side drives the transfer; when neither
//   side is device memory the caller's `backend` is used. Device-to-device
//   copies across two different devices are rejected, and non-contiguous
//   device copies are rejected (materialize the view with Clone() first).
//
// Execution plan (after validation):
//   1. Both contiguous -> a single CopyBuffer of src.ByteSize() (fast path).
//   2. Device + non-contiguous -> kUnsupported.
//   3. Host + non-contiguous -> TensorIterator walks the logical elements and
//      issues one CopyBuffer per element (slow path, stride-aware).
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
    // Device-to-device across two different devices has no staging path yet.
    if (src_device && dst_owner != src_owner) {
      return Status::Unsupported(
          "peer-to-peer copies between different devices are not implemented");
    }
    executor = dst_owner.get();
  }

  Stream* copy_stream = stream != nullptr ? stream : &executor->stream();

  if (dst->IsContiguous() && src.IsContiguous()) {
    // Branch 1: both layouts are contiguous, so the whole logical range is a
    // single byte block copied from src.offset() to dst.offset() in backend
    // executor.
    return executor->CopyBuffer(dst->buffer().get(), dst->offset(),
                                src.buffer().get(), src.offset(),
                                src.ByteSize(), copy_stream);
  } else if (src_device || dst_device) {
    // Branch 2: at least one side is device memory AND the tensors are not
    // both contiguous (Branch 1 already handled the contiguous case).
    // Element-wise device addressing needs an explicit staging buffer, which
    // is not provided here; require the caller to materialize the view with
    // Clone() first.
    return Status::Unsupported(
        "non-contiguous device copies are not supported; materialize the view "
        "with Clone() first");
  } else {
    // Branch 3: host-only copy where at least one tensor is non-contiguous.
    // TensorIterator walks the logical row-major elements of each tensor
    // independently, so src and dst may have different strides (e.g. a view
    // copied into a fresh contiguous tensor); one CopyBuffer per element.
    TensorIterator dst_iterator(*dst);
    TensorIterator src_iterator(src);
    const size_t element_size = src.ElementSize();
    while (src_iterator.valid()) {
      MYCV_RETURN_IF_ERROR(executor->CopyBuffer(
          dst->buffer().get(), dst->offset() + dst_iterator.ByteOffset(),
          src.buffer().get(), src.offset() + src_iterator.ByteOffset(),
          element_size, copy_stream));
      dst_iterator.Next();
      src_iterator.Next();
    }
    return Status::Ok();
  }
}

Status Context::Copy(Tensor* dst, const Tensor& src, Stream* stream) {
  return CopyTensorData(*backend_, dst, src, stream);
}

}  // namespace mycv
