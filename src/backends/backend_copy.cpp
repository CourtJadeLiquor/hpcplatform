#include "mycv/core/context.h"

#include <cstring>
#include <string>

#include "backends/backends_internal.h"
#include "mycv/core/tensor_iterator.h"

namespace mycv {

Status Backend::CopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                           size_t src_offset, size_t bytes, Stream* stream) {
  (void)stream;
  if (dst == nullptr || src == nullptr) {
    return Status::InvalidArgument("copy requires non-null buffers");
  }
  if (dst_offset > dst->size() || bytes > dst->size() - dst_offset ||
      src_offset > src->size() || bytes > src->size() - src_offset) {
    return Status::InvalidArgument("copy range exceeds its buffer");
  }
  if (dst->kind() == MemoryKind::kDevice || src->kind() == MemoryKind::kDevice) {
    return Status::Unsupported(std::string("backend ") + BackendTypeName(type()) +
                               " cannot copy device memory");
  }
  std::memmove(static_cast<char*>(dst->data()) + dst_offset,
               static_cast<const char*>(src->data()) + src_offset, bytes);
  return Status::Ok();
}

Result<std::shared_ptr<Backend>> ResolveOwningBackend(const Tensor& tensor) {
  auto owner = tensor.owning_backend().lock();
  if (owner != nullptr) return owner;
  return CreateBackend(tensor.backend(), tensor.device().index);
}

Status CopyTensor(Backend& backend, Tensor* dst, const Tensor& src,
                  Stream* stream) {
  if (dst == nullptr || !dst->defined() || !src.defined()) {
    return Status::InvalidArgument("Copy requires two defined tensors");
  }
  if (dst->shape() != src.shape() || dst->dtype() != src.dtype()) {
    return Status::InvalidArgument("Copy requires matching shape and dtype");
  }

  // The backend owning the device side drives the transfer; host-to-host copies
  // use the caller's backend. No backend type is special-cased here.
  std::shared_ptr<Backend> src_owner;
  std::shared_ptr<Backend> dst_owner;
  const bool src_device = src.buffer()->kind() == MemoryKind::kDevice;
  const bool dst_device = dst->buffer()->kind() == MemoryKind::kDevice;
  Backend* executor = &backend;
  if (src_device) {
    auto resolved = ResolveOwningBackend(src);
    if (!resolved.ok()) return resolved.status();
    src_owner = resolved.value();
    executor = src_owner.get();
  }
  if (dst_device) {
    auto resolved = ResolveOwningBackend(*dst);
    if (!resolved.ok()) return resolved.status();
    dst_owner = resolved.value();
    if (src_device && dst_owner != src_owner) {
      return Status::Unsupported(
          "peer-to-peer copies between different devices are not implemented");
    }
    executor = dst_owner.get();
  }

  Stream* copy_stream = stream != nullptr ? stream : &executor->stream();
  if (dst->IsContiguous() && src.IsContiguous()) {
    return executor->CopyBuffer(dst->buffer().get(), dst->offset(),
                                src.buffer().get(), src.offset(), src.ByteSize(),
                                copy_stream);
  }
  if (src_device || dst_device) {
    return Status::Unsupported(
        "non-contiguous device copies are not supported; materialize the view "
        "with Clone() first");
  }

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

Status Context::Copy(Tensor* dst, const Tensor& src, Stream* stream) {
  return CopyTensor(*backend_, dst, src, stream);
}

}  // namespace mycv
