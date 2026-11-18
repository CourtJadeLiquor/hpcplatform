#include "mycv/core/context.h"

#include <cstring>

#include "backends/backends_internal.h"
#include "mycv/core/tensor_iterator.h"

namespace mycv {
namespace {

Status CopyBufferBytes(Buffer* dst, size_t dst_offset, const Buffer* src,
                       size_t src_offset, size_t bytes, Stream* stream) {
  if (dst == nullptr || src == nullptr) {
    return Status::InvalidArgument("copy requires non-null buffers");
  }
  if (dst_offset > dst->size() || bytes > dst->size() - dst_offset ||
      src_offset > src->size() || bytes > src->size() - src_offset) {
    return Status::InvalidArgument("tensor copy range exceeds its buffer");
  }

  const bool device_copy = dst->kind() == MemoryKind::kDevice ||
                           src->kind() == MemoryKind::kDevice;
  if (!device_copy) {
    std::memmove(static_cast<char*>(dst->data()) + dst_offset,
                 static_cast<const char*>(src->data()) + src_offset, bytes);
    return Status::Ok();
  }

#ifdef MYCV_WITH_CUDA
  return CudaCopyBuffer(dst, dst_offset, src, src_offset, bytes, stream);
#else
  (void)stream;
  return Status::Unsupported("device copy requires a CUDA-enabled build");
#endif
}

}  // namespace

Status Context::Copy(Tensor* dst, const Tensor& src, Stream* stream) {
  if (dst == nullptr || !dst->defined() || !src.defined()) {
    return Status::InvalidArgument("Copy requires two defined tensors");
  }
  if (dst->shape() != src.shape() || dst->dtype() != src.dtype()) {
    return Status::InvalidArgument("Copy requires matching shape and dtype");
  }
  if (dst->backend() != backend_type() || dst->device().type != device().type ||
      dst->device().index != device().index) {
    return Status::InvalidArgument(
        "destination tensor must belong to this Context and device");
  }

  Stream* copy_stream = stream != nullptr ? stream : &this->stream();
  if (dst->IsContiguous() && src.IsContiguous()) {
    return CopyBufferBytes(dst->buffer().get(), dst->offset(),
                           src.buffer().get(), src.offset(), src.ByteSize(),
                           copy_stream);
  }

  TensorIterator dst_iterator(*dst);
  TensorIterator src_iterator(src);
  const size_t element_size = src.ElementSize();
  while (src_iterator.valid()) {
    MYCV_RETURN_IF_ERROR(CopyBufferBytes(
        dst->buffer().get(), dst->offset() + dst_iterator.ByteOffset(),
        src.buffer().get(), src.offset() + src_iterator.ByteOffset(),
        element_size, copy_stream));
    dst_iterator.Next();
    src_iterator.Next();
  }
  return Status::Ok();
}

}  // namespace mycv
