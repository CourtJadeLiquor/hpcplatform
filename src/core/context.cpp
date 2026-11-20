#include "mycv/core/context.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "mycv/core/dtype.h"

namespace mycv {

Result<std::shared_ptr<Context>> Context::Create(BackendType backend,
                                                 int device_index) {
  auto backend_result = CreateBackend(backend, device_index);
  if (!backend_result.ok()) return backend_result.status();
  return std::make_shared<Context>(std::move(backend_result.value()));
}

Result<Tensor> Context::Wrap(void* data, const Shape& shape, DataType dtype,
                             MemoryKind kind, Buffer::Deleter deleter) {
  MYCV_RETURN_IF_ERROR(shape.Validate());
  if (data == nullptr) {
    return Status::InvalidArgument("Wrap requires non-null data");
  }
  const size_t element_size = DTypeSize(dtype);
  if (element_size == 0 || !backend_->SupportsDType(dtype)) {
    return Status::Unsupported(std::string("backend ") +
                               BackendTypeName(backend_->type()) +
                               " does not support dtype: " + DTypeName(dtype));
  }
  const int64_t numel = shape.Numel();
  if (numel <= 0) {
    return Status::InvalidArgument("Wrap requires a non-empty shape");
  }
  const uint64_t count = static_cast<uint64_t>(numel);
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    return Status::InvalidArgument("Wrap byte size overflows size_t");
  }
  const size_t bytes = static_cast<size_t>(count) * element_size;

  auto buffer = std::make_shared<Buffer>(data, bytes, backend_->device(), kind,
                                         std::move(deleter));
  Tensor tensor;
  tensor.buffer_ = std::move(buffer);
  tensor.owner_ = backend_;
  tensor.shape_ = shape;
  tensor.dtype_ = dtype;
  tensor.offset_ = 0;
  tensor.device_ = backend_->device();
  tensor.backend_ = backend_->type();

  int64_t stride = 1;
  for (int i = shape.Rank() - 1; i >= 0; --i) {
    tensor.strides_[static_cast<size_t>(i)] = stride;
    stride *= shape.Dim(i);
  }
  return tensor;
}

}  // namespace mycv
