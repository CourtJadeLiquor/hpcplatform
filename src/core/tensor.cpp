#include "hpcplatform/core/tensor.h"

#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "core/backend.h"
#include "core/copy.h"
#include "hpcplatform/core/context.h"

namespace hpcplatform {
namespace {

Status CheckDefined(const Tensor& tensor) {
  if (!tensor.defined()) return Status::BadState("tensor is not defined");
  return Status::Ok();
}

}  // namespace

Result<Tensor> Tensor::Create(Context& context, const Shape& shape,
                              DataType dtype, size_t alignment) {
  return Create(context.backend_ptr(), shape, dtype, alignment);
}

Result<Tensor> Tensor::Create(const std::shared_ptr<Backend>& backend,
                              const Shape& shape, DataType dtype,
                              size_t alignment) {
  if (backend == nullptr) {
    return Status::InvalidArgument("tensor allocation requires a live backend");
  }
  hpcplatform_RETURN_IF_ERROR(shape.Validate());

  const size_t element_size = DTypeSize(dtype);
  if (element_size == 0 || !backend->SupportsDType(dtype)) {
    return Status::Unsupported(std::string("backend ") +
                               BackendTypeName(backend->type()) +
                               " does not support dtype: " + DTypeName(dtype));
  }

  const int64_t numel = shape.Numel();
  if (numel <= 0) {
    return Status::InvalidArgument("tensor must contain at least one element");
  }

  const uint64_t count = static_cast<uint64_t>(numel);
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    return Status::InvalidArgument("tensor byte size overflows size_t");
  }
  const size_t bytes = static_cast<size_t>(count) * element_size;

  auto buffer_result = backend->memory().Allocate(bytes, alignment);
  if (!buffer_result.ok()) return buffer_result.status();

  Tensor tensor;
  tensor.buffer_ = std::move(buffer_result.value());
  tensor.owner_ = backend;
  tensor.shape_ = shape;
  tensor.dtype_ = dtype;
  tensor.offset_ = 0;
  tensor.device_ = backend->device();
  tensor.backend_ = backend->type();

  int64_t stride = 1;
  for (int i = shape.Rank() - 1; i >= 0; --i) {
    tensor.strides_[static_cast<size_t>(i)] = stride;
    stride *= shape.Dim(i);
  }
  return tensor;
}

Result<Tensor> Tensor::Wrap(const std::shared_ptr<Backend>& backend, void* data,
                            const Shape& shape, DataType dtype, MemoryKind kind,
                            Buffer::Deleter deleter) {
  if (backend == nullptr) {
    return Status::InvalidArgument("tensor wrap requires a live backend");
  }
  hpcplatform_RETURN_IF_ERROR(shape.Validate());
  if (data == nullptr) {
    return Status::InvalidArgument("Wrap requires non-null data");
  }
  const size_t element_size = DTypeSize(dtype);
  if (element_size == 0 || !backend->SupportsDType(dtype)) {
    return Status::Unsupported(std::string("backend ") +
                               BackendTypeName(backend->type()) +
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

  auto buffer = std::make_shared<Buffer>(data, bytes, backend->device(), kind,
                                         std::move(deleter));
  Tensor tensor;
  tensor.buffer_ = std::move(buffer);
  tensor.owner_ = backend;
  tensor.shape_ = shape;
  tensor.dtype_ = dtype;
  tensor.offset_ = 0;
  tensor.device_ = backend->device();
  tensor.backend_ = backend->type();

  int64_t stride = 1;
  for (int i = shape.Rank() - 1; i >= 0; --i) {
    tensor.strides_[static_cast<size_t>(i)] = stride;
    stride *= shape.Dim(i);
  }
  return tensor;
}

Result<Tensor> Tensor::View(const Shape& shape) const {
  hpcplatform_RETURN_IF_ERROR(CheckDefined(*this));
  hpcplatform_RETURN_IF_ERROR(shape.Validate());
  if (!IsContiguous()) {
    return Status::InvalidArgument("View requires a contiguous tensor");
  }
  if (shape.Numel() != Numel()) {
    return Status::InvalidArgument("View must preserve the element count");
  }

  Tensor result = *this;
  result.shape_ = shape;
  int64_t stride = 1;
  for (int i = shape.Rank() - 1; i >= 0; --i) {
    result.strides_[static_cast<size_t>(i)] = stride;
    stride *= shape.Dim(i);
  }
  return result;
}

Result<Tensor> Tensor::Slice(int dim, int64_t begin, int64_t end) const {
  hpcplatform_RETURN_IF_ERROR(CheckDefined(*this));
  if (dim < 0 || dim >= shape_.Rank()) {
    return Status::InvalidArgument("Slice dimension out of range");
  }
  if (begin < 0 || end <= begin || end > shape_.Dim(dim)) {
    return Status::InvalidArgument("Slice requires 0 <= begin < end <= dim");
  }

  Tensor result = *this;
  std::vector<int64_t> dims(shape_.data(), shape_.data() + shape_.Rank());
  dims[static_cast<size_t>(dim)] = end - begin;
  result.shape_ = Shape(dims);
  result.offset_ =
      offset_ + static_cast<size_t>(begin * Stride(dim)) * ElementSize();
  return result;
}

Result<Tensor> Tensor::Permute(const std::vector<int>& order) const {
  hpcplatform_RETURN_IF_ERROR(CheckDefined(*this));
  if (order.size() != static_cast<size_t>(shape_.Rank())) {
    return Status::InvalidArgument("Permute rank must match tensor rank");
  }
  std::vector<bool> seen(order.size(), false);
  std::vector<int64_t> dims(order.size());
  Tensor result = *this;
  for (size_t i = 0; i < order.size(); ++i) {
    const int source_dim = order[i];
    if (source_dim < 0 || source_dim >= shape_.Rank() ||
        seen[static_cast<size_t>(source_dim)]) {
      return Status::InvalidArgument("Permute order is not a permutation");
    }
    seen[static_cast<size_t>(source_dim)] = true;
    dims[i] = shape_.Dim(source_dim);
    result.strides_[i] = strides_[static_cast<size_t>(source_dim)];
  }
  result.shape_ = Shape(dims);
  return result;
}

Result<Tensor> Tensor::Reshape(const Shape& shape) const {
  if (IsContiguous()) return View(shape);
  auto clone = Clone();
  if (!clone.ok()) return clone.status();
  return clone.value().View(shape);
}

Result<Tensor> Tensor::Clone() const {
  hpcplatform_RETURN_IF_ERROR(CheckDefined(*this));
  auto owner = ResolveOwningBackend(*this);
  if (!owner.ok()) return owner.status();
  auto result = Tensor::Create(owner.value(), shape_, dtype_);
  if (!result.ok()) return result.status();
  hpcplatform_RETURN_IF_ERROR(
      CopyTensorData(*owner.value(), &result.value(), *this, nullptr));
  return result;
}

bool Tensor::IsContiguous() const {
  int64_t expected = 1;
  for (int i = shape_.Rank() - 1; i >= 0; --i) {
    if (strides_[static_cast<size_t>(i)] != expected) return false;
    expected *= shape_.Dim(i);
  }
  return true;
}

std::string Tensor::ToString() const {
  std::ostringstream oss;
  oss << "Tensor(shape=" << shape_.ToString() << ", dtype=" << DTypeName(dtype_)
      << ", device=" << device_.ToString()
      << ", backend=" << BackendTypeName(backend_)
      << ", contiguous=" << (IsContiguous() ? "true" : "false") << ")";
  return oss.str();
}

}  // namespace hpcplatform
