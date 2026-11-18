#include "mycv/core/tensor.h"

#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "core/value_cast.h"
#include "mycv/core/context.h"
#include "mycv/core/tensor_iterator.h"

namespace mycv {
namespace {

Result<std::shared_ptr<Context>> ContextForTensor(const Tensor& tensor) {
  return Context::Create(tensor.backend(), tensor.device().index);
}

Status CheckDefined(const Tensor& tensor) {
  if (!tensor.defined()) return Status::BadState("tensor is not defined");
  return Status::Ok();
}

}  // namespace

Result<Tensor> Tensor::Create(Context& context, const Shape& shape,
                              DataType dtype, size_t alignment) {
  MYCV_RETURN_IF_ERROR(shape.Validate());

  const size_t element_size = DTypeSize(dtype);
  if (element_size == 0 || !context.backend().SupportsDType(dtype)) {
    return Status::Unsupported(std::string("backend ") +
                               BackendTypeName(context.backend_type()) +
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

  auto buffer_result = context.memory().Allocate(bytes, alignment);
  if (!buffer_result.ok()) return buffer_result.status();

  Tensor tensor;
  tensor.buffer_ = std::move(buffer_result.value());
  tensor.shape_ = shape;
  tensor.dtype_ = dtype;
  tensor.offset_ = 0;
  tensor.device_ = context.device();
  tensor.backend_ = context.backend_type();

  int64_t stride = 1;
  for (int i = shape.Rank() - 1; i >= 0; --i) {
    tensor.strides_[static_cast<size_t>(i)] = stride;
    stride *= shape.Dim(i);
  }
  return tensor;
}

Result<Tensor> Tensor::View(const Shape& shape) const {
  MYCV_RETURN_IF_ERROR(CheckDefined(*this));
  MYCV_RETURN_IF_ERROR(shape.Validate());
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
  MYCV_RETURN_IF_ERROR(CheckDefined(*this));
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
  result.offset_ = offset_ + static_cast<size_t>(begin * Stride(dim)) *
                                ElementSize();
  return result;
}

Result<Tensor> Tensor::Permute(const std::vector<int>& order) const {
  MYCV_RETURN_IF_ERROR(CheckDefined(*this));
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
  MYCV_RETURN_IF_ERROR(CheckDefined(*this));
  auto context = ContextForTensor(*this);
  if (!context.ok()) return context.status();
  auto result = Tensor::Create(*context.value(), shape_, dtype_);
  if (!result.ok()) return result.status();
  MYCV_RETURN_IF_ERROR(context.value()->Copy(&result.value(), *this));
  return result;
}

Result<Tensor> Tensor::Cast(DataType dst_dtype) const {
  MYCV_RETURN_IF_ERROR(CheckDefined(*this));
  if (DTypeSize(dst_dtype) == 0) {
    return Status::Unsupported("unsupported destination dtype");
  }
  if (dst_dtype == dtype_) return Clone();

  auto destination_context = ContextForTensor(*this);
  if (!destination_context.ok()) return destination_context.status();
  auto destination = Tensor::Create(*destination_context.value(), shape_, dst_dtype);
  if (!destination.ok()) return destination.status();

  if (device_.type == DeviceType::kCPU) {
    TensorIterator source(*this);
    char* output = static_cast<char*>(destination.value().data());
    for (int64_t i = 0; i < Numel(); ++i, source.Next()) {
      detail::CastElement(source.data(), dtype_,
                          output + static_cast<size_t>(i) * DTypeSize(dst_dtype),
                          dst_dtype);
    }
    return destination;
  }

  // Device values are staged explicitly through host memory. The copy layer
  // owns the backend-specific transfer and keeps core free of CUDA headers.
  auto host_backend = CreateBackend(BackendType::kNativeCpu, 0);
  if (!host_backend.ok()) return host_backend.status();
  Context host_context(host_backend.value());
  auto host_source = Tensor::Create(host_context, shape_, dtype_);
  if (!host_source.ok()) return host_source.status();
  MYCV_RETURN_IF_ERROR(host_context.Copy(&host_source.value(), *this));
  auto host_destination = Tensor::Create(host_context, shape_, dst_dtype);
  if (!host_destination.ok()) return host_destination.status();

  TensorIterator source(host_source.value());
  char* output = static_cast<char*>(host_destination.value().data());
  for (int64_t i = 0; i < Numel(); ++i, source.Next()) {
    detail::CastElement(source.data(), dtype_,
                        output + static_cast<size_t>(i) * DTypeSize(dst_dtype),
                        dst_dtype);
  }
  MYCV_RETURN_IF_ERROR(destination_context.value()->Copy(&destination.value(),
                                                         host_destination.value()));
  return destination;
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

}  // namespace mycv
