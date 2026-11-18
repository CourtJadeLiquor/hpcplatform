#ifndef MYCV_CORE_TENSOR_H_
#define MYCV_CORE_TENSOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mycv/core/device.h"
#include "mycv/core/dtype.h"
#include "mycv/core/error.h"
#include "mycv/core/memory.h"
#include "mycv/core/shape.h"

namespace mycv {

class Context;

// Tensor is a descriptor: shape + dtype + strides + a shared Buffer.
// Metadata-only views (crop / reshape / transpose) reuse the same Buffer and
// are introduced in P1.
class Tensor {
 public:
  Tensor() = default;

  static Result<Tensor> Create(Context& context, const Shape& shape,
                               DataType dtype, size_t alignment = 64);

  bool defined() const { return static_cast<bool>(buffer_); }

  const Shape& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  Device device() const { return device_; }
  BackendType backend() const { return backend_; }
  int64_t Numel() const { return shape_.Numel(); }
  size_t ElementSize() const { return DTypeSize(dtype_); }
  size_t ByteSize() const {
    return static_cast<size_t>(Numel()) * ElementSize();
  }

  const std::shared_ptr<Buffer>& buffer() const { return buffer_; }
  size_t offset() const { return offset_; }

  void* data() {
    return buffer_ ? static_cast<char*>(buffer_->data()) + offset_ : nullptr;
  }
  const void* data() const {
    return buffer_ ? static_cast<const char*>(buffer_->data()) + offset_
                   : nullptr;
  }

  template <typename T>
  T* Data() {
    return reinterpret_cast<T*>(data());
  }
  template <typename T>
  const T* Data() const {
    return reinterpret_cast<const T*>(data());
  }

  // Metadata-only views share the same Buffer and never copy elements.
  Result<Tensor> View(const Shape& shape) const;
  Result<Tensor> Slice(int dim, int64_t begin, int64_t end) const;
  Result<Tensor> Permute(const std::vector<int>& order) const;

  // Reshape is zero-copy for contiguous tensors and clones otherwise.
  Result<Tensor> Reshape(const Shape& shape) const;
  Result<Tensor> Clone() const;
  Result<Tensor> Cast(DataType dst_dtype) const;

  // Stride in elements (not bytes).
  int64_t Stride(int dim) const { return strides_[static_cast<size_t>(dim)]; }

  bool IsContiguous() const;
  std::string ToString() const;

 private:
  std::shared_ptr<Buffer> buffer_;
  Shape shape_;
  DataType dtype_ = DataType::kFloat32;
  std::array<int64_t, Shape::kMaxDims> strides_{};
  size_t offset_ = 0;
  Device device_;
  BackendType backend_ = BackendType::kNativeCpu;
};

}  // namespace mycv

#endif  // MYCV_CORE_TENSOR_H_
