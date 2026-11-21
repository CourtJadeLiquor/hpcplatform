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

class Backend;
class Context;

// Tensor is a descriptor: shape + dtype + strides + a shared Buffer.
// Metadata-only views (crop / reshape / transpose) reuse the same Buffer and
// never copy elements.
class Tensor {
 public:
  Tensor() = default;

  static Result<Tensor> Create(Context& context, const Shape& shape,
                               DataType dtype, size_t alignment = 64);

  // Allocates on an explicit backend, keeping the allocation inside the same
  // resource domain as the tensors it is derived from.
  static Result<Tensor> Create(const std::shared_ptr<Backend>& backend,
                               const Shape& shape, DataType dtype,
                               size_t alignment = 64);

  // Wraps externally owned memory without allocating or copying. `data` must
  // hold at least shape.Numel() * DTypeSize(dtype) bytes. When `deleter` is
  // empty the Tensor borrows the memory (caller keeps it alive); otherwise the
  // deleter runs when the last reference dies, so memory from another runtime
  // (cv::Mat, cudaMalloc, NPU, ...) can be adopted.
  static Result<Tensor> Wrap(const std::shared_ptr<Backend>& backend,
                             void* data, const Shape& shape, DataType dtype,
                             MemoryKind kind = MemoryKind::kHost,
                             Buffer::Deleter deleter = nullptr);

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
  // Type conversion lives in ops: use mycv::Cast(OpContext, tensor, dtype) so
  // the registered SIMD / OpenCV / CUDA kernels actually run.

  // Backend that allocated this tensor's buffer. Set only by the allocation
  // and wrap factories above; there is deliberately no setter. weak_ptr keeps
  // a tensor from extending its Context's lifetime; when the Context is gone,
  // Clone / copy create a temporary backend via ResolveOwningBackend instead of
  // touching a dead allocator.
  std::weak_ptr<Backend> owning_backend() const { return owner_; }

  // Stride in elements (not bytes).
  int64_t Stride(int dim) const { return strides_[static_cast<size_t>(dim)]; }

  bool IsContiguous() const;
  std::string ToString() const;

 private:
  std::shared_ptr<Buffer> buffer_;
  // Non-owning reference to the allocating backend; written only by Create and
  // Wrap, which are Tensor members, so no friend declaration is needed.
  std::weak_ptr<Backend> owner_;
  Shape shape_;
  DataType dtype_ = DataType::kFloat32;
  std::array<int64_t, Shape::kMaxDims> strides_{};
  size_t offset_ = 0;
  Device device_;
  BackendType backend_ = BackendType::kNativeCpu;
};

}  // namespace mycv

#endif  // MYCV_CORE_TENSOR_H_
