#ifndef hpcplatform_CORE_TENSOR_H_
#define hpcplatform_CORE_TENSOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hpcplatform/core/device.h"
#include "hpcplatform/core/dtype.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/memory.h"
#include "hpcplatform/core/shape.h"

namespace hpcplatform {

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

  // Returns whether memory is bound; false for default-constructed or moved-from
  // Tensors.
  bool defined() const { return static_cast<bool>(buffer_); }

  // Returns metadata: shape, dtype, device, and backend.
  const Shape& shape() const { return shape_; }
  DataType dtype() const { return dtype_; }
  Device device() const { return device_; }
  BackendType backend() const { return backend_; }
  // Number of logical elements; 1 for scalars.
  int64_t Numel() const { return shape_.Numel(); }
  // Size of one element in bytes.
  size_t ElementSize() const { return DTypeSize(dtype_); }
  // Size of logical elements in bytes, excluding gaps from view strides.
  size_t ByteSize() const {
    return static_cast<size_t>(Numel()) * ElementSize();
  }

  // Underlying Buffer and view start offset in bytes.
  const std::shared_ptr<Buffer>& buffer() const { return buffer_; }
  size_t offset() const { return offset_; }

  // Returns the first-element pointer including offset; nullptr if no memory is
  // bound.
  void* data() {
    return buffer_ ? static_cast<char*>(buffer_->data()) + offset_ : nullptr;
  }
  const void* data() const {
    return buffer_ ? static_cast<const char*>(buffer_->data()) + offset_
                   : nullptr;
  }

  // Accesses the first element as T; caller must ensure dtype matches T.
  template <typename T>
  T* Data() {
    return reinterpret_cast<T*>(data());
  }
  template <typename T>
  const T* Data() const {
    return reinterpret_cast<const T*>(data());
  }

  // Metadata-only view: shares the Buffer without copying elements.
  Result<Tensor> View(const Shape& shape) const;
  // Slices [begin, end) along dimension dim.
  Result<Tensor> Slice(int dim, int64_t begin, int64_t end) const;
  // Reorders dimensions according to order; any dimension order is supported.
  Result<Tensor> Permute(const std::vector<int>& order) const;

  // Reshapes contiguous tensors with zero copy; clones non-contiguous tensors
  // first.
  Result<Tensor> Reshape(const Shape& shape) const;
  // Deep copy: allocates a new Buffer on the backend, preserving metadata.
  Result<Tensor> Clone() const;
  // Type conversion lives in ops: use hpcplatform::Cast(OpContext, tensor, dtype) so
  // the registered SIMD / OpenCV / CUDA kernels actually run.

  // Backend that allocated this tensor's buffer. Set only by the allocation
  // and wrap factories above; there is deliberately no setter. weak_ptr keeps
  // a tensor from extending its Context's lifetime; when the Context is gone,
  // Clone / copy create a temporary backend via ResolveOwningBackend instead of
  // touching a dead allocator.
  std::weak_ptr<Backend> owning_backend() const { return owner_; }

  // Stride in elements (not bytes).
  int64_t Stride(int dim) const { return strides_[static_cast<size_t>(dim)]; }

  // Checks whether strides describe a contiguous layout in logical order.
  bool IsContiguous() const;
  // Formats shape, dtype, strides, and device for logging.
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

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_TENSOR_H_
