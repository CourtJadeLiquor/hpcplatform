#ifndef MYCV_CORE_TENSOR_ITERATOR_H_
#define MYCV_CORE_TENSOR_ITERATOR_H_

#include <cstddef>
#include <cstdint>

#include "mycv/core/tensor.h"

namespace mycv {

// Iterates logical row-major elements of any strided Tensor. ByteOffset() is
// relative to Tensor::data(), so it also works for sliced views.
class TensorIterator {
 public:
  explicit TensorIterator(const Tensor& tensor) : tensor_(tensor) {}

  bool valid() const { return linear_index_ < tensor_.Numel(); }
  int64_t linear_index() const { return linear_index_; }
  void Reset() { linear_index_ = 0; }
  void Next() { ++linear_index_; }

  size_t ByteOffset() const {
    int64_t remaining = linear_index_;
    size_t offset = 0;
    for (int dim = tensor_.shape().Rank() - 1; dim >= 0; --dim) {
      const int64_t extent = tensor_.shape().Dim(dim);
      const int64_t coordinate = remaining % extent;
      remaining /= extent;
      offset += static_cast<size_t>(coordinate * tensor_.Stride(dim)) *
                tensor_.ElementSize();
    }
    return offset;
  }

  const void* data() const {
    return static_cast<const char*>(tensor_.data()) + ByteOffset();
  }

 private:
  const Tensor& tensor_;
  int64_t linear_index_ = 0;
};

}  // namespace mycv

#endif  // MYCV_CORE_TENSOR_ITERATOR_H_
