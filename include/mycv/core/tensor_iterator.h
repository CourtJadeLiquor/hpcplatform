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

  // 线性下标是否仍指向有效元素。
  bool valid() const { return linear_index_ < tensor_.Numel(); }
  // 当前逻辑元素的行主序线性下标。
  int64_t linear_index() const { return linear_index_; }
  // 回到第一个元素。
  void Reset() { linear_index_ = 0; }
  // 前进到下一个逻辑元素。
  void Next() { ++linear_index_; }

  // 当前元素相对 Tensor::data() 的字节偏移，支持非连续视图。
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

  // 当前元素的只读裸指针。
  const void* data() const {
    return static_cast<const char*>(tensor_.data()) + ByteOffset();
  }

 private:
  const Tensor& tensor_;
  int64_t linear_index_ = 0;
};

}  // namespace mycv

#endif  // MYCV_CORE_TENSOR_ITERATOR_H_
