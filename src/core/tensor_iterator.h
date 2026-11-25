#ifndef hpcplatform_CORE_TENSOR_ITERATOR_H_
#define hpcplatform_CORE_TENSOR_ITERATOR_H_

#include <cstddef>
#include <cstdint>

#include "hpcplatform/core/tensor.h"

// Core-internal utility: walks the logical row-major elements of any strided
// Tensor. Lives in src/core (not include/hpcplatform) because it is an implementation
// detail of the copy path and future element-wise kernels, not public API.
namespace hpcplatform
{

    // Iterates logical row-major elements of any strided Tensor. ByteOffset() is
    // relative to Tensor::data(), so it also works for sliced views.
    class TensorIterator
    {
    public:
        explicit TensorIterator(const Tensor& tensor) : tensor_(tensor) {}

        // Returns whether the linear index still refers to a valid element.
        bool valid() const { return linear_index_ < tensor_.Numel(); }
        // Row-major linear index of the current logical element.
        int64_t linear_index() const { return linear_index_; }
        // Rewinds to the first element.
        void Reset() { linear_index_ = 0; }
        // Advances to the next logical element.
        void Next() { ++linear_index_; }

        // Byte offset of the current element relative to Tensor::data(); supports
        // non-contiguous views.
        size_t ByteOffset() const
        {
            int64_t remaining = linear_index_;
            size_t offset = 0;
            for (int dim = tensor_.shape().Rank() - 1; dim >= 0; --dim)
            {
                const int64_t extent = tensor_.shape().Dim(dim);
                const int64_t coordinate = remaining % extent;
                remaining /= extent;
                offset +=
                    static_cast<size_t>(coordinate * tensor_.Stride(dim)) * tensor_.ElementSize();
            }
            return offset;
        }

        // Read-only raw pointer to the current element.
        const void* data() const { return static_cast<const char*>(tensor_.data()) + ByteOffset(); }

    private:
        const Tensor& tensor_;
        int64_t linear_index_ = 0;
    };

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_TENSOR_ITERATOR_H_
