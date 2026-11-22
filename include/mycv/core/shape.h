#ifndef MYCV_CORE_SHAPE_H_
#define MYCV_CORE_SHAPE_H_

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "mycv/core/error.h"

namespace mycv {

// Fixed-capacity shape descriptor (no heap allocation in hot paths).
// Precondition: no more than kMaxDims dimensions are passed to the constructor.
class Shape {
 public:
  static constexpr int kMaxDims = 8;

  Shape() = default;

  // Keeps at most kMaxDims dimensions and truncates the rest; callers should
  // avoid longer lists.
  Shape(std::initializer_list<int64_t> dims) {
    rank_ = static_cast<int>(dims.size() < static_cast<size_t>(kMaxDims)
                                 ? dims.size()
                                 : static_cast<size_t>(kMaxDims));
    int i = 0;
    for (int64_t dim : dims) {
      if (i >= rank_) break;
      dims_[static_cast<size_t>(i++)] = dim;
    }
  }

  // Constructs from a vector, likewise keeping at most kMaxDims dimensions.
  explicit Shape(const std::vector<int64_t>& dims) {
    rank_ = static_cast<int>(dims.size() < static_cast<size_t>(kMaxDims)
                                 ? dims.size()
                                 : static_cast<size_t>(kMaxDims));
    for (int i = 0; i < rank_; ++i) {
      dims_[static_cast<size_t>(i)] = dims[static_cast<size_t>(i)];
    }
  }

  // Returns the number of dimensions.
  int Rank() const { return rank_; }
  // Returns the length of dimension index; caller must ensure
  // 0 <= index < Rank().
  int64_t Dim(int index) const { return dims_[static_cast<size_t>(index)]; }
  // Returns the contiguous dimension array, whose length is Rank().
  const int64_t* data() const { return dims_.data(); }

  // Scalar (rank 0) has exactly one element.
  int64_t Numel() const {
    int64_t numel = 1;
    for (int i = 0; i < rank_; ++i)
      numel *= dims_[static_cast<size_t>(i)];
    return numel;
  }

  // Returns true for scalars (rank 0).
  bool IsScalar() const { return rank_ == 0; }

  // Semantic validation (dims must be positive). Construction itself is
  // unchecked so Shape stays trivially cheap inside kernels.
  Status Validate() const {
    for (int i = 0; i < rank_; ++i) {
      if (dims_[static_cast<size_t>(i)] <= 0) {
        return Status::InvalidArgument("shape dims must be positive: " +
                                       ToString());
      }
    }
    return Status::Ok();
  }

  // Compares rank and every dimension length.
  bool operator==(const Shape& other) const {
    if (rank_ != other.rank_) return false;
    for (int i = 0; i < rank_; ++i) {
      if (dims_[static_cast<size_t>(i)] !=
          other.dims_[static_cast<size_t>(i)]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  // Formats as "[d0,d1,...]" for error messages and logs.
  std::string ToString() const;

 private:
  std::array<int64_t, kMaxDims> dims_{};
  int rank_ = 0;
};

}  // namespace mycv

#endif  // MYCV_CORE_SHAPE_H_
