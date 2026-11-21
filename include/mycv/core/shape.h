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

  // 最多保留 kMaxDims 维；超出的维度会被截断，调用方应避免传入过长列表。
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

  // 从 vector 构造，同样最多保留 kMaxDims 维。
  explicit Shape(const std::vector<int64_t>& dims) {
    rank_ = static_cast<int>(dims.size() < static_cast<size_t>(kMaxDims)
                                 ? dims.size()
                                 : static_cast<size_t>(kMaxDims));
    for (int i = 0; i < rank_; ++i) {
      dims_[static_cast<size_t>(i)] = dims[static_cast<size_t>(i)];
    }
  }

  // 返回维度数。
  int Rank() const { return rank_; }
  // 返回第 index 维长度；调用方需保证 0 <= index < Rank()。
  int64_t Dim(int index) const { return dims_[static_cast<size_t>(index)]; }
  // 返回连续存放的维度数组，长度等于 Rank()。
  const int64_t* data() const { return dims_.data(); }

  // Scalar (rank 0) has exactly one element.
  int64_t Numel() const {
    int64_t numel = 1;
    for (int i = 0; i < rank_; ++i)
      numel *= dims_[static_cast<size_t>(i)];
    return numel;
  }

  // 标量（rank 0）返回 true。
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

  // 逐维比较 rank 与每个维度长度。
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

  // 格式化为 "[d0,d1,...]"，用于错误信息与日志。
  std::string ToString() const;

 private:
  std::array<int64_t, kMaxDims> dims_{};
  int rank_ = 0;
};

}  // namespace mycv

#endif  // MYCV_CORE_SHAPE_H_
