#ifndef hpcplatform_SRC_FRAMEWORK_IMAGE_UTILS_H_
#define hpcplatform_SRC_FRAMEWORK_IMAGE_UTILS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/ops/image.h"

namespace hpcplatform {
namespace ops_internal {

// Logical image view: n/h/w/c and layout, parsed from Shape.
struct ImageInfo {
  int rank = 0;
  int64_t n = 1;
  int64_t h = 0;
  int64_t w = 0;
  int64_t c = 1;
  ImageLayout layout = ImageLayout::kNHWC;
};

// Parses a rank-2/3/4 Shape into ImageInfo; returns kInvalidArgument for invalid
// shapes.
inline Result<ImageInfo> ParseImage(const Shape& shape, ImageLayout layout) {
  hpcplatform_RETURN_IF_ERROR(shape.Validate());
  const int rank = shape.Rank();
  if (rank < 2 || rank > 4 || (rank == 2 && layout != ImageLayout::kNHWC)) {
    return Status::InvalidArgument(
        "image must be rank-2 HW, rank-3 HWC/CHW, or rank-4 NHWC/NCHW");
  }
  ImageInfo info;
  info.rank = rank;
  info.layout = layout;
  if (rank == 2) {
    info.h = shape.Dim(0);
    info.w = shape.Dim(1);
  } else if (layout == ImageLayout::kNHWC) {
    if (rank == 3) {
      info.h = shape.Dim(0);
      info.w = shape.Dim(1);
      info.c = shape.Dim(2);
    } else {
      info.n = shape.Dim(0);
      info.h = shape.Dim(1);
      info.w = shape.Dim(2);
      info.c = shape.Dim(3);
    }
  } else {
    if (rank == 3) {
      info.c = shape.Dim(0);
      info.h = shape.Dim(1);
      info.w = shape.Dim(2);
    } else {
      info.n = shape.Dim(0);
      info.c = shape.Dim(1);
      info.h = shape.Dim(2);
      info.w = shape.Dim(3);
    }
  }
  return info;
}

// Creates a Shape with the given height/width in the original layout.
inline Shape MakeImageShape(const ImageInfo& info, int64_t height,
                            int64_t width) {
  if (info.rank == 2) return Shape{height, width};
  if (info.rank == 3) {
    if (info.layout == ImageLayout::kNHWC) return Shape{height, width, info.c};
    return Shape{info.c, height, width};
  }
  if (info.layout == ImageLayout::kNHWC) {
    return Shape{info.n, height, width, info.c};
  }
  return Shape{info.n, info.c, height, width};
}

// Returns the byte offset of element (n, y, x, c) relative to Tensor::data();
// supports non-contiguous layouts.
inline size_t ImageByteOffset(const Tensor& tensor, const ImageInfo& info,
                              int64_t n, int64_t y, int64_t x, int64_t c) {
  std::array<int64_t, 4> coordinates{};
  if (info.rank == 2) {
    coordinates = {y, x, 0, 0};
  } else if (info.rank == 3 && info.layout == ImageLayout::kNHWC) {
    coordinates = {y, x, c, 0};
  } else if (info.rank == 3) {
    coordinates = {c, y, x, 0};
  } else if (info.layout == ImageLayout::kNHWC) {
    coordinates = {n, y, x, c};
  } else {
    coordinates = {n, c, y, x};
  }
  size_t offset = 0;
  for (int dim = 0; dim < info.rank; ++dim) {
    offset += static_cast<size_t>(coordinates[static_cast<size_t>(dim)] *
                                  tensor.Stride(dim)) *
              tensor.ElementSize();
  }
  return offset;
}

// Returns the logical linear index of element (n, y, x, c) in a contiguous
// layout.
inline int64_t ImageLinearIndex(const ImageInfo& info, int64_t n, int64_t y,
                                int64_t x, int64_t c) {
  if (info.layout == ImageLayout::kNHWC) {
    return ((n * info.h + y) * info.w + x) * info.c + c;
  }
  return ((n * info.c + c) * info.h + y) * info.w + x;
}

// Data types accepted by image operators: u8/f32/f64.
inline bool IsSupportedImageDType(DataType dtype) {
  return dtype == DataType::kUInt8 || dtype == DataType::kFloat32 ||
         dtype == DataType::kFloat64;
}

}  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_FRAMEWORK_IMAGE_UTILS_H_
