#ifndef hpcplatform_SRC_OPS_RESIZE_RESIZE_CPU_H_
#define hpcplatform_SRC_OPS_RESIZE_RESIZE_CPU_H_

// Shared CPU skeleton for resize: the pixel kernel ResizeElement plus three
// outer-loop organizations (serial naive / OpenMP / TBB). Every CPU level calls
// the same kernel and writes each output element independently with the same
// formulas and no accumulation, so all levels are bit-exact.
//
// Algorithm:
//   Nearest: src = floor(dst * scale), then clamp to h-1/w-1;
//   Bilinear: src = (dst + 0.5) * scale - 0.5; clamp to >= 0 and floor for the
//   top-left corner. Out-of-range y1/x1 fall back to h-1/w-1, with weights
//   wy/wx = src - floor(src). Interpolate twice along x, then once along y.
// Accumulate intermediates in long double; round u8 output with std::round
// before saturating on store.

#include <algorithm>
#include <cmath>

#ifdef hpcplatform_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "core/value_cast.h"
#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "resize_internal.h"

namespace hpcplatform {
namespace ops_internal {

// Compute one output element, output[output_linear].
// Inputs: source image input, input/output layouts image/output_info, resize
// parameters options, and flat output index output_linear encoded according to
// output_info.layout. Steps: decode (n,y,x,c) -> compute source coordinates and
// weights -> read 1 or 4 source pixels -> take the nearest value or interpolate
// bilinearly -> store according to dtype. O(1) per element (at most 4 memory
// accesses); O(N*H_out*W_out*C) for the full output.
inline void ResizeElement(const Tensor& input, const ImageInfo& image,
                          const ImageInfo& output_info,
                          const ResizeParams& options, int64_t output_linear,
                          Tensor* output) {
  // Decode the flat index to (n, y, x, c) according to the output layout.
  int64_t n = 0;
  int64_t y = 0;
  int64_t x = 0;
  int64_t c = 0;
  int64_t remaining = output_linear;
  if (output_info.layout == ImageLayout::kNHWC) {
    c = remaining % output_info.c;
    remaining /= output_info.c;
    x = remaining % output_info.w;
    remaining /= output_info.w;
    y = remaining % output_info.h;
    n = remaining / output_info.h;
  } else {
    x = remaining % output_info.w;
    remaining /= output_info.w;
    y = remaining % output_info.h;
    remaining /= output_info.h;
    c = remaining % output_info.c;
    n = remaining / output_info.c;
  }

  // scale = input extent / output extent (< 1 when enlarging, > 1 when shrinking).
  const long double scale_y =
      static_cast<long double>(image.h) / options.height;
  const long double scale_x = static_cast<long double>(image.w) / options.width;
  int64_t y0 = 0;
  int64_t x0 = 0;
  int64_t y1 = 0;
  int64_t x1 = 0;
  long double wy = 0;
  long double wx = 0;
  if (options.interpolation == InterpolationMode::kNearest) {
    // Nearest: align top-left corners, then clamp floor(dst*scale) to the bounds.
    y0 = std::min(image.h - 1, static_cast<int64_t>(y * scale_y));
    x0 = std::min(image.w - 1, static_cast<int64_t>(x * scale_x));
  } else {
    // Bilinear: half-pixel-center mapping; clamp the lower bound to 0, then floor.
    const long double source_y = (y + 0.5L) * scale_y - 0.5L;
    const long double source_x = (x + 0.5L) * scale_x - 0.5L;
    const long double clamped_y = std::max(0.0L, source_y);
    const long double clamped_x = std::max(0.0L, source_x);
    y0 = static_cast<int64_t>(std::floor(clamped_y));
    x0 = static_cast<int64_t>(std::floor(clamped_x));
    // Fall back to the last valid row/column for out-of-range bottom/right neighbors (replicate).
    y1 = std::min(image.h - 1, y0 + 1);
    x1 = std::min(image.w - 1, x0 + 1);
    wy = clamped_y - y0;
    wx = clamped_x - x0;
  }

  // Address the potentially non-contiguous input by strides; promote values to long double.
  const char* source = static_cast<const char*>(input.data());
  long double value = 0;
  if (options.interpolation == InterpolationMode::kNearest) {
    value = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y0, x0, c), input.dtype());
  } else {
    const long double top_left = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y0, x0, c), input.dtype());
    const long double top_right = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y0, x1, c), input.dtype());
    const long double bottom_left = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y1, x0, c), input.dtype());
    const long double bottom_right = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y1, x1, c), input.dtype());
    // Interpolate the top and bottom rows by wx, then blend them by wy.
    const long double top = top_left + (top_right - top_left) * wx;
    const long double bottom = bottom_left + (bottom_right - bottom_left) * wx;
    value = top + (bottom - top) * wy;
  }
  // Round u8 half away from zero, then convert with saturation in Store.
  if (input.dtype() == DataType::kUInt8) value = std::round(value);
  char* destination = static_cast<char*>(output->data());
  detail::StoreFromLongDouble(
      destination + static_cast<size_t>(output_linear) * output->ElementSize(),
      output->dtype(), value);
}

// Parse the input ImageInfo from the input shape, then copy it and replace h/w
// with the target dimensions to obtain the output ImageInfo. Both retain the
// same rank/n/c/layout. Return InvalidArgument for an invalid shape/layout
// combination. O(rank).
inline Status PrepareResize(const Tensor& input, const ResizeParams& options,
                            ImageInfo* image, ImageInfo* output_info) {
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  *image = parsed.value();
  *output_info = parsed.value();
  output_info->h = options.height;
  output_info->w = options.width;
  return Status::Ok();
}

// Naive implementation: single-threaded ResizeElement calls over flat output
// indices [0, Numel()). No parallel-library dependencies; serves as the
// reference for other CPU levels. O(N*H_out*W_out*C).
inline Status ResizeNaiveImpl(const Tensor& input, const ResizeParams& options,
                              Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  hpcplatform_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
  for (int64_t i = 0; i < output->Numel(); ++i) {
    ResizeElement(input, image, output_info, options, i, output);
  }
  return Status::Ok();
}

// OpenMP implementation: parallel for partitions flat output indices at
// single-element granularity. With no schedule clause, the OpenMP runtime
// determines scheduling and thread count (typically static partitioning, with
// OMP_NUM_THREADS and similar controls). Each i writes only its own output
// element with no cross-iteration dependencies and shares naive's pixel kernel,
// so results are bit-exact. O(N*H_out*W_out*C), distributed across threads.
inline Status ResizeOpenMPImpl(const Tensor& input, const ResizeParams& options,
                               Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  hpcplatform_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
#pragma omp parallel for
  for (int64_t i = 0; i < output->Numel(); ++i) {
    ResizeElement(input, image, output_info, options, i, output);
  }
  return Status::Ok();
}

// TBB implementation: parallel_for(0, Numel(), body) uses the default
// auto_partitioner to recursively bisect the half-open range with dynamic
// work-stealing scheduling. body invokes the same pixel kernel for each index;
// independent iterations and naive-identical computation ensure bit-exact
// results. O(N*H_out*W_out*C).
#ifdef hpcplatform_WITH_TBB
inline Status ResizeTbbImpl(const Tensor& input, const ResizeParams& options,
                            Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  hpcplatform_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
  tbb::parallel_for(int64_t{0}, output->Numel(), [&](int64_t i) {
    ResizeElement(input, image, output_info, options, i, output);
  });
  return Status::Ok();
}
#endif

}  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_RESIZE_RESIZE_CPU_H_
