#ifndef MYCV_SRC_OPS_CPU_RESIZE_H_
#define MYCV_SRC_OPS_CPU_RESIZE_H_

#include <algorithm>
#include <cmath>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "core/value_cast.h"
#include "ops/image_utils.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

inline void ResizeElement(const Tensor& input, const ImageInfo& image,
                          const ImageInfo& output_info,
                          const ResizeParams& options, int64_t output_linear,
                          Tensor* output) {
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

  const long double scale_y = static_cast<long double>(image.h) / options.height;
  const long double scale_x = static_cast<long double>(image.w) / options.width;
  int64_t y0 = 0;
  int64_t x0 = 0;
  int64_t y1 = 0;
  int64_t x1 = 0;
  long double wy = 0;
  long double wx = 0;
  if (options.interpolation == InterpolationMode::kNearest) {
    y0 = std::min(image.h - 1, static_cast<int64_t>(y * scale_y));
    x0 = std::min(image.w - 1, static_cast<int64_t>(x * scale_x));
  } else {
    const long double source_y = (y + 0.5L) * scale_y - 0.5L;
    const long double source_x = (x + 0.5L) * scale_x - 0.5L;
    const long double clamped_y = std::max(0.0L, source_y);
    const long double clamped_x = std::max(0.0L, source_x);
    y0 = static_cast<int64_t>(std::floor(clamped_y));
    x0 = static_cast<int64_t>(std::floor(clamped_x));
    y1 = std::min(image.h - 1, y0 + 1);
    x1 = std::min(image.w - 1, x0 + 1);
    wy = clamped_y - y0;
    wx = clamped_x - x0;
  }

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
    const long double top = top_left + (top_right - top_left) * wx;
    const long double bottom = bottom_left + (bottom_right - bottom_left) * wx;
    value = top + (bottom - top) * wy;
  }
  if (input.dtype() == DataType::kUInt8) value = std::round(value);
  char* destination = static_cast<char*>(output->data());
  detail::StoreFromLongDouble(
      destination + static_cast<size_t>(output_linear) * output->ElementSize(),
      output->dtype(), value);
}

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

inline Status ResizeNaiveImpl(const Tensor& input, const ResizeParams& options,
                              Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  MYCV_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
  for (int64_t i = 0; i < output->Numel(); ++i) {
    ResizeElement(input, image, output_info, options, i, output);
  }
  return Status::Ok();
}

inline Status ResizeOpenMPImpl(const Tensor& input, const ResizeParams& options,
                               Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  MYCV_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
#pragma omp parallel for
  for (int64_t i = 0; i < output->Numel(); ++i) {
    ResizeElement(input, image, output_info, options, i, output);
  }
  return Status::Ok();
}

#ifdef MYCV_WITH_TBB
inline Status ResizeTbbImpl(const Tensor& input, const ResizeParams& options,
                            Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  MYCV_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
  tbb::parallel_for(int64_t{0}, output->Numel(), [&](int64_t i) {
    ResizeElement(input, image, output_info, options, i, output);
  });
  return Status::Ok();
}
#endif

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_CPU_RESIZE_H_
