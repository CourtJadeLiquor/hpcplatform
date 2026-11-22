#ifndef MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_CPU_H_
#define MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_CPU_H_

// Shared CPU skeleton for gaussian_blur (two separable blur passes).
//
// Each pass is O(H*W*K). The horizontal pass writes results to the double
// intermediate buffer tmp; the vertical pass reads tmp and stores according to
// the output dtype. Horizontal:
//   tmp[y,x] = sum_k in[y, x-r+k] * w[k]
// Vertical:
//   out[y,x] = sum_k tmp[y-r+k, x] * w[k]
// Out-of-range indices replicate edge pixels. All CPU levels accumulate in
// double with identical tap order (kernel indices 0 -> size-1, i.e. offsets
// -radius .. +radius), so their results are bit-exact;
// docs/architecture.md §7.2.

#include <cmath>
#include <cstdint>
#include <vector>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "framework/image_utils.h"
#include "gaussian_blur_internal.h"

namespace mycv {
namespace ops_internal {

// Store one pixel according to the output dtype: convert the double result to
// u8/f32/f64. u8 uses half-away-from-zero rounding (std::round). Since all
// kernel coefficients are positive and normalized, the accumulated sum is
// naturally within [0,255] and needs no extra saturation. Entry-point validation
// rejects invalid dtypes, which are ignored here.
inline void StoreBlurPixel(char* destination, DataType dtype, double value) {
  switch (dtype) {
    case DataType::kUInt8:
      *reinterpret_cast<uint8_t*>(destination) =
          static_cast<uint8_t>(std::round(value));
      break;
    case DataType::kFloat32:
      *reinterpret_cast<float*>(destination) = static_cast<float>(value);
      break;
    case DataType::kFloat64:
      *reinterpret_cast<double*>(destination) = value;
      break;
    default:
      break;
  }
}

// Load one pixel according to dtype and promote it to double; u8 spans 0..255,
// while f32/f64 are converted directly. Entry-point validation rejects invalid
// dtypes, for which this returns 0.
inline double LoadBlurPixel(const char* source, DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return *reinterpret_cast<const uint8_t*>(source);
    case DataType::kFloat32:
      return *reinterpret_cast<const float*>(source);
    case DataType::kFloat64:
      return *reinterpret_cast<const double*>(source);
    default:
      return 0.0;
  }
}

// Decode a linear pixel index to (n, y, x, c) using the same rules as resize:
// c varies fastest in NHWC; x varies fastest in CHW (channels are outermost).
inline void DecodeBlurIndex(const ImageInfo& info, int64_t linear, int64_t& n,
                            int64_t& y, int64_t& x, int64_t& c) {
  int64_t remaining = linear;
  if (info.layout == ImageLayout::kNHWC) {
    c = remaining % info.c;
    remaining /= info.c;
    x = remaining % info.w;
    remaining /= info.w;
    y = remaining % info.h;
    n = remaining / info.h;
  } else {
    x = remaining % info.w;
    remaining /= info.w;
    y = remaining % info.h;
    remaining /= info.h;
    c = remaining % info.c;
    n = remaining / info.c;
  }
}

// Replicate boundary: clamp negative indices to 0 and indices >= limit to limit-1.
inline int64_t ClampIndex(int64_t value, int64_t limit) {
  return value < 0 ? 0 : (value >= limit ? limit - 1 : value);
}

// One output element of the horizontal pass:
//   tmp[linear] = sum_i k[i] * in(n, y, clamp(x+i-r), c)
// Inputs are the source tensor, image info, kernel, and radius; output goes to
// the same index in tmp. Accumulate in double for i = 0..2r in order. O(K) per
// element and O(H*W*K) per pass. Clamp boundaries with replicate.
inline void BlurHorizontalElement(const Tensor& input, const ImageInfo& info,
                                  const double* kernel, int radius,
                                  int64_t linear, double* tmp) {
  int64_t n = 0;
  int64_t y = 0;
  int64_t x = 0;
  int64_t c = 0;
  DecodeBlurIndex(info, linear, n, y, x, c);
  const char* base = static_cast<const char*>(input.data());
  double acc = 0.0;
  for (int i = 0; i < 2 * radius + 1; ++i) {
    const int64_t xx = ClampIndex(x + i - radius, info.w);
    acc += kernel[i] *
           LoadBlurPixel(base + ImageByteOffset(input, info, n, y, xx, c),
                         input.dtype());
  }
  tmp[linear] = acc;
}

// One output element of the vertical pass:
//   out(linear) = store(sum_i k[i] * tmp(n, clamp(y+i-r), x, c))
// Read the horizontal pass's double buffer, accumulate in increasing i order,
// and store according to the output dtype. O(K) per element and O(H*W*K) per
// pass, with replicate boundaries along y as well.
inline void BlurVerticalElement(const double* tmp, const ImageInfo& info,
                                const double* kernel, int radius,
                                int64_t linear, Tensor* output) {
  int64_t n = 0;
  int64_t y = 0;
  int64_t x = 0;
  int64_t c = 0;
  DecodeBlurIndex(info, linear, n, y, x, c);
  double acc = 0.0;
  for (int i = 0; i < 2 * radius + 1; ++i) {
    const int64_t yy = ClampIndex(y + i - radius, info.h);
    acc += kernel[i] * tmp[ImageLinearIndex(info, n, yy, x, c)];
  }
  StoreBlurPixel(static_cast<char*>(output->data()) +
                     static_cast<size_t>(linear) * output->ElementSize(),
                 output->dtype(), acc);
}

// Validate and parse the input shape according to options.layout, filling ImageInfo (n/h/w/c/layout).
inline Status PrepareBlur(const Tensor& input,
                          const GaussianBlurParams& options, ImageInfo* info) {
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  *info = parsed.value();
  return Status::Ok();
}

// Naive reference implementation: runs both passes sequentially and is the
// bit-exact reference for all other levels. Steps: parse image -> build kernel
// -> run the horizontal pass for every linear index (writing a double temporary
// buffer) -> run the vertical pass into output. Each pass is O(H*W*K), with
// O(H*W*C) doubles of additional memory.
inline Status BlurNaiveImpl(const Tensor& input,
                            const GaussianBlurParams& options, Tensor* output) {
  ImageInfo info;
  MYCV_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  const int radius = options.kernel_size / 2;
  std::vector<double> tmp(static_cast<size_t>(output->Numel()));
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurHorizontalElement(input, info, kernel.data(), radius, i, tmp.data());
  }
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurVerticalElement(tmp.data(), info, kernel.data(), radius, i, output);
  }
  return Status::Ok();
}

#ifdef MYCV_WITH_OPENMP
// OpenMP level: parallelize over flat pixel indices [0, Numel), with one
// `#pragma omp parallel for` per pass. No schedule is specified, so the runtime
// default applies (typically static). Each thread writes only its own tmp/output
// elements, with no critical sections. An implicit barrier between passes makes
// all horizontal results visible to the vertical pass. The per-pixel tap order
// is unchanged, ensuring bit-exact agreement with naive.
inline Status BlurOpenMPImpl(const Tensor& input,
                             const GaussianBlurParams& options,
                             Tensor* output) {
  ImageInfo info;
  MYCV_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  const int radius = options.kernel_size / 2;
  std::vector<double> tmp(static_cast<size_t>(output->Numel()));
#pragma omp parallel for
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurHorizontalElement(input, info, kernel.data(), radius, i, tmp.data());
  }
#pragma omp parallel for
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurVerticalElement(tmp.data(), info, kernel.data(), radius, i, output);
  }
  return Status::Ok();
}
#endif

#ifdef MYCV_WITH_TBB
// TBB level: partition [0, Numel) with tbb::parallel_for (default
// auto_partitioner; runtime-determined granularity), once per pass. The first
// call completes before the second returns control, so the vertical pass reads
// complete horizontal results. Each index is handled by exactly one thread and
// tap order is unchanged, ensuring bit-exact agreement with naive.
inline Status BlurTbbImpl(const Tensor& input,
                          const GaussianBlurParams& options, Tensor* output) {
  ImageInfo info;
  MYCV_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  const int radius = options.kernel_size / 2;
  std::vector<double> tmp(static_cast<size_t>(output->Numel()));
  tbb::parallel_for(int64_t{0}, output->Numel(), [&](int64_t i) {
    BlurHorizontalElement(input, info, kernel.data(), radius, i, tmp.data());
  });
  tbb::parallel_for(int64_t{0}, output->Numel(), [&](int64_t i) {
    BlurVerticalElement(tmp.data(), info, kernel.data(), radius, i, output);
  });
  return Status::Ok();
}
#endif

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_CPU_H_
