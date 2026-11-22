#include "gaussian_blur_cpu.h"

#include "framework/compiler.h"
#include "framework/dtype_dispatch.h"

// Portable SIMD level (generic SIMD): two separable 1D convolution passes,
// each O(H*W*K), reducing per-pixel cost from O(K^2) to O(2K). Differences from
// naive are implementation details only: typed pointer reads based on the
// actual dtype (no dtype switch inside the tap loop) and MYCV_CPU_IVDEP to tell
// the compiler that iterations are independent and can be auto-vectorized.
// Accumulation order exactly matches naive (i = 0..2r, double accumulation),
// so output is bit-exact with no integer/floating-point tolerance.
// Prerequisite: input must be contiguous; strided views are rejected.
namespace mycv {
namespace ops_internal {

namespace {

// One output pixel of the horizontal pass (template instantiated by dtype):
//   tmp[linear] = sum_i k[i] * in(n, y, clamp(x+i-r), c)
// Decode (n,y,x,c) from the linear index, derive the channel/x/row/batch strides
// from layout, and address pointers directly to avoid ImageByteOffset per tap.
// O(K) per element; replicate boundaries.
template <typename T>
void BlurHorizontalTyped(const T* input, const ImageInfo& info,
                         const double* kernel, int radius, int64_t linear,
                         double* tmp) {
  int64_t n = 0;
  int64_t y = 0;
  int64_t x = 0;
  int64_t c = 0;
  DecodeBlurIndex(info, linear, n, y, x, c);
  const int64_t channels = info.c;
  const int64_t width = info.w;
  const int64_t channel_stride =
      info.layout == ImageLayout::kNHWC ? 1 : info.h * info.w;
  const int64_t x_stride = info.layout == ImageLayout::kNHWC ? channels : 1;
  const int64_t row_stride =
      info.layout == ImageLayout::kNHWC ? info.w * channels : info.w;
  const int64_t batch_stride = info.layout == ImageLayout::kNHWC
                                   ? info.h * row_stride
                                   : channels * info.h * info.w;
  const T* image = input + (n * batch_stride);
  double acc = 0.0;
  MYCV_CPU_IVDEP
  for (int i = 0; i < 2 * radius + 1; ++i) {
    const int64_t xx = ClampIndex(x + i - radius, width);
    acc += kernel[i] *
           static_cast<double>(
               image[y * row_stride + xx * x_stride + c * channel_stride]);
  }
  tmp[linear] = acc;
}

// One output pixel of the vertical pass (template instantiated by dtype):
//   out(linear) = store(sum_i k[i] * tmp(n, clamp(y+i-r), x, c))
// Read the double intermediate buffer through ImageLinearIndex, using the same
// accumulation order as naive; then round/convert to the output dtype and write
// to output. O(K) per pixel.
template <typename T>
void BlurVerticalTyped(const double* tmp, const ImageInfo& info,
                       const double* kernel, int radius, int64_t linear,
                       Tensor* output) {
  int64_t n = 0;
  int64_t y = 0;
  int64_t x = 0;
  int64_t c = 0;
  DecodeBlurIndex(info, linear, n, y, x, c);
  double acc = 0.0;
  MYCV_CPU_IVDEP
  for (int i = 0; i < 2 * radius + 1; ++i) {
    const int64_t yy = ClampIndex(y + i - radius, info.h);
    acc += kernel[i] * tmp[ImageLinearIndex(info, n, yy, x, c)];
  }
  StoreBlurPixel(static_cast<char*>(output->data()) +
                     static_cast<size_t>(linear) * output->ElementSize(),
                 output->dtype(), acc);
}

// Two-pass implementation after type dispatch: parse the image, build the
// kernel, and allocate a double intermediate buffer. Run the horizontal pass
// over all linear indices, then the vertical pass into output. Each pass is
// O(H*W*K), with O(H*W*C) doubles of additional memory. Only contiguous tensors
// are accepted because the typed pointers below address contiguous arrays.
template <typename T>
Status BlurSimdTyped(const Tensor& input, const GaussianBlurParams& options,
                     Tensor* output, T*) {
  // Typed reads treat the tensor as a contiguous array and would read incorrect
  // pixels from a strided view, so reject it directly (v1 contract).
  if (!input.IsContiguous()) {
    return Status::Unsupported(
        "GaussianBlur SIMD level requires contiguous tensors");
  }
  ImageInfo info;
  MYCV_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  const int radius = options.kernel_size / 2;
  std::vector<double> tmp(static_cast<size_t>(output->Numel()));
  const T* input_data = static_cast<const T*>(input.data());
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurHorizontalTyped<T>(input_data, info, kernel.data(), radius, i,
                           tmp.data());
  }
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurVerticalTyped<T>(tmp.data(), info, kernel.data(), radius, i, output);
  }
  return Status::Ok();
}

}  // namespace

// Entry point: dispatch by input dtype to the u8/f32/f64 template
// implementations. All promote pixels to double for accumulation, ensuring
// bit-exact agreement with naive.
Status GaussianBlurCpuSimd(const OpContext&, const Tensor& input, const Tensor*,
                           const void* params, Tensor* output) {
  const auto& options = *static_cast<const GaussianBlurParams*>(params);
  switch (input.dtype()) {
    case DataType::kUInt8:
      return BlurSimdTyped<uint8_t>(input, options, output, nullptr);
    case DataType::kFloat32:
      return BlurSimdTyped<float>(input, options, output, nullptr);
    case DataType::kFloat64:
      return BlurSimdTyped<double>(input, options, output, nullptr);
    default:
      return Status::Unsupported("GaussianBlur supports u8, f32 and f64 only");
  }
}

}  // namespace ops_internal
}  // namespace mycv
