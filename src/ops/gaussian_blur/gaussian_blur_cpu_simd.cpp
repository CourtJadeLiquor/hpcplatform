#include "gaussian_blur_cpu.h"

#include "framework/compiler.h"
#include "framework/dtype_dispatch.h"

// 可移植 SIMD 层（generic SIMD）：可分离两趟一维卷积，
// 每趟 O(H*W*K)，把每像素 O(K^2) 降到 O(2K)。
// 与 naive 的差别只在实现细节：按真实 dtype 做模板化指针读取
// （tap 循环内不再切换 dtype），并用 MYCV_CPU_IVDEP 提示编译器
// 各次迭代之间没有依赖、可自动向量化。
// 累加顺序与 naive 完全相同（i = 0..2r，double 累加），
// 因此输出与 naive 位精确一致，没有整数/浮点容差。
// 前提：输入必须连续（strided view 会被拒绝）。
namespace mycv {
namespace ops_internal {

namespace {

// 水平 pass 的单个输出像素（模板按 dtype 实例化）：
//   tmp[linear] = sum_i k[i] * in(n, y, clamp(x+i-r), c)
// 先由线性下标解码 (n,y,x,c)，再按 layout 推出 channel/x/row/batch
// 四个步长，指针直接寻址，避免每 tap 走 ImageByteOffset。
// 单元素复杂度 O(K)；边界 replicate。
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
  const int64_t x_stride =
      info.layout == ImageLayout::kNHWC ? channels : 1;
  const int64_t row_stride =
      info.layout == ImageLayout::kNHWC ? info.w * channels : info.w;
  const int64_t batch_stride =
      info.layout == ImageLayout::kNHWC ? info.h * row_stride : channels * info.h * info.w;
  const T* image = input + (n * batch_stride);
  double acc = 0.0;
  MYCV_CPU_IVDEP
  for (int i = 0; i < 2 * radius + 1; ++i) {
    const int64_t xx = ClampIndex(x + i - radius, width);
    acc += kernel[i] * static_cast<double>(
                            image[y * row_stride + xx * x_stride + c * channel_stride]);
  }
  tmp[linear] = acc;
}

// 垂直 pass 的单个输出像素（模板按 dtype 实例化）：
//   out(linear) = store(sum_i k[i] * tmp(n, clamp(y+i-r), x, c))
// 从 double 中间缓冲按 ImageLinearIndex 取数，累加顺序与 naive
// 一致；最后按输出 dtype 舍入/转换后写入 output。O(K) / 像素。
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

// 类型分派后的两趟实现：解析图像、构造核、分配 double 中间缓冲，
// 先对全部线性下标做水平 pass，再做垂直 pass 写入 output。
// 每 pass O(H*W*K)，额外内存 O(H*W*C) 个 double。
// 只接受连续张量，因为下面的 typed 指针按连续数组寻址。
template <typename T>
Status BlurSimdTyped(const Tensor& input, const GaussianBlurParams& options,
                     Tensor* output, T*) {
  // typed 读取把张量当作连续数组，strided view 会读错像素，直接拒绝
  // （v1 约定）。
  if (!input.IsContiguous()) {
    return Status::Unsupported("GaussianBlur SIMD level requires contiguous tensors");
  }
  ImageInfo info;
  MYCV_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  const int radius = options.kernel_size / 2;
  std::vector<double> tmp(static_cast<size_t>(output->Numel()));
  const T* input_data = static_cast<const T*>(input.data());
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurHorizontalTyped<T>(input_data, info, kernel.data(), radius, i, tmp.data());
  }
  for (int64_t i = 0; i < output->Numel(); ++i) {
    BlurVerticalTyped<T>(tmp.data(), info, kernel.data(), radius, i, output);
  }
  return Status::Ok();
}

}  // namespace

// 入口：按输入 dtype 分派到 u8/f32/f64 的模板实现。
// 三者内部都把像素提升为 double 累加，故与 naive 位精确一致。
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
