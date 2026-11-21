#ifndef MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_CPU_H_
#define MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_CPU_H_

// gaussian_blur 的 CPU 共享骨架（可分离两趟模糊）。
//
// 两趟各 O(H*W*K)：水平 pass 把结果写进 double 中间缓冲 tmp，
// 垂直 pass 读 tmp 并按输出 dtype 存储；水平方向
//   tmp[y,x] = sum_k in[y, x-r+k] * w[k]
// 垂直方向
//   out[y,x] = sum_k tmp[y-r+k, x] * w[k]
// 越界下标按 replicate 复制边缘像素。
// 所有 CPU 层都用 double 累加且 tap 顺序一致（核下标 0 -> size-1，
// 即偏移 -radius .. +radius），因此各层结果位精确一致；
// docs/architecture.md §7.2。

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

// 按输出 dtype 写回一个像素：把 double 结果转成 u8/f32/f64。
// u8 采用 half away from zero 舍入（std::round）；由于核系数全为正且
// 已归一化，累加和天然落在 [0,255]，无需额外饱和处理。
// 非法 dtype 会被入口校验拦下，这里直接忽略。
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

// 按 dtype 读取一个像素并提升为 double；u8 值域 0..255，f32/f64 直接扩展。
// 非法 dtype 会被入口校验拦下，这里返回 0。
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

// 把一维线性像素下标解码为 (n, y, x, c)，解码规则与 resize 相同：
// NHWC 下 c 变化最快，CHW 下 x 变化最快（channel 为外层）。
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

// replicate 边界：负下标夹到 0，>= limit 夹到 limit-1，其余不变。
inline int64_t ClampIndex(int64_t value, int64_t limit) {
  return value < 0 ? 0 : (value >= limit ? limit - 1 : value);
}

// 水平 pass 的单个输出元素：
//   tmp[linear] = sum_i k[i] * in(n, y, clamp(x+i-r), c)
// 输入为原始张量、图像信息、核与半径，输出写入 tmp 的同一下标。
// 按 i 从 0 到 2r 顺序累加（double），复杂度 O(K)；整个 pass 为
// O(H*W*K)。边界用 replicate 夹取。
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

// 垂直 pass 的单个输出元素：
//   out(linear) = store(sum_i k[i] * tmp(n, clamp(y+i-r), x, c))
// 从水平 pass 的 double 缓冲读取，按 i 递增顺序累加后按输出 dtype
// 存储；单元素 O(K)，整个 pass O(H*W*K)，y 方向边界同样 replicate。
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

// 校验并按 options.layout 解析输入形状，填充 ImageInfo（n/h/w/c/layout）。
inline Status PrepareBlur(const Tensor& input,
                          const GaussianBlurParams& options, ImageInfo* info) {
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  *info = parsed.value();
  return Status::Ok();
}

// naive 参考实现：顺序执行两趟，是所有其他层的位精确基准。
// 步骤：解析图像 -> 构造核 -> 为每个线性下标做水平 pass（写 double
// 临时缓冲）-> 再做一遍垂直 pass 写入 output。
// 每 pass O(H*W*K)，额外内存 O(H*W*C) 个 double。
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
// OpenMP 层：把扁平化的像素下标 [0, Numel) 作为并行粒度，
// 两趟各开一个 `#pragma omp parallel for`（未指定 schedule，采用运行时
// 默认调度，通常为 static）。每个线程只写自己的 tmp/output 元素，
// 无临界区；垂直 pass 需要水平结果全部可见，故两趟之间隐含屏障。
// 单像素内部的 tap 顺序未变，因此与 naive 位精确一致。
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
// TBB 层：用 tbb::parallel_for 在区间 [0, Numel) 上做划分（默认
// auto_partitioner，粒度由运行时决定），两趟各调用一次；第二次调用返回
// 时第一次已全部完成，保证垂直 pass 读到完整的水平结果。
// 每个下标被唯一线程处理，tap 顺序不变，故与 naive 位精确一致。
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
