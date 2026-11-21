#ifndef MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_INTERNAL_H_
#define MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_INTERNAL_H_

// gaussian_blur 算子目录的内部契约（docs/architecture.md §6.1/§7.2）：
// 参数结构体、各后端内核入口以及注册钩子。均不对外暴露。
//
// 算法为可分离高斯卷积。一维核按
//   G(x) = exp(-x^2 / (2*sigma^2)) / (sqrt(2*pi)*sigma)
// 采样后归一化（保证增益为 1），二维模糊拆成两趟一维卷积，把每个
// 输出像素的 O(K^2) 次乘加降为 O(2K) 次：
//   水平 pass：tmp[y,x] = sum_k in[y, x-r+k] * w[k]
//   垂直 pass：out[y,x] = sum_k tmp[y-r+k, x] * w[k]
// 其中 r = kernel_size / 2。越界下标按 replicate 复制边缘像素处理。
// 每 pass 复杂度 O(H*W*K)，K = kernel_size。
//
// CPU 各实现都以 double 累加、tap 顺序相同（核下标 0 -> size-1，
// 即偏移 -radius .. +radius），因此彼此位精确一致。

#include <cmath>
#include <vector>

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// 算子参数：kernel_size 必须为 >= 3 的奇数（半径 r = kernel_size/2），
// sigma 为有限正数，layout 指定输入/输出图像的维度排布。
struct GaussianBlurParams {
  int kernel_size;
  double sigma;
  ImageLayout layout;
};

// 构造归一化的一维高斯核（长度 kernel_size，复杂度 O(K)）。
// 输入 kernel_size 与 sigma；对 i = 0..K-1 取 d = i - radius，先算
// G(d) 再累加求和，最后每个系数除以该和使总和为 1。
// 归一化求和的累加顺序与 tap 顺序一致，因此各后端构造出的系数
// 逐字节相同；越界像素由调用方按 replicate 处理，核本身不感知边界。
inline std::vector<double> BuildGaussianKernel(int kernel_size, double sigma) {
  const int radius = kernel_size / 2;
  std::vector<double> kernel(static_cast<size_t>(kernel_size));
  double sum = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    const double d = i - radius;
    kernel[static_cast<size_t>(i)] = std::exp(-(d * d) / (2.0 * sigma * sigma));
    sum += kernel[static_cast<size_t>(i)];
  }
  for (double& value : kernel) value /= sum;
  return kernel;
}

// CPU 各层内核入口（实现分散在对应的 _cpu_*.cpp 中）。
// 所有 CPU 层均以 double 累加、tap 顺序一致，必须彼此位精确一致；
// 每层签名统一为 (context, input, other, params, output)。
Status GaussianBlurCpuNaive(const OpContext& context, const Tensor& input,
                            const Tensor* other, const void* params,
                            Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status GaussianBlurCpuOpenMP(const OpContext&, const Tensor&, const Tensor*,
                             const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status GaussianBlurCpuTbb(const OpContext&, const Tensor&, const Tensor*,
                          const void*, Tensor*);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
Status GaussianBlurCpuSimd(const OpContext&, const Tensor&, const Tensor*,
                           const void*, Tensor*);
#endif
#ifdef MYCV_ARCH_AVX2
Status GaussianBlurCpuAvx2(const OpContext&, const Tensor&, const Tensor*,
                           const void*, Tensor*);
#endif
#ifdef MYCV_WITH_CUDA
Status GaussianBlurCuda(const OpContext&, const Tensor&, const Tensor*,
                        const void*, Tensor*);
#endif
#ifdef MYCV_WITH_OPENCV
Status GaussianBlurOpenCv(const OpContext&, const Tensor&, const Tensor*,
                          const void*, Tensor*);
#endif

// 注册编译进本二进制的全部实现：所有已启用的 CPU 层
// （naive/OpenMP/TBB/SIMD/AVX2），以及已启用的可选后端
// CUDA 与 OpenCV（docs/architecture.md §6.3）。
void RegisterGaussianBlurKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_INTERNAL_H_
