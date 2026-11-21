#include "gaussian_blur_cpu.h"

// OpenMP 后端：可分离高斯模糊的两趟一维卷积（水平后垂直），
// 把每像素 O(K^2) 降到 O(2K)。本文件只负责把入口接到共享骨架
// BlurOpenMPImpl，并行策略见该函数注释。
namespace mycv {
namespace ops_internal {

// 入口：校验/解析后执行两趟模糊。并行粒度是扁平化的输出像素下标
// [0, Numel)，两趟各一个 parallel for；每像素内部仍按相同 tap 顺序
// 以 double 累加，所以结果与 naive 位精确一致，不引入任何容差。
Status GaussianBlurCpuOpenMP(const OpContext&, const Tensor& input,
                             const Tensor*, const void* params,
                             Tensor* output) {
  return BlurOpenMPImpl(input, *static_cast<const GaussianBlurParams*>(params),
                        output);
}

}  // namespace ops_internal
}  // namespace mycv
