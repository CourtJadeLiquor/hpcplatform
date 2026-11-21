#include "gaussian_blur_cpu.h"

// TBB 后端：可分离高斯模糊的两趟一维卷积（水平后垂直），
// 把每像素 O(K^2) 降到 O(2K)。本文件只负责把入口接到共享骨架
// BlurTbbImpl，任务划分方式见该函数注释。
namespace mycv {
namespace ops_internal {

// 入口：校验/解析后用 tbb::parallel_for 在 [0, Numel) 上划分任务，
// 水平与垂直各一次并行调用。每个输出下标只由一个 task 写，且
// tap 顺序与 naive 相同、double 累加，因此位精确一致。
Status GaussianBlurCpuTbb(const OpContext&, const Tensor& input, const Tensor*,
                          const void* params, Tensor* output) {
  return BlurTbbImpl(input, *static_cast<const GaussianBlurParams*>(params),
                     output);
}

}  // namespace ops_internal
}  // namespace mycv
