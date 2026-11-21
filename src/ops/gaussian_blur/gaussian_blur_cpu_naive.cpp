#include "gaussian_blur_cpu.h"

namespace mycv {
namespace ops_internal {

// naive 后端（最基础 CPU 层）：薄封装，真正逻辑在共享骨架的
// BlurNaiveImpl 中完成——顺序执行水平、垂直两趟一维卷积，
// 每趟 O(H*W*K)，中间结果为 double。
// 该实现可移植、不依赖任何并行/向量扩展，是其他所有层的
// 位精确基准（docs/architecture.md §7.2）。
Status GaussianBlurCpuNaive(const OpContext&, const Tensor& input,
                            const Tensor*, const void* params, Tensor* output) {
  // params 由入口校验后传入，这里直接转成参数结构体。
  return BlurNaiveImpl(input, *static_cast<const GaussianBlurParams*>(params),
                       output);
}

}  // namespace ops_internal
}  // namespace mycv
