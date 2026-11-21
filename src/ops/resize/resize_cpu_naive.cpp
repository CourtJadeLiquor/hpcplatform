// resize 的 CPU naive 后端适配层：不加任何并行，串行执行整幅缩放。
// 实际计算在 resize_cpu.h 的 ResizeNaiveImpl / ResizeElement 中完成，
// 本文件只负责把 void* 参数还原为 ResizeParams 并转交。

#include "resize_cpu.h"

namespace mycv {
namespace ops_internal {
// 入口签名由内核表约定：OpContext 与第二个输入张量对缩放无意义（单输入
// 逐元素算子），故匿名忽略。复杂度 O(N*H_out*W_out*C)。
Status ResizeCpuNaive(const OpContext&, const Tensor& input, const Tensor*,
                      const void* params, Tensor* output) {
  return ResizeNaiveImpl(input, *static_cast<const ResizeParams*>(params), output);
}
}  // namespace ops_internal
}  // namespace mycv
