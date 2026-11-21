// resize 的 CPU TBB 后端适配层：本文件只做参数还原与转交，并行策略
// （parallel_for 递归二分输出下标区间、work stealing 调度）在
// resize_cpu.h 的 ResizeTbbImpl 中实现，与 naive 使用同一像素核，位精确一致。

#include "resize_cpu.h"

namespace mycv {
namespace ops_internal {
// 入口签名由内核表约定；OpContext 与第二个输入张量未使用。
// 复杂度 O(N*H_out*W_out*C)，由 TBB 任务并行分摊。
Status ResizeCpuTbb(const OpContext&, const Tensor& input, const Tensor*,
                    const void* params, Tensor* output) {
  return ResizeTbbImpl(input, *static_cast<const ResizeParams*>(params),
                       output);
}
}  // namespace ops_internal
}  // namespace mycv
