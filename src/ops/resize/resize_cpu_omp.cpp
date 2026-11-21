// resize 的 CPU OpenMP 后端适配层：本文件只做参数还原与转交，并行策略
// （parallel for 切分输出扁平下标、默认调度）在 resize_cpu.h 的
// ResizeOpenMPImpl 中实现，且与 naive 使用同一像素核，结果位精确一致。

#include "resize_cpu.h"

namespace mycv {
namespace ops_internal {
// 入口签名由内核表约定；OpContext 与第二个输入张量未使用。
// 复杂度 O(N*H_out*W_out*C)，由 OpenMP 线程分摊。
Status ResizeCpuOpenMP(const OpContext&, const Tensor& input, const Tensor*,
                       const void* params, Tensor* output) {
  return ResizeOpenMPImpl(input, *static_cast<const ResizeParams*>(params), output);
}
}  // namespace ops_internal
}  // namespace mycv
