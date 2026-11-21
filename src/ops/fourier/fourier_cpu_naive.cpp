// fourier 的 CPU naive 级别：只用 SequentialRunner 实例化共享骨架，
// 不引入任何并行。它是其他级别的正确性基线（docs/architecture.md §7.2）。

#include "fourier_cpu.h"

namespace mycv {
namespace ops_internal {

// 顺序执行每个 (n, c) 平面的行/列一维变换。输入/输出契约见
// fourier_internal.h；复杂度同共享骨架：radix-2 为 O(H*W*log)，
// 非 2 的幂回退为 O(H*W*(H+W))。
Status FourierCpuNaive(const OpContext&, const Tensor& input,
                       const Tensor* other, const void* params, Tensor* output) {
  return FourierImpl<SequentialRunner>(
      input, other, *static_cast<const FourierParams*>(params), output);
}

}  // namespace ops_internal
}  // namespace mycv
