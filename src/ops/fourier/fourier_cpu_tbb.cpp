// fourier 的 CPU TBB 级别：用 TbbRunner 实例化共享骨架。
// 与 naive/OpenMP 是同一套平面驱动，仅把行/列循环交给 oneTBB 的
// parallel_for；每条线内部的运算顺序不变，因此输出位精确。
//
// 粒度：一条 1-D 变换就是一个任务，TBB 自动分块并用工作窃取均衡负载。

#include "fourier_cpu.h"

namespace mycv {
namespace ops_internal {

// 与 naive 相同的驱动，只把行/列循环换成 tbb::parallel_for。
Status FourierCpuTbb(const OpContext&, const Tensor& input, const Tensor* other,
                     const void* params, Tensor* output) {
  return FourierImpl<TbbRunner>(
      input, other, *static_cast<const FourierParams*>(params), output);
}

}  // namespace ops_internal
}  // namespace mycv
