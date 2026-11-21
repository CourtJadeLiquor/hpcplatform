// fourier 的 CPU OpenMP 级别：用 OpenMPRunner 实例化共享骨架。
// 并行只作用在互相独立的行（H 次迭代）与列（W 次迭代）上；每条线内部
// 仍按固定顺序做 bit-reversal 与蝶形，因此结果与 naive 位精确。
//
// 粒度：一次 parallel for 迭代处理一整条 1-D 变换；行数/列数通常远大于
// 线程数，默认静态划分已足够均衡，不需要额外分块。

#include "fourier_cpu.h"

namespace mycv {
namespace ops_internal {

// 与 naive 相同的驱动，只把行/列循环换成 omp parallel for。
Status FourierCpuOpenMP(const OpContext&, const Tensor& input,
                        const Tensor* other, const void* params,
                        Tensor* output) {
  return FourierImpl<OpenMPRunner>(
      input, other, *static_cast<const FourierParams*>(params), output);
}

}  // namespace ops_internal
}  // namespace mycv
