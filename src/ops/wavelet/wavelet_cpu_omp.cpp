// wavelet 的 CPU OpenMP 级别：用 WaveletOpenMPRunner 实例化共享骨架。
// 并行只作用在互相独立的行（H 次迭代）和列（W/2 次迭代）上；每条线内部
// 仍按 k 升序、j 升序累加，因此与 naive 位精确。
//
// 粒度：一次 parallel for 迭代处理一整条一维分析/合成；单层工作量是
// O(H*W*taps)，行数通常远大于线程数，默认静态划分已足够。

#include "wavelet_cpu.h"

namespace mycv {
namespace ops_internal {

// 与 naive 相同的驱动，只把行/列循环换成 omp parallel for。
Status WaveletCpuOpenMP(const OpContext&, const Tensor& input,
                        const Tensor* /*other*/, const void* params,
                        Tensor* output) {
  return WaveletImpl<WaveletOpenMPRunner>(input, params, output);
}

}  // namespace ops_internal
}  // namespace mycv
