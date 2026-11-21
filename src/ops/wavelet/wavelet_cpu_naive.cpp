// wavelet 的 CPU naive 级别：只用 WaveletSequentialRunner 实例化共享骨架，
// 不引入任何并行。它是其他级别的正确性基线（docs/architecture.md §7.2）。

#include "wavelet_cpu.h"

namespace mycv {
namespace ops_internal {

// 顺序执行每个 (n, c) 平面的行/列滤波器组。正向写 LL/LH/HL/HH，逆向由
// 四带重建图像；复杂度 O(H*W*taps)，taps 为 Haar 2、Db4 4。
Status WaveletCpuNaive(const OpContext&, const Tensor& input,
                       const Tensor* /*other*/, const void* params,
                       Tensor* output) {
  return WaveletImpl<WaveletSequentialRunner>(input, params, output);
}

}  // namespace ops_internal
}  // namespace mycv
