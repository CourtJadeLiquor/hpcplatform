// CPU TBB level for wavelet: instantiates the shared skeleton with
// WaveletTbbRunner. It uses the same plane driver as naive/OpenMP, delegating
// only row/column loops to oneTBB parallel_for. Accumulation order within each
// line is unchanged, so output is bit-exact.
//
// Granularity: one 1-D analysis/synthesis per task. TBB automatically partitions
// work and balances load through work stealing.

#include "wavelet_cpu.h"

namespace mycv {
namespace ops_internal {

// Same driver as naive, with row/column loops replaced by tbb::parallel_for.
Status WaveletCpuTbb(const OpContext&, const Tensor& input,
                     const Tensor* /*other*/, const void* params,
                     Tensor* output) {
  return WaveletImpl<WaveletTbbRunner>(input, params, output);
}

}  // namespace ops_internal
}  // namespace mycv
