// CPU OpenMP level for wavelet: instantiates the shared skeleton with
// WaveletOpenMPRunner. Parallelism applies only to independent rows (H
// iterations) and columns (W/2 iterations); accumulation within each line
// remains in ascending k then j order, so results are bit-exact with naive.
//
// Granularity: each parallel for iteration processes one complete 1-D analysis
// or synthesis. One level costs O(H*W*taps); the row count usually far exceeds
// the thread count, so the default static scheduling is sufficient.

#include "wavelet_cpu.h"

namespace mycv {
namespace ops_internal {

// Same driver as naive, with row/column loops replaced by omp parallel for.
Status WaveletCpuOpenMP(const OpContext&, const Tensor& input,
                        const Tensor* /*other*/, const void* params,
                        Tensor* output) {
  return WaveletImpl<WaveletOpenMPRunner>(input, params, output);
}

}  // namespace ops_internal
}  // namespace mycv
