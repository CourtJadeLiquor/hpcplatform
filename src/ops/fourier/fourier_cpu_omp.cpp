// CPU OpenMP level for fourier: instantiates the shared skeleton with
// OpenMPRunner. Parallelism applies only to independent rows (H iterations)
// and columns (W iterations); bit-reversal and butterflies within each line
// retain their fixed order, so results are bit-exact with naive.
//
// Granularity: each parallel for iteration processes one complete 1-D
// transform. The row/column count usually far exceeds the thread count, so the
// default static scheduling is sufficiently balanced without extra chunking.

#include "fourier_cpu.h"

namespace mycv {
namespace ops_internal {

// Same driver as naive, with row/column loops replaced by omp parallel for.
Status FourierCpuOpenMP(const OpContext&, const Tensor& input,
                        const Tensor* other, const void* params,
                        Tensor* output) {
  return FourierImpl<OpenMPRunner>(
      input, other, *static_cast<const FourierParams*>(params), output);
}

}  // namespace ops_internal
}  // namespace mycv
