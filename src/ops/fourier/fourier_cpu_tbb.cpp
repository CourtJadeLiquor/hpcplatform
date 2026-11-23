// CPU TBB level for fourier: instantiates the shared skeleton with TbbRunner.
// It uses the same plane driver as naive/OpenMP, delegating only row/column
// loops to oneTBB parallel_for. Operation order within each line is unchanged,
// so output is bit-exact.
//
// Granularity: one 1-D transform per task. TBB automatically partitions work
// and balances load through work stealing.

#include "fourier_cpu.h"

namespace hpcplatform {
namespace ops_internal {

// Same driver as naive, with row/column loops replaced by tbb::parallel_for.
Status FourierCpuTbb(const OpContext&, const Tensor& input, const Tensor* other,
                     const void* params, Tensor* output) {
  return FourierImpl<TbbRunner>(
      input, other, *static_cast<const FourierParams*>(params), output);
}

}  // namespace ops_internal
}  // namespace hpcplatform
