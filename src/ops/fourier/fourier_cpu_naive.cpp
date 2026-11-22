// CPU naive level for fourier: instantiates the shared skeleton with only
// SequentialRunner and introduces no parallelism. It is the correctness
// baseline for the other levels (docs/architecture.md §7.2).

#include "fourier_cpu.h"

namespace mycv {
namespace ops_internal {

// Sequentially executes the row and column 1-D transforms of every (n, c)
// plane. See fourier_internal.h for the input/output contract. Complexity
// matches the shared skeleton: O(H*W*log) for radix-2, with an
// O(H*W*(H+W)) fallback for non-power-of-two sizes.
Status FourierCpuNaive(const OpContext&, const Tensor& input,
                       const Tensor* other, const void* params,
                       Tensor* output) {
  return FourierImpl<SequentialRunner>(
      input, other, *static_cast<const FourierParams*>(params), output);
}

}  // namespace ops_internal
}  // namespace mycv
