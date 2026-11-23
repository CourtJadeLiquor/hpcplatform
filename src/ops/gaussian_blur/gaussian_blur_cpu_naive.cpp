#include "gaussian_blur_cpu.h"

namespace hpcplatform {
namespace ops_internal {

// Naive backend (baseline CPU level): a thin wrapper around BlurNaiveImpl in
// the shared skeleton, which sequentially performs horizontal and vertical 1D
// convolutions. Each pass is O(H*W*K), with double intermediates. This portable
// implementation has no parallel/vector dependencies and is the bit-exact
// reference for all other levels (docs/architecture.md §7.2).
Status GaussianBlurCpuNaive(const OpContext&, const Tensor& input,
                            const Tensor*, const void* params, Tensor* output) {
  // params has been validated by the entry point; cast it directly to the parameter struct.
  return BlurNaiveImpl(input, *static_cast<const GaussianBlurParams*>(params),
                       output);
}

}  // namespace ops_internal
}  // namespace hpcplatform
