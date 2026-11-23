// CPU OpenMP backend adapter for resize: this file only restores and forwards
// parameters. ResizeOpenMPImpl in resize_cpu.h implements the parallel strategy
// (parallel for over flat output indices with default scheduling) and uses the
// same pixel kernel as naive, producing bit-exact results.

#include "resize_cpu.h"

namespace hpcplatform {
namespace ops_internal {
// The kernel table defines the entry signature; OpContext and the second input
// tensor are unused. Complexity is O(N*H_out*W_out*C), shared across OpenMP threads.
Status ResizeCpuOpenMP(const OpContext&, const Tensor& input, const Tensor*,
                       const void* params, Tensor* output) {
  return ResizeOpenMPImpl(input, *static_cast<const ResizeParams*>(params),
                          output);
}
}  // namespace ops_internal
}  // namespace hpcplatform
