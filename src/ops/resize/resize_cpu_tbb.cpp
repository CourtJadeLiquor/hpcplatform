// CPU TBB backend adapter for resize: this file only restores and forwards
// parameters. ResizeTbbImpl in resize_cpu.h implements the parallel strategy
// (parallel_for recursively bisects output index ranges with work-stealing
// scheduling) and uses the same pixel kernel as naive for bit-exact results.

#include "resize_cpu.h"

namespace hpcplatform {
namespace ops_internal {
// The kernel table defines the entry signature; OpContext and the second input
// tensor are unused. Complexity is O(N*H_out*W_out*C), shared across TBB tasks.
Status ResizeCpuTbb(const OpContext&, const Tensor& input, const Tensor*,
                    const void* params, Tensor* output) {
  return ResizeTbbImpl(input, *static_cast<const ResizeParams*>(params),
                       output);
}
}  // namespace ops_internal
}  // namespace hpcplatform
