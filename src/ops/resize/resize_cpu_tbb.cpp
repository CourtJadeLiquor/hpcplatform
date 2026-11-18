#include "ops/cpu_resize.h"

namespace mycv {
namespace ops_internal {
Status ResizeCpuTbb(const OpContext&, const Tensor& input, const Tensor*,
                    const void* params, Tensor* output) {
  return ResizeTbbImpl(input, *static_cast<const ResizeParams*>(params), output);
}
}  // namespace ops_internal
}  // namespace mycv
