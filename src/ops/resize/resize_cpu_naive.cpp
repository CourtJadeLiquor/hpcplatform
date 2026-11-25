// CPU naive backend adapter for resize: scales the entire image serially
// without parallelism. ResizeNaiveImpl / ResizeElement in resize_cpu.h perform
// the computation; this file only restores ResizeParams from void* and forwards it.

#include "resize_cpu.h"

namespace hpcplatform
{
    namespace ops_internal
    {
        // The kernel table defines the entry signature: OpContext and the second input
        // tensor are irrelevant to this single-input elementwise operator and remain
        // unnamed. Complexity is O(N*H_out*W_out*C).
        Status ResizeCpuNaive(const OpContext&,
                              const Tensor& input,
                              const Tensor*,
                              const void* params,
                              Tensor* output)
        {
            return ResizeNaiveImpl(input, *static_cast<const ResizeParams*>(params), output);
        }
    }  // namespace ops_internal
}  // namespace hpcplatform
