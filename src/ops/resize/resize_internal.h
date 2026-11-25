#ifndef hpcplatform_SRC_OPS_RESIZE_RESIZE_INTERNAL_H_
#define hpcplatform_SRC_OPS_RESIZE_RESIZE_INTERNAL_H_

// Internal contract for the resize operator directory (docs/architecture.md §6.1):
// declares the resize parameter struct, backend kernel entry points, and the
// registration hook. Used only by src/ops/resize/ and the kernel registry;
// not exposed through the public API.

#include "framework/image_utils.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        // All parameters for a resize invocation: target height/width, interpolation
        // mode, and input image layout. layout determines how linear indices decode to
        // (n, y, x, c) and determines the output tensor shape.
        struct ResizeParams
        {
            int64_t height;
            int64_t width;
            InterpolationMode interpolation;
            ImageLayout layout;
        };

        // CPU kernel entry points: naive is always compiled; OpenMP/TBB exist only
        // when their features are enabled. Each entry point is defined in its own .cpp
        // and shares the same pixel kernel in resize_cpu.h, so all levels produce
        // bit-exact results for the same input.
        Status ResizeCpuNaive(const OpContext& context,
                              const Tensor& input,
                              const Tensor* other,
                              const void* params,
                              Tensor* output);
#ifdef hpcplatform_WITH_OPENMP
        Status ResizeCpuOpenMP(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_WITH_TBB
        Status ResizeCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
// Optional backend entry points: CUDA and OpenCV use the same signatures as
// the CPU entries and accept only kNaive (see ValidateInvocation's restrictions
// on non-CPU backends).
#ifdef hpcplatform_WITH_CUDA
        Status ResizeCuda(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_WITH_OPENCV
        Status ResizeOpenCv(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

        // Register all resize implementations compiled into this binary in the kernel
        // table (docs/architecture.md §6.3): every enabled CPU level plus enabled
        // optional backends (CUDA/OpenCV).
        void RegisterResizeKernels();

    }  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_RESIZE_RESIZE_INTERNAL_H_
