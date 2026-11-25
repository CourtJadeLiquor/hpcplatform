#include "gaussian_blur_cpu.h"

// TBB backend: two separable 1D Gaussian convolution passes (horizontal, then
// vertical), reducing per-pixel cost from O(K^2) to O(2K). This file only
// connects the entry point to BlurTbbImpl in the shared skeleton; see that
// function's comments for task partitioning.
namespace hpcplatform
{
    namespace ops_internal
    {

        // Entry point: validate/parse, then partition tasks over [0, Numel) with
        // tbb::parallel_for, once each for the horizontal and vertical passes. Each
        // output index is written by only one task, and double accumulation uses the
        // same tap order as naive, ensuring bit-exact results.
        Status GaussianBlurCpuTbb(const OpContext&,
                                  const Tensor& input,
                                  const Tensor*,
                                  const void* params,
                                  Tensor* output)
        {
            return BlurTbbImpl(input, *static_cast<const GaussianBlurParams*>(params), output);
        }

    }  // namespace ops_internal
}  // namespace hpcplatform
