#include "gaussian_blur_cpu.h"

// OpenMP backend: two separable 1D Gaussian convolution passes (horizontal,
// then vertical), reducing per-pixel cost from O(K^2) to O(2K). This file only
// connects the entry point to BlurOpenMPImpl in the shared skeleton; see that
// function's comments for the parallel strategy.
namespace hpcplatform
{
    namespace ops_internal
    {

        // Entry point: validate/parse, then execute both blur passes. Parallelize over
        // flat output pixel indices [0, Numel), with one parallel for per pass. Each
        // pixel still accumulates in double using the same tap order, so results are
        // bit-exact with naive and require no tolerance.
        Status GaussianBlurCpuOpenMP(const OpContext&,
                                     const Tensor& input,
                                     const Tensor*,
                                     const void* params,
                                     Tensor* output)
        {
            return BlurOpenMPImpl(input, *static_cast<const GaussianBlurParams*>(params), output);
        }

    }  // namespace ops_internal
}  // namespace hpcplatform
