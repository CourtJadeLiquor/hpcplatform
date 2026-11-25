#ifndef hpcplatform_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_INTERNAL_H_
#define hpcplatform_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_INTERNAL_H_

// Internal contract for the gaussian_blur operator directory
// (docs/architecture.md §6.1/§7.2): parameter struct, backend kernel entry
// points, and registration hook; none are exposed publicly.
//
// The algorithm uses separable Gaussian convolution. Sample and normalize the
// 1D kernel (ensuring unity gain) according to
//   G(x) = exp(-x^2 / (2*sigma^2)) / (sqrt(2*pi)*sigma)
// then split 2D blur into two 1D convolution passes, reducing multiply-adds per
// output pixel from O(K^2) to O(2K):
//   Horizontal pass: tmp[y,x] = sum_k in[y, x-r+k] * w[k]
//   Vertical pass: out[y,x] = sum_k tmp[y-r+k, x] * w[k]
// where r = kernel_size / 2. Out-of-range indices replicate edge pixels.
// Each pass is O(H*W*K), where K = kernel_size.
//
// All CPU implementations accumulate in double with the same tap order (kernel
// indices 0 -> size-1, i.e. offsets -radius .. +radius), so they are bit-exact.

#include <cmath>
#include <vector>

#include "framework/image_utils.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        // Operator parameters: kernel_size must be an odd integer >= 3 (radius
        // r = kernel_size/2), sigma must be finite and positive, and layout specifies
        // the input/output image dimension ordering.
        struct GaussianBlurParams
        {
            int kernel_size;
            double sigma;
            ImageLayout layout;
        };

        // Build a normalized 1D Gaussian kernel of length kernel_size in O(K).
        // Given kernel_size and sigma, set d = i - radius for i = 0..K-1, compute G(d)
        // and accumulate the sum, then divide each coefficient by that sum so they sum
        // to 1. Normalization accumulates in tap order, so all backends construct
        // byte-identical coefficients. Callers replicate out-of-range pixels; the
        // kernel itself is unaware of boundaries.
        inline std::vector<double> BuildGaussianKernel(int kernel_size, double sigma)
        {
            const int radius = kernel_size / 2;
            std::vector<double> kernel(static_cast<size_t>(kernel_size));
            double sum = 0.0;
            for (int i = 0; i < kernel_size; ++i)
            {
                const double d = i - radius;
                kernel[static_cast<size_t>(i)] = std::exp(-(d * d) / (2.0 * sigma * sigma));
                sum += kernel[static_cast<size_t>(i)];
            }
            for (double& value : kernel) value /= sum;
            return kernel;
        }

        // Kernel entry points for all CPU levels (implemented in the corresponding
        // _cpu_*.cpp files). Every CPU level accumulates in double with identical tap
        // order and must be bit-exact with the others. All levels use the signature
        // (context, input, other, params, output).
        Status GaussianBlurCpuNaive(const OpContext& context,
                                    const Tensor& input,
                                    const Tensor* other,
                                    const void* params,
                                    Tensor* output);
#ifdef hpcplatform_WITH_OPENMP
        Status GaussianBlurCpuOpenMP(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_WITH_TBB
        Status GaussianBlurCpuTbb(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_WITH_SIMD_GENERIC
        Status GaussianBlurCpuSimd(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_ARCH_AVX2
        Status GaussianBlurCpuAvx2(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_WITH_CUDA
        Status GaussianBlurCuda(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef hpcplatform_WITH_OPENCV
        Status GaussianBlurOpenCv(
            const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

        // Register every implementation compiled into this binary: all enabled CPU
        // levels (naive/OpenMP/TBB/SIMD/AVX2) and enabled optional CUDA/OpenCV backends
        // (docs/architecture.md §6.3).
        void RegisterGaussianBlurKernels();

    }  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_GAUSSIAN_BLUR_GAUSSIAN_BLUR_INTERNAL_H_
