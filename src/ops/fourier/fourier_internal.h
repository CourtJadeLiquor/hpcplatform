#ifndef hpcplatform_SRC_OPS_FOURIER_FOURIER_INTERNAL_H_
#define hpcplatform_SRC_OPS_FOURIER_FOURIER_INTERNAL_H_

// Internal contract for the fourier operator directory
// (docs/architecture.md §6.1). This file declares only the entry point and the
// parameter structures and functions shared by backends; it performs no
// computation.
//
// The 1-D DFT is X[k] = sum_n x[n] * exp(-i*2*pi*k*n/N). The 2-D kernel is
// separable into row and column 1-D kernels, so each (n, c) plane is transformed
// by applying 1-D transforms to every row and then every column. The two steps
// are commutative; this implementation always processes rows first.
//
// The fourier.cpp entry point handles validation, dispatch, and output
// allocation. Every backend receives the same parameters:
//   Forward: input = image (real part), other = nullptr, output = Re(X)
//            params->imag = newly allocated Im(X); backends write the
//            imaginary part to this tensor
//   Inverse: input = Re(X), other = Im(X), output = real part of x
//            params->imag = nullptr; backends apply inverse normalization by
//            1/(H*W) uniformly during write-back

#include "framework/image_utils.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        struct FourierParams
        {
            ImageLayout layout;  // Input/output image layout (NHWC or NCHW)
            bool inverse;        // false: forward; true: inverse
            // Forward: imaginary-output slot allocated and owned by the entry point;
            // backends write Im(X) to it.
            // Inverse: unused. The imaginary input is passed through `other`, and
            // normalization is applied once during write-back.
            Tensor* imag;
        };

        // All backend entry points use (context, input, other, params, output).
        // Tensors are allocated by the entry point. Backends only read input/other and
        // only write output/params->imag. See the file header for parameter meanings;
        // each (n, c) plane is transformed independently.

        // CPU reference implementation: processes all rows and columns sequentially,
        // serving as the byte-for-byte baseline for the other levels.
        Status FourierCpuNaive(const OpContext& context,
                               const Tensor& input,
                               const Tensor* other,
                               const void* params,
                               Tensor* output);
#ifdef hpcplatform_WITH_OPENMP
        // CPU OpenMP level: distributes only independent rows/columns across threads;
        // accumulation order is unchanged.
        Status FourierCpuOpenMP(const OpContext& context,
                                const Tensor& input,
                                const Tensor* other,
                                const void* params,
                                Tensor* output);
#endif
#ifdef hpcplatform_WITH_TBB
        // CPU TBB level: same skeleton, with oneTBB parallel_for scheduling rows/columns.
        Status FourierCpuTbb(const OpContext& context,
                             const Tensor& input,
                             const Tensor* other,
                             const void* params,
                             Tensor* output);
#endif
#ifdef hpcplatform_WITH_CUDA
        // CUDA backend: device kernels execute the entire pipeline; the host only
        // generates the twiddle tables and launches kernels.
        Status FourierCuda(const OpContext& context,
                           const Tensor& input,
                           const Tensor* other,
                           const void* params,
                           Tensor* output);
#endif

        // Registers implementations compiled into this binary. OpenCV has no
        // corresponding transform, so lookup returns kNotImplemented rather than
        // silently running host code on device memory.
        void RegisterFourierKernels();

    }  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_FOURIER_FOURIER_INTERNAL_H_
