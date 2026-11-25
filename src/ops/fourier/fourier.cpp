// Entry point for the fourier operator: handles registration, parameter
// validation, kernel lookup, and output allocation before dispatching to a
// backend. All backends share the parameter contract in fourier_internal.h.
//
// Forward flow: validate dtype/rank/layout -> ParseImage -> FindKernel ->
// allocate Re/Im -> invoke the backend with the real part as input (the
// imaginary part is written to params.imag).
// Inverse flow: additionally verify matching real/imag
// shape/dtype/backend/device -> allocate output -> pass imag as other;
// backends multiply by 1/(H*W) during write-back.
//
// This file performs no element-wise computation, only orchestration, so every
// backend receives identical parameters.

#include "hpcplatform/ops/fourier.h"

#include <utility>

#include "fourier_internal.h"
#include "framework/image_utils.h"
#include "ops/common/op_common.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        // Registers implementations compiled into this binary: CPU naive/OpenMP/TBB
        // levels and the optional CUDA backend. The GPU implementation is registered
        // in the CpuKernelLevel::kNaive slot; actual dispatch considers only
        // BackendType. Levels not compiled are absent at registration time.
        void RegisterFourierKernels()
        {
            const BackendType cpu = BackendType::kNativeCpu;
            RegisterKernel(OpKind::kFourierTransform, cpu, CpuKernelLevel::kNaive,
                           &FourierCpuNaive);
#ifdef hpcplatform_WITH_OPENMP
            RegisterKernel(OpKind::kFourierTransform, cpu, CpuKernelLevel::kOpenMP,
                           &FourierCpuOpenMP);
#endif
#ifdef hpcplatform_WITH_TBB
            RegisterKernel(OpKind::kFourierTransform, cpu, CpuKernelLevel::kTbb, &FourierCpuTbb);
#endif
#ifdef hpcplatform_WITH_CUDA
            RegisterKernel(OpKind::kFourierTransform, BackendType::kCUDA, CpuKernelLevel::kNaive,
                           &FourierCuda);
#endif
            // No OpenCV version is registered: lookup for that backend returns
            // kNotImplemented rather than silently running a host kernel on device memory.
        }

    }  // namespace ops_internal

    namespace
    {

        // The spectrum must represent signed values, so only f32/f64 are accepted;
        // u8 would lose negative and fractional values. Return kUnsupported here.
        Status ValidateSpectrumDType(DataType dtype)
        {
            if (dtype != DataType::kFloat32 && dtype != DataType::kFloat64)
            {
                return Status::Unsupported(
                    "FourierTransform supports f32 and f64 only: the spectrum carries "
                    "signed values and cannot be represented in u8");
            }
            return Status::Ok();
        }

    }  // namespace

    // Forward 2-D DFT. The input is an H x W real-valued image (possibly a
    // non-contiguous view); outputs are contiguous Re(X) and Im(X) tensors with the
    // same shape/dtype as the input.
    //
    // Steps: validate invocation and dtype -> parse layout -> find backend ->
    // allocate two outputs -> invoke backend (input supplies the real part and
    // params.imag receives the imaginary part) -> assemble ComplexImage.
    // Backend-dependent complexity: O(H*W*log) for radix-2, with an
    // O(H*W*(H+W)) fallback for non-power-of-two sizes.
    Result<ComplexImage> FourierTransform(const OpContext& context,
                                          const Tensor& input,
                                          ImageLayout layout)
    {
        hpcplatform_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
        hpcplatform_RETURN_IF_ERROR(ValidateSpectrumDType(input.dtype()));
        auto image = ops_internal::ParseImage(input.shape(), layout);
        if (!image.ok()) return image.status();
        auto kernel = ops_internal::FindKernel(OpKind::kFourierTransform, input.backend(),
                                               context.cpu_level());
        if (!kernel.ok()) return kernel.status();

        auto real = ops_internal::AllocateOutput(context, input.shape(), input.dtype());
        if (!real.ok()) return real.status();
        auto imag = ops_internal::AllocateOutput(context, input.shape(), input.dtype());
        if (!imag.ok()) return imag.status();
        const ops_internal::FourierParams params{layout, false, &imag.value()};
        hpcplatform_RETURN_IF_ERROR(
            kernel.value()(context, input, nullptr, &params, &real.value()));

        ComplexImage result;
        result.real = std::move(real.value());
        result.imag = std::move(imag.value());
        return result;
    }

    // Inverse 2-D DFT: reconstructs an image from Re(X)/Im(X) and returns the real
    // part of x. Returning the real part is by convention: the input is treated as
    // a real-valued image, and backends write only the real part, multiplying it by
    // 1/(H*W) during write-back so forward -> inverse has unit gain.
    //
    // Steps: validate invocation, dtype, and matching real/imag
    // shape/dtype/backend/device -> parse layout -> find backend -> allocate output
    // -> invoke backend with real as input and imag as other (params.imag is
    // nullptr; no spectral imaginary output is written).
    Result<Tensor> InverseFourierTransform(const OpContext& context,
                                           const Tensor& real,
                                           const Tensor& imag,
                                           ImageLayout layout)
    {
        hpcplatform_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, real));
        if (!imag.defined())
        {
            return Status::InvalidArgument("imag tensor is undefined");
        }
        if (real.shape() != imag.shape() || real.dtype() != imag.dtype() ||
            real.backend() != imag.backend() || real.device() != imag.device())
        {
            return Status::InvalidArgument(
                "real and imag must share shape, dtype, backend and device");
        }
        hpcplatform_RETURN_IF_ERROR(ValidateSpectrumDType(real.dtype()));
        auto image = ops_internal::ParseImage(real.shape(), layout);
        if (!image.ok()) return image.status();
        auto kernel = ops_internal::FindKernel(OpKind::kFourierTransform, real.backend(),
                                               context.cpu_level());
        if (!kernel.ok()) return kernel.status();

        auto output = ops_internal::AllocateOutput(context, real.shape(), real.dtype());
        if (!output.ok()) return output.status();
        const ops_internal::FourierParams params{layout, true, nullptr};
        hpcplatform_RETURN_IF_ERROR(kernel.value()(context, real, &imag, &params, &output.value()));
        return output;
    }

}  // namespace hpcplatform
