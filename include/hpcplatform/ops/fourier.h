#ifndef hpcplatform_OPS_FOURIER_H_
#define hpcplatform_OPS_FOURIER_H_

#include <memory>
#include <string>

#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"
#include "hpcplatform/ops/operator.h"

namespace hpcplatform
{

    // Two-dimensional discrete Fourier transform, evaluated independently on every
    // (n, c) plane of an image tensor.
    //
    // Forward (no normalization):
    //     X[u, v] = sum over y, x of x[y, x] * exp(-2*pi*i*(u*y/H + v*x/W))
    //
    // Inverse (unit-gain on a round trip):
    //     x[y, x] = (1 / (H*W)) * sum over u, v of X[u, v] * exp(+2*pi*i*(u*y/H +
    //     v*x/W))
    //
    // The spectrum is returned as two real tensors with the same shape, dtype,
    // backend and layout as the input, so the caller decides how to store or
    // display magnitude/phase:
    //     ComplexImage::real  -> Re(X)
    //     ComplexImage::imag  -> Im(X)
    //
    // Only f32/f64 are accepted: a spectrum carries signed values, which u8 cannot
    // represent. Any H/W is valid. Power-of-two lengths use the radix-2
    // Cooley-Tukey algorithm; other lengths fall back to the O(n^2) DFT (see the
    // operator README for the complexity table).
    // Complex spectrum: real and imag hold the real and imaginary parts, matching
    // the input shape, dtype, and layout.
    struct ComplexImage
    {
        Tensor real;
        Tensor imag;
    };

    // Computes the 2D discrete Fourier transform for each (n, c) plane; accepts only
    // f32/f64.
    Result<ComplexImage> FourierTransform(const OpContext& context,
                                          const Tensor& input,
                                          ImageLayout layout = ImageLayout::kNHWC);

    // Reconstructs an image from real and imaginary parts; real and imag must
    // describe the same image.
    //
    // `real` and `imag` must describe the same image (shape, dtype, backend and
    // device). Only the real part is returned: that is all an image-domain round
    // trip needs, and the inverse already scales by 1/(H*W).
    Result<Tensor> InverseFourierTransform(const OpContext& context,
                                           const Tensor& real,
                                           const Tensor& imag,
                                           ImageLayout layout = ImageLayout::kNHWC);

    // A configured forward 2D DFT holding a bound layout. Built by LoadOperators
    // from a config entry; call Run with an input allocated on the bound
    // backend/device. See docs/operator_config.md.
    class FourierTransformOperator : public Operator
    {
    public:
        FourierTransformOperator(std::string name,
                                 std::shared_ptr<Context> context,
                                 CpuKernelLevel cpu_level,
                                 ImageLayout layout);

        const std::string& name() const override;
        OpKind kind() const override;

        // Computes the spectrum of `input`; bit-identical to calling
        // hpcplatform::FourierTransform directly.
        Result<ComplexImage> Run(const Tensor& input) const;

    private:
        const std::string name_;
        const ImageLayout layout_;
    };

}  // namespace hpcplatform

#endif  // hpcplatform_OPS_FOURIER_H_
