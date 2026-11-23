// Entry point for the wavelet operator: handles registration, parameter
// validation, kernel lookup, and output allocation before dispatching to a
// backend. All backends share the parameter contract in wavelet_internal.h.
//
// Forward flow: validate dtype/rank/layout and even H/W -> ParseImage ->
// FindKernel -> allocate four H/2 x W/2 tensors for LL/LH/HL/HH -> invoke the
// backend with the image as input and LH/HL/HH passed through extra_out.
// Inverse flow: additionally validate matching shape/dtype/backend/device across
// all four bands -> allocate the H x W output -> invoke the backend with LL as
// input and the other three bands passed through extra_in.
// This file only orchestrates; it performs no element-wise computation.

#include "hpcplatform/ops/wavelet.h"

#include <utility>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "wavelet_internal.h"

namespace hpcplatform {
namespace ops_internal {

// Register implementations compiled into this binary: the CPU naive/OpenMP/TBB
// levels and the optional CUDA backend. The GPU implementation occupies the
// CpuKernelLevel::kNaive slot, while actual dispatch depends only on BackendType;
// levels not compiled are absent at registration time.
void RegisterWaveletKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kWaveletTransform, cpu, CpuKernelLevel::kNaive,
                 &WaveletCpuNaive);
#ifdef hpcplatform_WITH_OPENMP
  RegisterKernel(OpKind::kWaveletTransform, cpu, CpuKernelLevel::kOpenMP,
                 &WaveletCpuOpenMP);
#endif
#ifdef hpcplatform_WITH_TBB
  RegisterKernel(OpKind::kWaveletTransform, cpu, CpuKernelLevel::kTbb,
                 &WaveletCpuTbb);
#endif
#ifdef hpcplatform_WITH_CUDA
  RegisterKernel(OpKind::kWaveletTransform, BackendType::kCUDA,
                 CpuKernelLevel::kNaive, &WaveletCuda);
#endif
  // Do not register an OpenCV version: lookup for that backend returns
  // kNotImplemented rather than silently running a host kernel on device memory.
}

}  // namespace ops_internal

namespace {

// Detail bands are signed and accept only f32/f64; u8 cannot represent negative
// coefficients, so return kUnsupported directly.
Status ValidateBandDType(DataType dtype) {
  if (dtype != DataType::kFloat32 && dtype != DataType::kFloat64) {
    return Status::Unsupported(
        "WaveletTransform supports f32 and f64 only: detail bands carry "
        "signed values and cannot be represented in u8");
  }
  return Status::Ok();
}

// A single-level periodic DWT halves both directions, so H and W must be even;
// otherwise subband dimensions are undefined and kInvalidArgument is returned.
Status ValidateEvenExtent(const ops_internal::ImageInfo& info) {
  if (info.h % 2 != 0 || info.w % 2 != 0) {
    return Status::InvalidArgument(
        "WaveletTransform requires even height and width: the periodic "
        "single-level DWT halves both extents");
  }
  return Status::Ok();
}

}  // namespace

// Forward single-level 2D DWT: decompose an H x W image into four H/2 x W/2
// subbands.
//
// Steps: validate invocation/dtype/even dimensions -> parse layout -> find the
// backend -> allocate four bands -> invoke the backend with the image as input,
// LL as output, and the other three bands passed through extra_out.
// Complexity: O(H*W*taps), where taps is the filter length (Haar: 2, Db4: 4).
Result<WaveletBands> WaveletTransform(const OpContext& context,
                                      const Tensor& input, WaveletKind kind,
                                      ImageLayout layout) {
  hpcplatform_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  hpcplatform_RETURN_IF_ERROR(ValidateBandDType(input.dtype()));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  hpcplatform_RETURN_IF_ERROR(ValidateEvenExtent(image.value()));
  auto kernel = ops_internal::FindKernel(OpKind::kWaveletTransform,
                                         input.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  const ops_internal::ImageInfo& info = image.value();
  const Shape band_shape =
      ops_internal::MakeImageShape(info, info.h / 2, info.w / 2);
  auto ll = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!ll.ok()) return ll.status();
  auto lh = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!lh.ok()) return lh.status();
  auto hl = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!hl.ok()) return hl.status();
  auto hh = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!hh.ok()) return hh.status();

  ops_internal::WaveletParams params;
  params.kind = kind;
  params.layout = layout;
  params.inverse = false;
  params.extra_out[0] = &lh.value();
  params.extra_out[1] = &hl.value();
  params.extra_out[2] = &hh.value();
  hpcplatform_RETURN_IF_ERROR(
      kernel.value()(context, input, nullptr, &params, &ll.value()));

  WaveletBands bands;
  bands.ll = std::move(ll.value());
  bands.lh = std::move(lh.value());
  bands.hl = std::move(hl.value());
  bands.hh = std::move(hh.value());
  return bands;
}

// Inverse single-level 2D DWT: perfectly reconstruct an H x W image from
// LL/LH/HL/HH.
//
// Steps: validate invocation, dtype, and matching shape/dtype/backend/device
// across all four bands -> parse layout from the band shape (inferring image
// dimensions) -> find the backend -> allocate output -> invoke the backend with
// LL as input and the other three bands passed through extra_in.
// Complexity: O(H*W*taps). `kind` must match decomposition; otherwise
// reconstruction is no longer an identity operation.
Result<Tensor> InverseWaveletTransform(const OpContext& context,
                                       const WaveletBands& bands,
                                       WaveletKind kind, ImageLayout layout) {
  hpcplatform_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, bands.ll));
  const Tensor* all[4] = {&bands.ll, &bands.lh, &bands.hl, &bands.hh};
  for (const Tensor* band : all) {
    if (!band->defined()) {
      return Status::InvalidArgument(
          "InverseWaveletTransform requires all four bands");
    }
    if (band->shape() != bands.ll.shape() ||
        band->dtype() != bands.ll.dtype() ||
        band->backend() != bands.ll.backend() ||
        band->device() != bands.ll.device()) {
      return Status::InvalidArgument(
          "WaveletBands must share shape, dtype, backend and device");
    }
  }
  hpcplatform_RETURN_IF_ERROR(ValidateBandDType(bands.ll.dtype()));
  auto image = ops_internal::ParseImage(bands.ll.shape(), layout);
  if (!image.ok()) return image.status();
  const ops_internal::ImageInfo& band_info = image.value();
  auto kernel = ops_internal::FindKernel(OpKind::kWaveletTransform,
                                         bands.ll.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  const Shape image_shape =
      ops_internal::MakeImageShape(band_info, band_info.h * 2, band_info.w * 2);
  auto output =
      ops_internal::AllocateOutput(context, image_shape, bands.ll.dtype());
  if (!output.ok()) return output.status();

  ops_internal::WaveletParams params;
  params.kind = kind;
  params.layout = layout;
  params.inverse = true;
  params.extra_in[0] = &bands.lh;
  params.extra_in[1] = &bands.hl;
  params.extra_in[2] = &bands.hh;
  hpcplatform_RETURN_IF_ERROR(
      kernel.value()(context, bands.ll, nullptr, &params, &output.value()));
  return output;
}

}  // namespace hpcplatform
