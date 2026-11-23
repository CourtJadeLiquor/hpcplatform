// CUDA implementation of laplacian_pyramid: device-side composed suboperators
// plus one subtraction kernel.
//
// The per-level pipeline matches the host version (plain text):
//   G_{i+1} = ResizeNearest(GaussianBlur(G_i))   size ceil(H/2) x ceil(W/2)
//   up      = ResizeBilinear(G_{i+1}, H_i, W_i)
//   L_i     = G_i - up                          same size as G_i
//   Top level L_n = G_n
// Gaussian filtering and both resize operations invoke registered CUDA
// suboperators; only the final element-wise subtraction needs this file's own
// kernel (SubtractBandsKernel).
//
// Precision: subtraction itself is an exact floating-point operation, but device
// suboperators may use FMA contraction and differ from the host path in low bits;
// cross-backend comparisons therefore use a 1e-4 relative tolerance.

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

#include "backends/cuda/cuda_device_utils.h"
#include "hpcplatform/ops/gaussian_blur.h"
#include "hpcplatform/ops/resize.h"
#include "laplacian_pyramid_internal.h"

namespace hpcplatform {
namespace ops_internal {
namespace {

// Element-wise subtraction: output[index] = fine[index] - up[index], entirely in
// device memory.
// Thread mapping: kThreadsPerBlock threads per block and BlockCount(count) blocks,
//   using a grid-stride loop to cover all indices when elements exceed one wave.
// Indexing: up and output are newly allocated contiguous tensors and can use the
//   linear index directly; fine may be a strided view (caller input or suboperator
//   output), so StridedElementOffset resolves actual element offsets by dimension.
//   The contiguous output is written linearly.
// Complexity: O(count), with one subtraction per element.
template <typename T>
__global__ void SubtractBandsKernel(DeviceTensor fine, DeviceTensor up,
                                    T* output, int64_t count) {
  const T* fine_data = static_cast<const T*>(fine.data);
  const T* up_data = static_cast<const T*>(up.data);
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    output[index] =
        fine_data[StridedElementOffset(fine, index)] - up_data[index];
  }
}

// Dispatch the f32/f64 SubtractBandsKernel by dtype. BlockCount(Numel) determines
// the block count (256 threads per block, capped at kMaxBlocks), and the
// grid-stride loop handles the remainder. FinishLaunch first checks
// cudaGetLastError for launch failures, then cudaStreamSynchronize waits for
// completion and exposes runtime errors; output is ready when Ok is returned.
// Complexity: O(Numel), performed by SubtractBandsKernel.
Status LaunchSubtract(const Tensor& fine, const Tensor& up, Tensor* output,
                      cudaStream_t stream) {
  const unsigned int blocks = BlockCount(output->Numel());
  switch (output->dtype()) {
    case DataType::kFloat32:
      SubtractBandsKernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
          Describe(fine), Describe(up), static_cast<float*>(output->data()),
          output->Numel());
      break;
    case DataType::kFloat64:
      SubtractBandsKernel<double><<<blocks, kThreadsPerBlock, 0, stream>>>(
          Describe(fine), Describe(up), static_cast<double*>(output->data()),
          output->Numel());
      break;
    default:
      return Status::Unsupported(
          "CUDA LaplacianPyramid supports f32 and f64 only");
  }
  return FinishLaunch(stream, "CUDA LaplacianPyramid subtract launch");
}

}  // namespace

// CUDA stage: the complete composed path for one level. First obtain the stream
// (context.stream or the Context default stream), execute all suboperators in
// order on that stream, then compute L_i with LaunchSubtract.
// Inputs/outputs match the host version: output is L_i and
// params->next_gaussian is G_{i+1}.
// Dimensions: the next level uses (h+1)/2 and (w+1)/2 (ceil half-size), and
// upsampling targets the current h x w.
// Complexity: O(N * kernel_size); results match the host within a 1e-4 relative
// tolerance.
Status LaplacianPyramidCudaStage(const OpContext& context, const Tensor& input,
                                 const Tensor* /*other*/, const void* params,
                                 Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const LaplacianPyramidParams*>(params);
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();

  // The next level uses ceil half-size, giving G_{i+1} a well-defined shape even
  // for odd extents.
  const int64_t next_h = (image.value().h + 1) / 2;
  const int64_t next_w = (image.value().w + 1) / 2;
  auto blurred = GaussianBlur(context, input, options.kernel_size,
                              options.sigma, options.layout);
  if (!blurred.ok()) return blurred.status();
  auto next = Resize(context, blurred.value(), next_h, next_w,
                     InterpolationMode::kNearest, options.layout);
  if (!next.ok()) return next.status();
  auto up = Resize(context, next.value(), image.value().h, image.value().w,
                   InterpolationMode::kBilinear, options.layout);
  if (!up.ok()) return up.status();
  // The suboperator output up already matches current level dimensions, so only
  // element-wise subtraction remains.
  hpcplatform_RETURN_IF_ERROR(LaunchSubtract(input, up.value(), output,
                                      stream_result.value()->handle()));

  *options.next_gaussian = std::move(next.value());
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace hpcplatform
