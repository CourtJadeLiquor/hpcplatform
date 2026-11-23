// CUDA backend for resize: one template kernel, dtype dispatch, and a host entry.
//
// Parallel decomposition: the grid uses BlockCount(count) = ceil(count / 256),
// capped at kMaxBlocks = 65535 blocks; each block always has
// kThreadsPerBlock = 256 threads. The kernel uses ThreadIndex()/ThreadStride()
// for a grid-stride loop, so threads process subsequent indices when count
// exceeds one grid's capacity.
//
// Indexing and addressing: the newly allocated output is contiguous, so flat
// indices can be used directly for stores. DecodeImageIndex decodes flat indices
// to (n, y, x, c) according to the output layout. Input may be non-contiguous;
// all reads use ImageElementOffset with the recorded strides.
//
// Algorithm: nearest uses floor(dst * scale), clamped to h-1/w-1. Bilinear uses
// src = (dst + 0.5) * scale - 0.5, clamped to >= 0; out-of-range y1/x1 fall back
// to h-1/w-1. Interpolate along x, then y, using double intermediates. u8 stores
// are rounded and then saturated by SaturatingCast.
//
// Synchronization: FinishLaunch checks cudaGetLastError and then calls
// cudaStreamSynchronize, so results are ready and launch errors are reported
// when the call returns.

#include <cuda_runtime.h>

#include <cstdint>

#include "backends/cuda/cuda_device_utils.h"
#include "hpcplatform/ops/image.h"
#include "resize_internal.h"

namespace hpcplatform {
namespace ops_internal {
namespace {

// Device kernel for resize; template parameter T is u8, float, or double.
// Inputs: input is a tensor descriptor (with strides; possibly non-contiguous),
// input_image/output_image describe image metadata, output is a contiguous
// output buffer, count is the output element count, and interpolation selects
// nearest or bilinear. Each thread claims output indices with grid-stride,
// decodes (n,y,x,c), and computes one pixel per iteration.
// O(N*H_out*W_out*C) overall; O(1) per iteration.
template <typename T>
__global__ void ResizeKernel(DeviceTensor input, DeviceImage input_image,
                             DeviceImage output_image, T* output, int64_t count,
                             InterpolationMode interpolation) {
  const T* input_data = static_cast<const T*>(input.data);
  // Compute both scale factors once outside the loop and reuse them for all indices.
  const double scale_y = static_cast<double>(input_image.h) / output_image.h;
  const double scale_x = static_cast<double>(input_image.w) / output_image.w;
  // Grid-stride: step by the grid's total thread count to cover count beyond one grid's capacity.
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    // The output is contiguous, so flat index is directly the store position.
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(output_image, index, n, y, x, c);
    double value = 0.0;
    if (interpolation == InterpolationMode::kNearest) {
      // Nearest: floor(dst*scale), then clamp to h-1/w-1.
      int64_t source_y = static_cast<int64_t>(y * scale_y);
      int64_t source_x = static_cast<int64_t>(x * scale_x);
      source_y = source_y < input_image.h ? source_y : input_image.h - 1;
      source_x = source_x < input_image.w ? source_x : input_image.w - 1;
      // Strided input addressing supports padded or transposed non-contiguous tensors.
      value = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, source_y, source_x, c)]);
    } else {
      // Bilinear: half-pixel-center mapping; clamp the lower bound to 0, then floor.
      const double source_y = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
      const double source_x = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
      const double clamped_y = source_y > 0.0 ? source_y : 0.0;
      const double clamped_x = source_x > 0.0 ? source_x : 0.0;
      const int64_t y0 = static_cast<int64_t>(::floor(clamped_y));
      const int64_t x0 = static_cast<int64_t>(::floor(clamped_x));
      // Fall back to the last valid row/column for out-of-range bottom/right neighbors.
      const int64_t y1 = y0 + 1 < input_image.h ? y0 + 1 : input_image.h - 1;
      const int64_t x1 = x0 + 1 < input_image.w ? x0 + 1 : input_image.w - 1;
      const double wy = clamped_y - y0;
      const double wx = clamped_x - x0;
      const double top_left = static_cast<double>(
          input_data[ImageElementOffset(input, input_image, n, y0, x0, c)]);
      const double top_right = static_cast<double>(
          input_data[ImageElementOffset(input, input_image, n, y0, x1, c)]);
      const double bottom_left = static_cast<double>(
          input_data[ImageElementOffset(input, input_image, n, y1, x0, c)]);
      const double bottom_right = static_cast<double>(
          input_data[ImageElementOffset(input, input_image, n, y1, x1, c)]);
      // Interpolate the top and bottom rows by wx, then blend them by wy.
      const double top = top_left + (top_right - top_left) * wx;
      const double bottom = bottom_left + (bottom_right - bottom_left) * wx;
      value = top + (bottom - top) * wy;
    }
    // Contiguous output: store by flat index; u8 uses round + SaturatingCast.
    StoreImageValue(output, index, value);
  }
}

// Host launcher: instantiate ResizeKernel for u8/f32/f64 according to dtype,
// with grid/block set to BlockCount(count) and kThreadsPerBlock. Inputs and
// outputs have the same meanings as in the kernel; unsupported dtypes return
// Unsupported. FinishLaunch checks the launch and synchronizes the stream.
// Host-side complexity is O(1).
Status LaunchResize(const DeviceTensor& input, const DeviceImage& input_image,
                    const DeviceImage& output_image, DataType dtype,
                    void* output, int64_t count,
                    InterpolationMode interpolation, cudaStream_t stream) {
  const unsigned int blocks = BlockCount(count);
  switch (dtype) {
    case DataType::kUInt8:
      ResizeKernel<uint8_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<uint8_t*>(output),
          count, interpolation);
      break;
    case DataType::kFloat32:
      ResizeKernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<float*>(output), count,
          interpolation);
      break;
    case DataType::kFloat64:
      ResizeKernel<double><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<double*>(output), count,
          interpolation);
      break;
    default:
      return Status::Unsupported("CUDA Resize supports u8, f32 and f64 only");
  }
  return FinishLaunch(stream, "CUDA Resize kernel launch");
}

}  // namespace

// CUDA backend entry point: obtain CudaStream from context, parse input
// ImageInfo by layout, replace output h/w with the target dimensions from the
// parameters, and pass POD descriptors to LaunchResize. input and output are
// device tensors; params contains target height/width, interpolation, and layout.
// The second input tensor is unused. O(rank) on the host; computation runs on GPU.
Status ResizeCuda(const OpContext& context, const Tensor& input, const Tensor*,
                  const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const ResizeParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  ImageInfo output_info = image_result.value();
  // Output has the same rank/layout as input; only H/W differ.
  output_info.h = options.height;
  output_info.w = options.width;
  return LaunchResize(Describe(input), Describe(image_result.value()),
                      Describe(output_info), input.dtype(), output->data(),
                      output->Numel(), options.interpolation,
                      stream_result.value()->handle());
}

}  // namespace ops_internal
}  // namespace hpcplatform
