#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "backends/cuda/cuda_device_utils.h"
#include "gaussian_blur_internal.h"
#include "hpcplatform/ops/image.h"

// CUDA backend: separable Gaussian blur. The host builds the kernel and the
// device runs two 1D convolution passes, reducing per-pixel multiply-adds from
// O(K^2) to O(2K):
//
// Pipeline:
//  1) GaussianBlurCuda() parses the shape, builds a normalized kernel on the
//     host with BuildGaussianKernel, and uses Context::AllocateTensor to
//     allocate a same-shaped f64 intermediate tensor (the horizontal pass's
//     output tmp, in device memory);
//  2) LaunchBlur() uploads the kernel with cudaMalloc + cudaMemcpyAsync
//     (H2D, on the same stream);
//  3) BlurHorizontalKernel performs the horizontal pass:
//       tmp[n,y,x,c] = sum_i k[i] * in[n, y, clamp(x+i-r), c]
//     with replicate for out-of-range x;
//  4) BlurVerticalKernel performs the vertical pass and stores directly in the
//     output dtype:
//       out[n,y,x,c] = sum_i k[i] * tmp[n, clamp(y+i-r), x, c]
//  5) FinishLaunch checks launch errors and synchronizes the stream before
//     cudaFree releases the kernel coefficients.
//
// Both kernels use the same stream and are therefore ordered. The vertical pass
// depends on the horizontal pass completing the entire intermediate tensor, so
// no operations on other streams may be inserted between them.
// Thread decomposition: 256 threads per block and ceil(count/256) blocks,
// capped at 65535. Each thread claims linear indices with a grid-stride loop
// (ThreadIndex/ThreadStride), then DecodeImageIndex maps them to (n,y,x,c).
// Adjacent threads access adjacent c in NHWC and adjacent x in CHW, so both
// layouts are coalesced. Shared/constant memory is unused; the small
// coefficient array can use L1/L2. Error source: device-side FMA contraction by
// default rounds differently from host-side stepwise mul+add, so results are
// not bit-exact with CPU levels. Compare across backends with 1e-4 relative
// tolerance (docs/qa/testing.md §4.2).
namespace hpcplatform {
namespace ops_internal {
namespace {

// Device-side linear index for a contiguous tensor according to layout, where
// C is the channel count:
//   NHWC: ((n*h + y)*w + x)*C + c
//   Other (CHW): ((n*C + c)*h + y)*w + x
// The vertical pass uses this to index the f64 intermediate tensor.
__device__ inline int64_t ImageLinearIndexDevice(const DeviceImage& image,
                                                 int64_t n, int64_t y,
                                                 int64_t x, int64_t c) {
  if (image.layout == ImageLayout::kNHWC) {
    return ((n * image.h + y) * image.w + x) * image.c + c;
  }
  return ((n * image.c + c) * image.h + y) * image.w + x;
}

// Load one element from a device pointer according to dtype and promote it to
// double: u8 spans 0..255, f32 is widened before accumulation, and f64 is read
// directly. O(1).
__device__ inline double LoadPixelDouble(const void* base, int64_t offset,
                                         DataType dtype) {
  if (dtype == DataType::kUInt8) {
    return static_cast<double>(static_cast<const uint8_t*>(base)[offset]);
  }
  if (dtype == DataType::kFloat32) {
    return static_cast<double>(static_cast<const float*>(base)[offset]);
  }
  return static_cast<const double*>(base)[offset];
}

// Horizontal pass kernel. Inputs: source tensor input, image descriptor image,
// kernel coefficients (device pointer), tap_count, and radius; output: f64
// intermediate tensor tmp. Each thread processes multiple linear indices
// index < count with a grid-stride loop:
// 1) DecodeImageIndex decodes index to (n,y,x,c);
// 2) For i = 0..tap_count-1, set xx = x+i-radius, clamp with replicate to
//    [0, w-1], and accumulate kernel[i] * LoadPixelDouble(in[n,y,xx,c]);
// 3) Write tmp[index] (tmp has the input shape, f64 dtype, and is contiguous).
// O(K) per element; O(H*W*K) per pass.
__global__ void BlurHorizontalKernel(DeviceTensor input, DeviceImage image,
                                     double* tmp, int64_t count,
                                     const double* kernel, int tap_count,
                                     int radius, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(image, index, n, y, x, c);
    double acc = 0.0;
    for (int i = 0; i < tap_count; ++i) {
      int64_t xx = x + i - radius;
      xx = xx < 0 ? 0 : (xx >= image.w ? image.w - 1 : xx);
      acc +=
          kernel[i] *
          LoadPixelDouble(input.data,
                          ImageElementOffset(input, image, n, y, xx, c), dtype);
    }
    tmp[index] = acc;
  }
}

// Vertical pass kernel. Inputs: the horizontal pass's f64 intermediate tensor
// tmp and the image descriptor; output: output_data (interpreted by dtype).
// Threads similarly claim linear indices with grid-stride, decode (n,y,x,c),
// clamp y+i-radius with replicate to [0, h-1] for each i, and accumulate
// kernel[i] * tmp[ImageLinearIndexDevice(n,yy,x,c)]. The newly allocated output
// is contiguous, so the linear index is the element index. u8 is rounded with
// ::round and saturated to [0,255] by SaturatingCast; f32/f64 convert directly.
// O(K) per element; O(H*W*K) per pass.
__global__ void BlurVerticalKernel(const double* tmp, DeviceImage image,
                                   void* output_data, int64_t count,
                                   const double* kernel, int tap_count,
                                   int radius, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(image, index, n, y, x, c);
    double acc = 0.0;
    for (int i = 0; i < tap_count; ++i) {
      int64_t yy = y + i - radius;
      yy = yy < 0 ? 0 : (yy >= image.h ? image.h - 1 : yy);
      acc += kernel[i] * tmp[ImageLinearIndexDevice(image, n, yy, x, c)];
    }
    // Contiguous output: index is the element index (u8 rounds half away from zero).
    if (dtype == DataType::kUInt8) {
      static_cast<uint8_t*>(output_data)[index] =
          SaturatingCast<uint8_t>(::round(acc));
    } else if (dtype == DataType::kFloat32) {
      static_cast<float*>(output_data)[index] = static_cast<float>(acc);
    } else {
      static_cast<double*>(output_data)[index] = acc;
    }
  }
}

// Launch both kernels and manage device memory for kernel coefficients. Steps:
// 1) cudaMalloc allocates tap_count doubles for the coefficients, then
//    cudaMemcpyAsync(H2D) uploads them on the same stream as the subsequent
//    kernels. The copy therefore completes before the kernels without extra
//    synchronization;
// 2) count = output->Numel(), blocks = BlockCount(count) (each block has
//    kThreadsPerBlock = 256 threads, at most 65535 blocks; grid-stride loops
//    cover remaining elements);
// 3) Launch horizontal then vertical kernels, using caller-provided tmp;
// 4) FinishLaunch checks launch errors and synchronizes the stream. Only after
//    both kernels finish does cudaFree release the coefficients, avoiding
//    freeing device memory still in use; return its status.
Status LaunchBlur(const Tensor& input, const ImageInfo& info,
                  const GaussianBlurParams& options, const double* kernel_host,
                  double* tmp, Tensor* output, cudaStream_t stream) {
  const int tap_count = options.kernel_size;
  const int radius = tap_count / 2;
  double* kernel_device = nullptr;
  if (cudaMalloc(&kernel_device, sizeof(double) * tap_count) != cudaSuccess) {
    return CudaStatus(cudaGetLastError(), "cudaMalloc for blur kernel");
  }
  cudaMemcpyAsync(kernel_device, kernel_host, sizeof(double) * tap_count,
                  cudaMemcpyHostToDevice, stream);
  const int64_t count = output->Numel();
  const unsigned int blocks = BlockCount(count);
  BlurHorizontalKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      Describe(input), Describe(info), tmp, count, kernel_device, tap_count,
      radius, input.dtype());
  BlurVerticalKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      tmp, Describe(info), output->data(), count, kernel_device, tap_count,
      radius, input.dtype());
  const Status status = FinishLaunch(stream, "GaussianBlur kernel launch");
  cudaFree(kernel_device);
  return status;
}

}  // namespace

// CUDA entry point: get the current CUDA stream -> parse the shape by layout
// -> build a normalized kernel on the host -> allocate a same-shaped f64
// intermediate tensor with context.context->AllocateTensor -> pass it to
// LaunchBlur to launch both kernels. The intermediate tensor contains Numel
// doubles and its lifetime is managed by context. No explicit synchronization
// occurs between passes; the same stream guarantees their order.
Status GaussianBlurCuda(const OpContext& context, const Tensor& input,
                        const Tensor*, const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const auto& options = *static_cast<const GaussianBlurParams*>(params);
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  auto intermediate =
      context.context->AllocateTensor(input.shape(), DataType::kFloat64);
  if (!intermediate.ok()) return intermediate.status();
  return LaunchBlur(input, image.value(), options, kernel.data(),
                    static_cast<double*>(intermediate.value().data()), output,
                    stream);
}

}  // namespace ops_internal
}  // namespace hpcplatform
