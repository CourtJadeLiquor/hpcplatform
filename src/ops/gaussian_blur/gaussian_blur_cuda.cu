#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "backends/cuda/cuda_device_utils.h"
#include "gaussian_blur_internal.h"
#include "mycv/ops/image.h"

// CUDA 后端：可分离高斯模糊。主机侧构造核，设备侧跑两趟一维卷积，
// 把每像素 O(K^2) 次乘加降为 O(2K)：
//
// 流水线：
//  1) GaussianBlurCuda() 解析形状，用 BuildGaussianKernel 在主机生成
//     归一化核，并用 Context::AllocateTensor 分配同形状的 f64 中间
//     张量（即水平 pass 的输出 tmp，位于显存）；
//  2) LaunchBlur() 用 cudaMalloc + cudaMemcpyAsync(H2D，同流) 上传核；
//  3) BlurHorizontalKernel 做水平 pass：
//       tmp[n,y,x,c] = sum_i k[i] * in[n, y, clamp(x+i-r), c]
//     x 方向越界 replicate；
//  4) BlurVerticalKernel 做垂直 pass 并直接按输出 dtype 写回：
//       out[n,y,x,c] = sum_i k[i] * tmp[n, clamp(y+i-r), x, c]
//  5) FinishLaunch 检查启动错误并同步流，之后才 cudaFree 核系数。
//
// 两个内核在同一条 stream 上，天然先后有序；垂直 pass 依赖水平 pass
// 写完整张中间张量，因此两者之间不能插入其他流的操作。
// 线程划分：每 block 256 线程，block 数 = ceil(count/256) 且上限
// 65535；每个线程用 grid-stride 循环（ThreadIndex/ThreadStride）
// 领取线性下标，再用 DecodeImageIndex 映射到 (n,y,x,c)。NHWC 下
// 相邻线程对应相邻 c，CHW 下对应相邻 x，两种布局都是 coalesced。
// 未使用 shared memory / constant memory：系数很小，走 L1/L2 即可。
// 误差来源：设备端默认做 FMA 收缩，舍入与主机逐步 mul+add 不同，
// 故与 CPU 层不位精确，跨后端按 1e-4 相对容差比较
// （docs/qa/testing.md §4.2）。
namespace mycv {
namespace ops_internal {
namespace {

// 设备端按布局计算连续张量的线性下标，C 为通道数：
//   NHWC: ((n*h + y)*w + x)*C + c
//   其他（CHW）: ((n*C + c)*h + y)*w + x
// 垂直 pass 用它索引 f64 中间张量。
__device__ inline int64_t ImageLinearIndexDevice(const DeviceImage& image,
                                                 int64_t n, int64_t y,
                                                 int64_t x, int64_t c) {
  if (image.layout == ImageLayout::kNHWC) {
    return ((n * image.h + y) * image.w + x) * image.c + c;
  }
  return ((n * image.c + c) * image.h + y) * image.w + x;
}

// 按 dtype 从设备指针读取一个元素并提升为 double：u8 取 0..255，
// f32 先扩为 double 再参与累加，f64 直接读取。O(1)。
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

// 水平 pass 内核。输入：源张量 input、图像描述 image、核系数 kernel
// （设备端点）、tap_count 与 radius；输出：f64 中间张量 tmp。
// 每个线程以 grid-stride 循环处理若干线性下标 index < count：
// 1) DecodeImageIndex 把 index 解成 (n,y,x,c)；
// 2) 对 i = 0..tap_count-1 取 xx = x+i-radius 并按 replicate 夹到
//    [0, w-1]，累加 kernel[i] * LoadPixelDouble(in[n,y,xx,c])；
// 3) 写 tmp[index]（tmp 与输入同形状、f64、连续）。
// 单元素 O(K)，整趟 O(H*W*K)。
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

// 垂直 pass 内核。输入：水平 pass 的 f64 中间张量 tmp、图像描述；
// 输出：output_data（按 dtype 解释）。同样 grid-stride 领取线性下标，
// 解码 (n,y,x,c) 后对 i 把 y+i-radius 按 replicate 夹到 [0, h-1]，
// 累加 kernel[i] * tmp[ImageLinearIndexDevice(n,yy,x,c)]。
// 输出是新分配且连续的张量，故线性下标就是元素下标；u8 先 ::round
// 再经 SaturatingCast 饱和到 [0,255]，f32/f64 直接转换。
// 单元素 O(K)，整趟 O(H*W*K)。
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
    // 输出连续，index 即元素下标（u8 按 half away from zero 舍入）。
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

// 启动两趟内核并管理核系数的设备内存。步骤：
// 1) cudaMalloc 分配 tap_count 个 double 存放核系数，再用
//    cudaMemcpyAsync(H2D) 在同一条 stream 上传；后续内核也在此 stream
//    上，因此拷贝先于内核完成，无需额外同步；
// 2) count = output->Numel()，blocks = BlockCount(count)（每 block
//    kThreadsPerBlock = 256 线程，块数上限 65535，剩余元素由
//    grid-stride 循环覆盖）；
// 3) 依次启动水平、垂直内核，中间张量 tmp 由调用方提供；
// 4) FinishLaunch 检查启动错误并同步流，确认两个内核都结束后才
//    cudaFree 核系数，保证不释放仍在使用中的显存；返回其状态。
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

// CUDA 入口：取当前 CUDA 流 -> 按 layout 解析形状 -> 主机侧构造
// 归一化核 -> 用 context.context->AllocateTensor 分配与输入同形状的
// f64 中间张量 -> 交给 LaunchBlur 启动两趟内核。
// 中间张量有 Numel 个 double，生命周期由 context 管理；两趟之间
// 不显式同步，顺序由同一条 stream 保证。
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
}  // namespace mycv
