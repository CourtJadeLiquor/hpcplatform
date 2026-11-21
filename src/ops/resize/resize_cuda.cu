// resize 的 CUDA 后端：一个模板 kernel + dtype 分派 + 主机侧入口。
//
// 并行划分：grid 用 BlockCount(count) = ceil(count / 256)，并封顶
// kMaxBlocks = 65535 个 block；block 恒为 kThreadsPerBlock = 256 线程。
// kernel 内以 ThreadIndex() / ThreadStride() 做 grid-stride 循环，
// 当 count 大于一次覆盖的容量时，同一线程继续处理后续下标。
//
// 索引与取址：输出是新分配且连续的张量，故扁平下标可直接用于写回；
// DecodeImageIndex 按输出布局把扁平下标解码成 (n, y, x, c)；输入可能
// 非连续，读取一律通过 ImageElementOffset 按记录的 strides 求偏移。
//
// 算法：最近邻取 floor(dst * scale) 并截到 h-1/w-1；双线性用
// src = (dst + 0.5) * scale - 0.5，clamp 到 >= 0，y1/x1 越界回退到
// h-1/w-1，先沿 x 水平插值、再沿 y 垂直插值；中间量用 double。
// u8 写回时先 round 再经 SaturatingCast 做饱和。
//
// 同步：FinishLaunch 先检查 cudaGetLastError，再 cudaStreamSynchronize，
// 因此调用返回时结果已就绪、启动错误也已上抛。

#include <cuda_runtime.h>

#include <cstdint>

#include "backends/cuda/cuda_device_utils.h"
#include "mycv/ops/image.h"
#include "resize_internal.h"

namespace mycv {
namespace ops_internal {
namespace {

// resize 的 device kernel，模板参数 T 为 u8/float/double 中的一种。
// 输入：input 是输入张量描述符（含 strides，可非连续），input_image /
// output_image 描述输入输出图像元信息，output 是连续输出缓冲区，
// count 为输出元素总数，interpolation 选择最近邻或双线性。
// 每个线程按 grid-stride 领走输出下标，解码 (n,y,x,c) 后算一个像素；
// 复杂度 O(N*H_out*W_out*C)，单次迭代 O(1)。
template <typename T>
__global__ void ResizeKernel(DeviceTensor input, DeviceImage input_image,
                             DeviceImage output_image, T* output, int64_t count,
                             InterpolationMode interpolation) {
  const T* input_data = static_cast<const T*>(input.data);
  // 两个方向的缩放因子在循环外算一次，所有下标复用。
  const double scale_y = static_cast<double>(input_image.h) / output_image.h;
  const double scale_x = static_cast<double>(input_image.w) / output_image.w;
  // grid-stride：步长为整个 grid 的线程数，保证 count 超出容量时全覆盖。
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    // 输出连续，故扁平下标 index 天然就是写回位置。
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(output_image, index, n, y, x, c);
    double value = 0.0;
    if (interpolation == InterpolationMode::kNearest) {
      // 最近邻：floor(dst*scale)，再截到 h-1/w-1。
      int64_t source_y = static_cast<int64_t>(y * scale_y);
      int64_t source_x = static_cast<int64_t>(x * scale_x);
      source_y = source_y < input_image.h ? source_y : input_image.h - 1;
      source_x = source_x < input_image.w ? source_x : input_image.w - 1;
      // 输入按步长寻址，兼容带填充或转置过的非连续张量。
      value = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, source_y, source_x, c)]);
    } else {
      // 双线性：半像素中心映射，clamp 下界到 0 后 floor。
      const double source_y = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
      const double source_x = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
      const double clamped_y = source_y > 0.0 ? source_y : 0.0;
      const double clamped_x = source_x > 0.0 ? source_x : 0.0;
      const int64_t y0 = static_cast<int64_t>(::floor(clamped_y));
      const int64_t x0 = static_cast<int64_t>(::floor(clamped_x));
      // 右下邻点越界时回退到最后一个有效行/列。
      const int64_t y1 = y0 + 1 < input_image.h ? y0 + 1 : input_image.h - 1;
      const int64_t x1 = x0 + 1 < input_image.w ? x0 + 1 : input_image.w - 1;
      const double wy = clamped_y - y0;
      const double wx = clamped_x - x0;
      const double top_left = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y0, x0, c)]);
      const double top_right = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y0, x1, c)]);
      const double bottom_left = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y1, x0, c)]);
      const double bottom_right = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y1, x1, c)]);
      // 先水平（上、下两行各按 wx），再垂直（按 wy 混合两行）。
      const double top = top_left + (top_right - top_left) * wx;
      const double bottom = bottom_left + (bottom_right - bottom_left) * wx;
      value = top + (bottom - top) * wy;
    }
    // 输出连续：直接用扁平下标写；u8 走 round + SaturatingCast。
    StoreImageValue(output, index, value);
  }
}

// 主机侧启动函数：按 dtype 把 ResizeKernel 实例化为 u8/f32/f64 三个版本，
// grid/block 分别为 BlockCount(count) 与 kThreadsPerBlock。
// 输入输出含义同 kernel；不支持的 dtype 返回 Unsupported。
// 启动后交给 FinishLaunch 检查并同步 stream。主机侧复杂度 O(1)。
Status LaunchResize(const DeviceTensor& input, const DeviceImage& input_image,
                    const DeviceImage& output_image, DataType dtype,
                    void* output, int64_t count,
                    InterpolationMode interpolation, cudaStream_t stream) {
  const unsigned int blocks = BlockCount(count);
  switch (dtype) {
    case DataType::kUInt8:
      ResizeKernel<uint8_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<uint8_t*>(output), count,
          interpolation);
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

// CUDA 后端入口：从 context 取 CudaStream，按布局解析输入 ImageInfo，
// 把输出的 h/w 换成参数中的目标尺寸，再转成 POD 描述符交给 LaunchResize。
// 输入 input、参数 params（目标高宽/插值/布局）、输出 output 均为
// 设备张量；第二个输入张量未使用。复杂度 O(rank)，计算在 GPU 上完成。
Status ResizeCuda(const OpContext& context, const Tensor& input, const Tensor*,
                  const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const ResizeParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  ImageInfo output_info = image_result.value();
  // 输出与输入同 rank/layout，仅 H/W 不同。
  output_info.h = options.height;
  output_info.w = options.width;
  return LaunchResize(Describe(input), Describe(image_result.value()),
                      Describe(output_info), input.dtype(), output->data(),
                      output->Numel(), options.interpolation,
                      stream_result.value()->handle());
}

}  // namespace ops_internal
}  // namespace mycv
