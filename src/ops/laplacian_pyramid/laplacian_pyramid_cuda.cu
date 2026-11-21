// laplacian_pyramid 的 CUDA 实现：设备侧组合子算子 + 一个减法内核。
//
// 逐级流水线与主机版一致（纯文本）：
//   G_{i+1} = ResizeNearest(GaussianBlur(G_i))   尺寸 ceil(H/2) x ceil(W/2)
//   up      = ResizeBilinear(G_{i+1}, H_i, W_i)
//   L_i     = G_i - up                          与 G_i 同尺寸
//   最顶层  L_n = G_n
// 高斯与两次缩放分别调用已注册的 CUDA 子算子；只有最后的逐元素减法需要
// 本文件自己的 kernel（SubtractBandsKernel）。
//
// 精度：减法本身是精确的浮点减法，但子算子在 device 上可能做 FMA 收缩，
// 与主机路径存在末位差异，因此跨后端比较采用 1e-4 相对容差。

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

#include "backends/cuda/cuda_device_utils.h"
#include "mycv/ops/gaussian_blur.h"
#include "mycv/ops/resize.h"
#include "laplacian_pyramid_internal.h"

namespace mycv {
namespace ops_internal {
namespace {

// 逐元素相减：output[index] = fine[index] - up[index]，全部位于设备内存。
// 线程划分：每块 kThreadsPerBlock 个线程、共 BlockCount(count) 块，采用
//   grid-stride 循环，元素数超过一个 wave 时仍能覆盖全部下标。
// 索引：up 与 output 是刚分配的连续张量，可直接用线性下标 index；fine 可能
//   是带 stride 的视图（调用者输入或子算子输出），故用 StridedElementOffset
//   按维度解算真实元素偏移。输出连续，所以直接线性写回。
// 复杂度：O(count)，每个元素一次减法。
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

// 按 dtype 分发 f32/f64 版本的 SubtractBandsKernel。块数由
// BlockCount(Numel) 给出（256 线程/块，上限 kMaxBlocks），grid-stride
// 循环负责余量。FinishLaunch 先检查 cudaGetLastError 捕获启动失败，再
// cudaStreamSynchronize 等待完成并使运行时错误可见；返回 Ok 时输出可用。
// 复杂度 O(Numel)，由 SubtractBandsKernel 承担。
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

// CUDA stage：一级的完整组合路径。先取流（context.stream 或 Context 默认
// 流），各子算子在同一流上顺序执行；随后 LaunchSubtract 计算 L_i。
// 输入/输出与主机版相同：output 为 L_i，params->next_gaussian 为 G_{i+1}。
// 尺寸：下一级取 (h+1)/2、(w+1)/2（ceil 半尺寸），上采样目标是当前 h x w。
// 复杂度：O(N * kernel_size)；与主机结果按 1e-4 相对容差对齐。
Status LaplacianPyramidCudaStage(const OpContext& context, const Tensor& input,
                                 const Tensor* /*other*/, const void* params,
                                 Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const LaplacianPyramidParams*>(params);
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();

  // 下一级为 ceil 半尺寸，保证奇数边长也有确定的 G_{i+1} 形状。
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
  // 子算子输出 up 已对齐到当前层尺寸，这里只需逐元素相减。
  MYCV_RETURN_IF_ERROR(LaunchSubtract(input, up.value(), output,
                                      stream_result.value()->handle()));

  *options.next_gaussian = std::move(next.value());
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
