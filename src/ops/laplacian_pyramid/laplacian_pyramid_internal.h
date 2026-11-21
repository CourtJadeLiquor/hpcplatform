#ifndef MYCV_SRC_OPS_LAPLACIAN_PYRAMID_LAPLACIAN_PYRAMID_INTERNAL_H_
#define MYCV_SRC_OPS_LAPLACIAN_PYRAMID_LAPLACIAN_PYRAMID_INTERNAL_H_

// laplacian_pyramid 算子目录的内部契约（docs/architecture.md §6.1/§11）。
// 本算子没有自己的像素级数学：一级（stage）由已注册的 GaussianBlur、
// Resize 两个子算子加一次逐元素减法组合而成，因此它的“版本”就是该级
// 转发给两个子算子的 (backend, CpuKernelLevel) 对。
// 本文件只声明参数结构与主机/设备 stage 入口，不包含实现；两个实现分别在
// laplacian_pyramid_host.cpp 与 laplacian_pyramid_cuda.cu 中。
//
// 逐级语义（纯文本）：
//   G_{i+1} = ResizeNearest(GaussianBlur(G_i))   尺寸 ceil(H/2) x ceil(W/2)
//   L_i     = G_i - upsample(G_{i+1})            与 G_i 同尺寸
//   最顶层  L_n = G_n（最粗高斯层直接作为低频残差）
// 重建：G_i = L_i + upsample(G_{i+1})。

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// 一级（stage）的不可变参数，生命周期仅限于一次同步内核调用。
struct LaplacianPyramidParams {
  int kernel_size;     // GaussianBlur 的核尺寸：>= 3 的奇数
  double sigma;        // GaussianBlur 的标准差：有限且 > 0
  ImageLayout layout;  // 图像布局：NHWC 或 NCHW
  // 输出槽：下一级（半尺寸）高斯层 G_{i+1}。stage 负责分配并 move 赋值到
  // 该槽；公共入口在调用期间保持槽位存活。
  Tensor* next_gaussian;
};

// 主机侧 stage：把 L_i 写入 output，把 G_{i+1} 写入 params->next_gaussian。
// 输入：context（后端/级别/stream）、G_i（input）、params；other 不使用。
// 输出：output 为带通 L_i，尺寸与 input 相同；next_gaussian 为下级高斯层。
// 步骤：GaussianBlur -> ResizeNearest 到 ceil(H/2) x ceil(W/2) 得 G_{i+1}，
//       再 ResizeBilinear 回 G_i 尺寸并逐元素相减。
// 复杂度：每级 O(N_i * kernel_size) 的高斯 + O(N_i) 的缩放与减法。
Status LaplacianPyramidHostStage(const OpContext& context, const Tensor& input,
                                 const Tensor* other, const void* params,
                                 Tensor* output);
#ifdef MYCV_WITH_CUDA
// 设备侧 stage：逐级语义与主机版相同，子算子走 CUDA，最后用设备内核相减。
// 要求 OpContext 能提供 CudaStream（context.stream 或 context.context 的流）。
// 复杂度与主机版一致；跨后端数值按 1e-4 相对容差比较。
Status LaplacianPyramidCudaStage(const OpContext& context, const Tensor& input,
                                 const Tensor* other, const void* params,
                                 Tensor* output);
#endif

// 注册本二进制内建的所有版本。只注册 resize 与 gaussian_blur 都实现的级别
// （当前没有 SIMD，因为 resize 没有 SIMD 内核），因此不支持的级别在查表时
// 就失败，而不是在逐级执行到一半才失败。复杂度 O(1)，仅写注册表。
void RegisterLaplacianPyramidKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_LAPLACIAN_PYRAMID_LAPLACIAN_PYRAMID_INTERNAL_H_
