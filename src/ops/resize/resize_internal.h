#ifndef MYCV_SRC_OPS_RESIZE_RESIZE_INTERNAL_H_
#define MYCV_SRC_OPS_RESIZE_RESIZE_INTERNAL_H_

// resize 算子目录的内部契约（docs/architecture.md §6.1）：
// 集中声明缩放参数结构体、各后端的内核入口，以及注册钩子。
// 本文件只供 src/ops/resize/ 与内核注册表使用，不暴露给公共 API。

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// 一次 resize 调用的全部参数：目标高宽、插值方式、输入图像布局。
// layout 决定线性下标如何解码成 (n, y, x, c)，也决定输出张量的形状。
struct ResizeParams {
  int64_t height;
  int64_t width;
  InterpolationMode interpolation;
  ImageLayout layout;
};

// CPU 内核入口：naive 始终编译，OpenMP/TBB 仅在对应特性开启时存在。
// 每个入口定义在各自的 .cpp 中，内部共用 resize_cpu.h 的同一像素核，
// 因此各 level 对同一输入逐位完全一致。
Status ResizeCpuNaive(const OpContext& context, const Tensor& input,
                      const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status ResizeCpuOpenMP(const OpContext&, const Tensor&, const Tensor*,
                       const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status ResizeCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*,
                    Tensor*);
#endif
// 可选后端入口：CUDA 与 OpenCV，签名与 CPU 入口一致，均只接受 kNaive
// 级别（见 ValidateInvocation 对非 CPU 后端的限制）。
#ifdef MYCV_WITH_CUDA
Status ResizeCuda(const OpContext&, const Tensor&, const Tensor*, const void*,
                  Tensor*);
#endif
#ifdef MYCV_WITH_OPENCV
Status ResizeOpenCv(const OpContext&, const Tensor&, const Tensor*, const void*,
                    Tensor*);
#endif

// 把本二进制内已编译的 resize 实现登记到内核表（docs/architecture.md
// §6.3）：所有启用的 CPU level，加上启用的可选后端（CUDA/OpenCV）。
void RegisterResizeKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_RESIZE_RESIZE_INTERNAL_H_
