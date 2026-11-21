// resize 算子流水线的入口部分：内核注册与公共 Resize() 入口。
// RegisterResizeKernels 只登记函数指针；Resize() 负责与后端无关的校验、
// 形状推导、输出分配和分派。真正的缩放核计算见各后端实现文件
// （CPU 共用 resize_cpu.h，CUDA/OpenCV 各自独立）。本文件不含插值公式。

#include "mycv/ops/resize.h"

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "resize_internal.h"

namespace mycv {
namespace ops_internal {

// 内核注册：naive 是 CPU 的必选实现；OpenMP/TBB 挂在 kNativeCpu 的
// 对应 level 上；CUDA/OpenCV 作为独立后端，只提供 kNaive 级别
// （非 CPU 后端不接受其他 level，见 ValidateInvocation）。
// 复杂度 O(1)，编译期条件决定登记哪些条目。
void RegisterResizeKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kResize, cpu, CpuKernelLevel::kNaive, &ResizeCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kResize, cpu, CpuKernelLevel::kOpenMP, &ResizeCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kResize, cpu, CpuKernelLevel::kTbb, &ResizeCpuTbb);
#endif
#ifdef MYCV_WITH_CUDA
  RegisterKernel(OpKind::kResize, BackendType::kCUDA, CpuKernelLevel::kNaive,
                 &ResizeCuda);
#endif
#ifdef MYCV_WITH_OPENCV
  RegisterKernel(OpKind::kResize, BackendType::kOpenCV, CpuKernelLevel::kNaive,
                 &ResizeOpenCv);
#endif
}

}  // namespace ops_internal

// resize 的公共入口，返回新分配的输出张量。
// 输入：上下文、输入图像张量、目标高宽、插值方式与布局。
// 步骤：1) 校验 context 与输入张量的后端/设备归属；
//       2) 按 layout 解析 rank-2/3/4 形状得到 ImageInfo；
//       3) 只接受 u8/f32/f64（其余返回 Unsupported），目标高宽须为正；
//       4) 按 input.backend() 与 context.cpu_level 查找已注册内核；
//       5) 保持 rank/layout 推导输出形状并分配；
//       6) 打包 ResizeParams 调用内核。
// 复杂度 O(1)（仅形状与分派），像素开销全部在内核中。
Result<Tensor> Resize(const OpContext& context, const Tensor& input,
                      int64_t output_height, int64_t output_width,
                      InterpolationMode interpolation, ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  // dtype 白名单：三种后端共用同一份支持列表。
  if (!ops_internal::IsSupportedImageDType(input.dtype())) {
    return Status::Unsupported("Resize supports u8, f32 and f64 images");
  }
  if (output_height <= 0 || output_width <= 0) {
    return Status::InvalidArgument("Resize output dimensions must be positive");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kResize, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  // 输出保持输入的 rank 与 layout，只替换 H/W。
  const Shape output_shape =
      ops_internal::MakeImageShape(image.value(), output_height, output_width);
  auto output = ops_internal::AllocateOutput(context, output_shape, input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::ResizeParams params{output_height, output_width,
                                          interpolation, layout};
  // 单输入算子：第二个张量传 nullptr。
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, &params,
                                      &output.value()));
  return output;
}

}  // namespace mycv
