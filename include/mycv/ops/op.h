#ifndef MYCV_OPS_OP_H_
#define MYCV_OPS_OP_H_

#include <cstdint>

#include "mycv/core/context.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/stream.h"

namespace mycv {

// 每次算子调用携带的执行描述：所属 Context、可选流与 CPU 内核级别。
struct OpContext {
  Context* context = nullptr;  // 必须非空
  Stream* stream = nullptr;    // nullptr 表示使用 Context 的默认流
  CpuKernelLevel cpu_level = CpuKernelLevel::kNaive;  // 仅 CPU 后端生效
};

// 稳定的算子编号：只追加、不重排，跨版本保持数值稳定。
enum class OpKind : uint16_t {
  kResize = 1,
  kGaussianBlur = 2,
  kLaplacianPyramid = 3,
  kFourierTransform = 4,
  kWaveletTransform = 5,
};

// 返回算子的稳定短名，用于日志、剖析标签与错误信息。
inline const char* OpKindName(OpKind kind) {
  switch (kind) {
    case OpKind::kResize:
      return "resize";
    case OpKind::kGaussianBlur:
      return "gaussian_blur";
    case OpKind::kLaplacianPyramid:
      return "laplacian_pyramid";
    case OpKind::kFourierTransform:
      return "fourier_transform";
    case OpKind::kWaveletTransform:
      return "wavelet_transform";
  }
  return "unknown";
}

}  // namespace mycv

#endif  // MYCV_OPS_OP_H_
