#ifndef MYCV_OPS_OP_H_
#define MYCV_OPS_OP_H_

#include <cstdint>

#include "mycv/core/context.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/stream.h"

namespace mycv {

// Per-call execution descriptor passed to every operator.
struct OpContext {
  Context* context = nullptr;
  Stream* stream = nullptr;  // nullptr -> use Context's default stream
  CpuKernelLevel cpu_level = CpuKernelLevel::kNaive;
};

// Stable op ids; new ops append and never renumber.
enum class OpKind : uint16_t {
  kResize = 1,
  kGaussianBlur = 2,
  kLaplacianPyramid = 3,
  kFourierTransform = 4,
  kWaveletTransform = 5,
};

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
