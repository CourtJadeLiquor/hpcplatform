#ifndef MYCV_OPS_OP_H_
#define MYCV_OPS_OP_H_

#include <cstdint>

#include "mycv/core/context.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/stream.h"

namespace mycv {

// Execution descriptor for each operator call: owning Context, optional stream,
// and CPU kernel level.
struct OpContext {
  Context* context = nullptr;  // Must be non-null
  Stream* stream = nullptr;    // nullptr means use the Context's default stream
  CpuKernelLevel cpu_level = CpuKernelLevel::kNaive;  // CPU backends only
};

// Stable operator IDs: append only, never reorder; numeric values remain stable
// across versions.
enum class OpKind : uint16_t {
  kResize = 1,
  kGaussianBlur = 2,
  kLaplacianPyramid = 3,
  kFourierTransform = 4,
  kWaveletTransform = 5,
};

// Returns the operator's stable short name for logs, profiling labels, and errors.
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
