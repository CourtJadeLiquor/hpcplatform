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
  kAdd = 1,
  kMul = 2,
  kCast = 3,
  kNormalize = 4,
  kResize = 5,
  kCrop = 6,
  kTranspose = 7,
  kWarpAffine = 8,
};

inline const char* OpKindName(OpKind kind) {
  switch (kind) {
    case OpKind::kAdd: return "add";
    case OpKind::kMul: return "mul";
    case OpKind::kCast: return "cast";
    case OpKind::kNormalize: return "normalize";
    case OpKind::kResize: return "resize";
    case OpKind::kCrop: return "crop";
    case OpKind::kTranspose: return "transpose";
    case OpKind::kWarpAffine: return "warp_affine";
  }
  return "unknown";
}

}  // namespace mycv

#endif  // MYCV_OPS_OP_H_
