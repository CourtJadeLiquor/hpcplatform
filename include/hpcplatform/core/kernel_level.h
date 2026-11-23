#ifndef hpcplatform_CORE_KERNEL_LEVEL_H_
#define hpcplatform_CORE_KERNEL_LEVEL_H_

#include <cstdint>
#include <string>

#include "hpcplatform/core/error.h"

namespace hpcplatform {

// Selects which CPU implementation of an op runs. This is an explicit,
// compile-time-enabled knob (no auto-tuning, no CPU feature probing).
// Non-CPU backends accept kNaive only and use their own best implementation.
enum class CpuKernelLevel : uint8_t {
  kNaive = 0,        // portable reference implementation, used as ground truth
  kOpenMP = 1,       // thread-parallel reference implementation
  kSimdGeneric = 2,  // portable SIMD (compiler vectorization / intrinsics)
  kSimdAvx2 = 3,     // x86-64 AVX2
  kSimdNeon = 4,     // aarch64 NEON
  kTbb = 5,          // oneTBB thread-parallel implementation
};

inline const char* CpuKernelLevelName(CpuKernelLevel level) {
  switch (level) {
    case CpuKernelLevel::kNaive:
      return "naive";
    case CpuKernelLevel::kOpenMP:
      return "openmp";
    case CpuKernelLevel::kSimdGeneric:
      return "simd_generic";
    case CpuKernelLevel::kSimdAvx2:
      return "simd_avx2";
    case CpuKernelLevel::kSimdNeon:
      return "simd_neon";
    case CpuKernelLevel::kTbb:
      return "tbb";
  }
  return "unknown";
}

// Parses a stable CPU kernel level name back to its enum. Inverse of
// CpuKernelLevelName (excluding "unknown"); unknown name -> kInvalidArgument.
inline Result<CpuKernelLevel> CpuKernelLevelFromName(const std::string& name) {
  if (name == "naive") return CpuKernelLevel::kNaive;
  if (name == "openmp") return CpuKernelLevel::kOpenMP;
  if (name == "simd_generic") return CpuKernelLevel::kSimdGeneric;
  if (name == "simd_avx2") return CpuKernelLevel::kSimdAvx2;
  if (name == "simd_neon") return CpuKernelLevel::kSimdNeon;
  if (name == "tbb") return CpuKernelLevel::kTbb;
  return Status::InvalidArgument("unknown cpu kernel level name: " + name);
}

// Returns true for SIMD levels (generic / AVX2 / NEON).
inline bool IsSimdKernelLevel(CpuKernelLevel level) {
  return level == CpuKernelLevel::kSimdGeneric ||
         level == CpuKernelLevel::kSimdAvx2 ||
         level == CpuKernelLevel::kSimdNeon;
}

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_KERNEL_LEVEL_H_
