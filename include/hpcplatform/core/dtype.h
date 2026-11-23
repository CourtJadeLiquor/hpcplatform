#ifndef hpcplatform_CORE_DTYPE_H_
#define hpcplatform_CORE_DTYPE_H_

#include <cstddef>
#include <cstdint>

namespace hpcplatform {

// Element types supported by the core. Backends may support only a subset;
// availability is queried at op level.
enum class DataType : uint8_t {
  kUInt8 = 0,
  kInt8 = 1,
  kUInt16 = 2,
  kInt16 = 3,
  kFloat16 = 4,
  kFloat32 = 5,
  kFloat64 = 6,
  kInt32 = 7,
  kInt64 = 8,
};

// Returns the element size in bytes; returns 0 for unknown types.
size_t DTypeSize(DataType dtype);
// Returns a stable type name for logs and error messages.
const char* DTypeName(DataType dtype);
// Returns true for floating-point types (f16/f32/f64).
bool IsFloatingPoint(DataType dtype);

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_DTYPE_H_
