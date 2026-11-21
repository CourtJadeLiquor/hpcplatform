#ifndef MYCV_CORE_DTYPE_H_
#define MYCV_CORE_DTYPE_H_

#include <cstddef>
#include <cstdint>

namespace mycv {

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

// 返回元素字节数；未知类型返回 0。
size_t DTypeSize(DataType dtype);
// 返回稳定的类型名（用于日志与错误信息）。
const char* DTypeName(DataType dtype);
// 浮点类型（f16/f32/f64）返回 true。
bool IsFloatingPoint(DataType dtype);

}  // namespace mycv

#endif  // MYCV_CORE_DTYPE_H_
