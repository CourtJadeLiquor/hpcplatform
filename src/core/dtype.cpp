#include "mycv/core/dtype.h"

namespace mycv {

size_t DTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
    case DataType::kInt8:
      return 1;
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kFloat16:
      return 2;
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kFloat64:
    case DataType::kInt64:
      return 8;
  }
  return 0;
}

const char* DTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return "uint8";
    case DataType::kInt8:
      return "int8";
    case DataType::kUInt16:
      return "uint16";
    case DataType::kInt16:
      return "int16";
    case DataType::kFloat16:
      return "float16";
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat64:
      return "float64";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
  }
  return "unknown";
}

bool IsFloatingPoint(DataType dtype) {
  return dtype == DataType::kFloat16 || dtype == DataType::kFloat32 ||
         dtype == DataType::kFloat64;
}

}  // namespace mycv
