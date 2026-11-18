#include "mycv/core/error.h"

namespace mycv {

const char* StatusCodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "OK";
    case StatusCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case StatusCode::kOutOfMemory: return "OUT_OF_MEMORY";
    case StatusCode::kUnsupported: return "UNSUPPORTED";
    case StatusCode::kNotImplemented: return "NOT_IMPLEMENTED";
    case StatusCode::kBadState: return "BAD_STATE";
    case StatusCode::kInternal: return "INTERNAL";
  }
  return "UNKNOWN";
}

std::string Status::ToString() const {
  std::string result = StatusCodeName(code_);
  if (!message_.empty()) {
    result += ": ";
    result += message_;
  }
  return result;
}

}  // namespace mycv
