#include "backends/backend_internal.h"

namespace hpcplatform {

// Placeholder backend: the NPU abstraction layer is in place so that upper
// layers can already be written against it, but no vendor SDK is wired in yet.
// Replacing this file with a real ACL/RKNN implementation must not require
// changes outside src/backends/npu (see docs/architecture.md §6.2).
Result<std::shared_ptr<Backend>> CreateNpuBackend(int device_index) {
  (void)device_index;
  return Status::Unsupported(
      "npu backend is a compile-time placeholder in this build "
      "(see docs/architecture.md §6.2)");
}

}  // namespace hpcplatform
