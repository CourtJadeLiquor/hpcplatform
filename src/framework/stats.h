#ifndef MYCV_SRC_FRAMEWORK_STATS_H_
#define MYCV_SRC_FRAMEWORK_STATS_H_

#include "mycv/core/context.h"
#include "mycv/core/device.h"
#include "mycv/core/memory.h"

namespace mycv {
namespace ops_internal {

// Memory usage snapshot grouped by MemoryKind, for operator analysis and
// benchmarks (docs/architecture.md §6.4). Each Context owns exactly one
// MemoryManager, so the snapshot reports its aggregate under the bucket that
// matches the manager's device type; host_pinned stays zero until a backend
// exposes a dedicated pinned-host manager.
struct MemorySnapshot {
  MemoryStats host;         // pageable host allocations
  MemoryStats host_pinned;  // page-locked host allocations (none reports it yet)
  MemoryStats device;       // device (GPU/NPU) allocations, i.e. VRAM usage
};

inline MemorySnapshot SnapshotMemory(Context& context) {
  MemorySnapshot snapshot;
  const MemoryStats stats = context.memory().Stats();
  switch (context.memory().device().type) {
    case DeviceType::kCPU:
      snapshot.host = stats;
      break;
    case DeviceType::kGPU:
    case DeviceType::kNPU:
      snapshot.device = stats;
      break;
  }
  return snapshot;
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_FRAMEWORK_STATS_H_
