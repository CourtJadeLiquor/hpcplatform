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
// 按 MemoryKind 分桶的内存快照。每个 Context 只持有一个 MemoryManager，
// 因此统计值会落在其设备类型对应的桶中；host_pinned 目前恒为零，直到某个
// 后端提供独立的页锁定主机分配器。
struct MemorySnapshot {
  MemoryStats host;         // 可分页主机内存
  MemoryStats host_pinned;  // 页锁定主机内存（暂无后端上报）
  MemoryStats device;       // 设备内存（GPU/NPU），即显存占用
};

// 读取 Context 当前内存统计并归入对应桶。
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
