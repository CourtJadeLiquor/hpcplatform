#ifndef hpcplatform_SRC_FRAMEWORK_PROFILING_H_
#define hpcplatform_SRC_FRAMEWORK_PROFILING_H_

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "hpcplatform/core/device.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/ops/op.h"

// Optional operator profiling (docs/architecture.md §6.4). Compile the target
// with -Dhpcplatform_ENABLE_PROFILING=1 to turn the macros below into real timers;
// the default build expands them to no-op statements, so profiling costs
// nothing when disabled. Results feed the per-operator benchmark/ programs and
// the numbers recorded in each operator's README.md.

#if defined(hpcplatform_ENABLE_PROFILING) && hpcplatform_ENABLE_PROFILING

namespace hpcplatform {
namespace profiling {

struct ProfileEntry {
  uint64_t hit_count = 0;
  int64_t total_nanoseconds = 0;
};

// Aggregation store shared across translation units (single instance via
// C++17 inline-function static semantics).
struct ProfileStore {
  std::map<std::string, ProfileEntry> entries;
  std::mutex mutex;
};

inline ProfileStore& Store() {
  static ProfileStore store;
  return store;
}

// Records one timing sample. Thread-safe: profiling may run inside OpenMP/TBB
// parallel regions.
inline void ProfileRecord(const std::string& label, int64_t nanoseconds) {
  ProfileStore& store = Store();
  std::lock_guard<std::mutex> lock(store.mutex);
  ProfileEntry& entry = store.entries[label];
  entry.hit_count += 1;
  entry.total_nanoseconds += nanoseconds;
}

// Returns aggregated results for a label; returns zero values if no samples were
// recorded.
inline ProfileEntry ProfileReport(const std::string& label) {
  ProfileStore& store = Store();
  std::lock_guard<std::mutex> lock(store.mutex);
  return store.entries[label];
}

// Clears all aggregates so independent benchmark runs do not affect each other.
inline void ProfileReset() {
  ProfileStore& store = Store();
  std::lock_guard<std::mutex> lock(store.mutex);
  store.entries.clear();
}

// Builds a dispatch-triple label in "op/backend/level" form, e.g.
// "resize/native_cpu/avx2".
inline std::string OpLabel(OpKind op, BackendType backend,
                           CpuKernelLevel level) {
  return std::string(OpKindName(op)) + "/" + BackendTypeName(backend) + "/" +
         CpuKernelLevelName(level);
}

// RAII scope timer; records (label, elapsed) on destruction.
class ScopeTimer {
 public:
  explicit ScopeTimer(std::string label)
      : label_(std::move(label)), start_(Clock::now()) {}
  ScopeTimer(const ScopeTimer&) = delete;
  ScopeTimer& operator=(const ScopeTimer&) = delete;
  ~ScopeTimer() {
    const auto stop = Clock::now();
    ProfileRecord(label_, std::chrono::duration_cast<std::chrono::nanoseconds>(
                              stop - start_)
                              .count());
  }

 private:
  using Clock = std::chrono::steady_clock;
  std::string label_;
  Clock::time_point start_;
};

}  // namespace profiling
}  // namespace hpcplatform

#define hpcplatform_PROFILE_CONCAT_(a, b) a##b
#define hpcplatform_PROFILE_CONCAT(a, b) hpcplatform_PROFILE_CONCAT_(a, b)

// Times the enclosing scope under `label`.
#define hpcplatform_PROFILE_SCOPE(label)                                \
  ::hpcplatform::profiling::ScopeTimer hpcplatform_PROFILE_CONCAT(hpcplatform_scope_, \
                                                    __LINE__)(label)

// Times the enclosing scope, labeled by the dispatch triple.
#define hpcplatform_PROFILE_OP(op_kind, backend, level) \
  hpcplatform_PROFILE_SCOPE(::hpcplatform::profiling::OpLabel(op_kind, backend, level))

#else  // profiling disabled: zero cost

#define hpcplatform_PROFILE_SCOPE(label) ((void)0)
#define hpcplatform_PROFILE_OP(op_kind, backend, level) ((void)0)

#endif  // hpcplatform_ENABLE_PROFILING

#endif  // hpcplatform_SRC_FRAMEWORK_PROFILING_H_
