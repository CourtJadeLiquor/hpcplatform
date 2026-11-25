// Performance tests for resize: multiple timed iterations after warm-up, with
// profiling/stats checks for timing aggregation and memory peak/leak statistics.
#include "test_utils.h"

#include <chrono>
#include <cstdint>
#include <iostream>

#include "profiling.h"
#include "stats.h"
#include "hpcplatform/hpcplatform.h"

namespace {

using namespace hpcplatform;

constexpr int64_t kInputHeight = 512;
constexpr int64_t kInputWidth = 512;
constexpr int64_t kChannels = 3;
constexpr int kIterations = 20;
constexpr size_t kOutputBytes = static_cast<size_t>(256 * 256 * 3 * 4);

OpContext CpuOpContext(Context& context) {
  OpContext op_context(context);
  op_context.set_cpu_level(CpuKernelLevel::kNaive);
  return op_context;
}

TEST(hpcplatform, ResizePerformanceAndMemorySnapshot) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input =
      Tensor::Empty(*context, Shape{1, kInputHeight, kInputWidth, kChannels},
                           DataType::kFloat32)
          .value();
  float* input_data = input.Data<float>();
  const int64_t numel = input.shape().Numel();
  for (int64_t i = 0; i < numel; ++i) {
    input_data[i] = static_cast<float>(i % 251) / 255.0f;
  }

  const OpContext op_context = CpuOpContext(*context);
  // Warm-up: exclude first allocation, page faults, and cold-cache effects from timing.
  hpcplatform_EXPECT_OK(
      Resize(op_context, input, kInputHeight / 2, kInputWidth / 2).status());

  const ops_internal::MemorySnapshot before =
      ops_internal::SnapshotMemory(*context);
  const auto start = std::chrono::steady_clock::now();
  {
    hpcplatform_PROFILE_OP(OpKind::kResize, BackendType::kNativeCpu,
                    CpuKernelLevel::kNaive);
    for (int i = 0; i < kIterations; ++i) {
      auto output =
          Resize(op_context, input, kInputHeight / 2, kInputWidth / 2);
      EXPECT_TRUE(output.ok());
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(stop - start).count();
  const ops_internal::MemorySnapshot after =
      ops_internal::SnapshotMemory(*context);

  EXPECT_TRUE(elapsed_ms > 0.0);
  EXPECT_TRUE(after.host.peak_bytes >=
                   before.host.allocated_bytes + kOutputBytes);
  // Outputs are released each iteration: active allocations and counts return to baseline; cached blocks are excluded.
  EXPECT_EQ(after.host.allocated_bytes, before.host.allocated_bytes);
  EXPECT_EQ(after.host.allocation_count, before.host.allocation_count);
  EXPECT_EQ(after.device.allocated_bytes, static_cast<size_t>(0));

  std::cout << "  [ PERF ] resize " << kInputWidth << "x" << kInputHeight
            << " -> " << kInputWidth / 2 << "x" << kInputHeight / 2 << ": "
            << elapsed_ms / kIterations << " ms/iter, peak_host="
            << static_cast<double>(after.host.peak_bytes) / (1024.0 * 1024.0)
            << " MB\n";

#if defined(hpcplatform_ENABLE_PROFILING) && hpcplatform_ENABLE_PROFILING
  const profiling::ProfileEntry entry =
      profiling::ProfileReport(profiling::OpLabel(
          OpKind::kResize, BackendType::kNativeCpu, CpuKernelLevel::kNaive));
  EXPECT_EQ(entry.hit_count, static_cast<uint64_t>(1));
  EXPECT_TRUE(entry.total_nanoseconds > 0);
  std::cout << "  [ PROFILE ] resize/native_cpu/naive: "
            << static_cast<double>(entry.total_nanoseconds) / 1e6
            << " ms total\n";
#endif
}

}  // namespace

