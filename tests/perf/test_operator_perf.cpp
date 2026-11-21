// 测试接口用于评估 resize 算子的性能：预热后多轮计时，并通过 profiling/stats
// 设施核对耗时聚合与内存峰值/泄漏统计。
#include "test_framework.h"

#include <chrono>
#include <cstdint>
#include <iostream>

#include "framework/profiling.h"
#include "framework/stats.h"
#include "mycv/mycv.h"

namespace {

using namespace mycv;

constexpr int64_t kInputHeight = 512;
constexpr int64_t kInputWidth = 512;
constexpr int64_t kChannels = 3;
constexpr int kIterations = 20;
constexpr size_t kOutputBytes = static_cast<size_t>(256 * 256 * 3 * 4);

OpContext CpuOpContext(Context& context) {
  OpContext op_context;
  op_context.context = &context;
  op_context.stream = &context.stream();
  return op_context;
}

MYCV_TEST(ResizePerformanceAndMemorySnapshot) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input =
      context
          ->AllocateTensor(Shape{1, kInputHeight, kInputWidth, kChannels},
                           DataType::kFloat32)
          .value();
  float* input_data = input.Data<float>();
  const int64_t numel = input.shape().Numel();
  for (int64_t i = 0; i < numel; ++i) {
    input_data[i] = static_cast<float>(i % 251) / 255.0f;
  }

  const OpContext op_context = CpuOpContext(*context);
  // 预热：排除首次分配、页错误与缓存冷启动对计时的影响。
  MYCV_EXPECT_OK(
      Resize(op_context, input, kInputHeight / 2, kInputWidth / 2).status());

  const ops_internal::MemorySnapshot before =
      ops_internal::SnapshotMemory(*context);
  const auto start = std::chrono::steady_clock::now();
  {
    MYCV_PROFILE_OP(OpKind::kResize, BackendType::kNativeCpu,
                    CpuKernelLevel::kNaive);
    for (int i = 0; i < kIterations; ++i) {
      auto output =
          Resize(op_context, input, kInputHeight / 2, kInputWidth / 2);
      MYCV_EXPECT_TRUE(output.ok());
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(stop - start).count();
  const ops_internal::MemorySnapshot after =
      ops_internal::SnapshotMemory(*context);

  MYCV_EXPECT_TRUE(elapsed_ms > 0.0);
  MYCV_EXPECT_TRUE(after.host.peak_bytes >=
                   before.host.allocated_bytes + kOutputBytes);
  // 每轮输出都已释放：活动分配与计数应回到基线，缓存块不计入。
  MYCV_EXPECT_EQ(after.host.allocated_bytes, before.host.allocated_bytes);
  MYCV_EXPECT_EQ(after.host.allocation_count, before.host.allocation_count);
  MYCV_EXPECT_EQ(after.device.allocated_bytes, static_cast<size_t>(0));

  std::cout << "  [ PERF ] resize " << kInputWidth << "x" << kInputHeight
            << " -> " << kInputWidth / 2 << "x" << kInputHeight / 2 << ": "
            << elapsed_ms / kIterations
            << " ms/iter, peak_host=" << after.host.peak_bytes << " B\n";

#if defined(MYCV_ENABLE_PROFILING) && MYCV_ENABLE_PROFILING
  const profiling::ProfileEntry entry =
      profiling::ProfileReport(profiling::OpLabel(
          OpKind::kResize, BackendType::kNativeCpu, CpuKernelLevel::kNaive));
  MYCV_EXPECT_EQ(entry.hit_count, static_cast<uint64_t>(1));
  MYCV_EXPECT_TRUE(entry.total_nanoseconds > 0);
  std::cout << "  [ PROFILE ] resize/native_cpu/naive: "
            << static_cast<double>(entry.total_nanoseconds) / 1e6
            << " ms total\n";
#endif
}

}  // namespace

MYCV_TEST_MAIN()
