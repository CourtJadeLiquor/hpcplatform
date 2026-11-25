// Tests for Buffer / MemoryManager allocation, alignment, counting, and
// deallocation (including custom deleters). Uses only public Context APIs —
// transfer::Copy for data movement and Context::GetMemoryStats for counters.
// Low-level MemoryManager byte-range Copy and direct Allocate are verified in
// the backend implementation tests (see src/backends/*).
#include "test_utils.h"

#include <cstring>
#include <vector>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/core/transfer.h"

namespace {

using namespace hpcplatform;

TEST(hpcplatform, CpuMemoryManagerCopiesBetweenBuffers) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto src = Tensor::Empty(*context, Shape{4}, DataType::kFloat32).value();
  auto dst = Tensor::Empty(*context, Shape{4}, DataType::kFloat32).value();

  float* src_data = src.Data<float>();
  for (int i = 0; i < 4; ++i)
    src_data[i] = static_cast<float>(i) + 1.0f;

  hpcplatform_EXPECT_OK(Copy(&dst, src));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.ByteSize()),
                 static_cast<int>(0));
  EXPECT_EQ(dst.Data<float>()[3], 4.0f);
}

TEST(hpcplatform, MemoryStatsTrackLifetime) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  const MemoryStats before = context->GetMemoryStats();

  {
    auto first = Tensor::Empty(*context, Shape{8}, DataType::kFloat32).value();
    auto second = Tensor::Empty(*context, Shape{8}, DataType::kFloat32).value();
    const MemoryStats during = context->GetMemoryStats();
    EXPECT_EQ(during.allocated_bytes, before.allocated_bytes + 64);
    EXPECT_EQ(during.allocation_count, before.allocation_count + 2);
    EXPECT_TRUE(during.peak_bytes >= during.allocated_bytes);
  }

  const MemoryStats after = context->GetMemoryStats();
  EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
  EXPECT_EQ(after.allocation_count, before.allocation_count);
}

}  // namespace
