// Tests for backend factory selection, CPU backend capabilities
// (dtype/copy/stream), and operator-context execution-level constraints.
#include "test_utils.h"

#include "core/backend.h"
#include "hpcplatform/core/context.h"
#include "hpcplatform/ops/op.h"

namespace {

using namespace hpcplatform;

TEST(hpcplatform, CpuBackendIsExplicitlySelected) {
  auto backend = CreateBackend(BackendType::kNativeCpu, 0);
  EXPECT_TRUE(backend.ok());
  EXPECT_EQ(backend.value()->type(), BackendType::kNativeCpu);
  EXPECT_EQ(backend.value()->device().type, DeviceType::kCPU);
  EXPECT_TRUE(
      backend.value()->SupportsKernelLevel(CpuKernelLevel::kNaive));
  if (backend.value()->SupportsKernelLevel(CpuKernelLevel::kSimdAvx2)) {
    EXPECT_TRUE(backend.value()->Describe().find("simd_avx2") !=
                     std::string::npos);
  }
  EXPECT_TRUE(backend.value()->Describe().find("naive") !=
                   std::string::npos);
}

TEST(hpcplatform, UnavailableBackendsFailWithoutFallback) {
  const BackendType unavailable[] = {
#ifndef hpcplatform_WITH_OPENCV
      BackendType::kOpenCV,
#endif
#ifndef hpcplatform_WITH_CUDA
      BackendType::kCUDA,
#endif
      BackendType::kNPU};
  for (BackendType type : unavailable) {
    auto backend = CreateBackend(type);
    EXPECT_FALSE(backend.ok());
    EXPECT_EQ(backend.status().code(), StatusCode::kUnsupported);
    EXPECT_TRUE(Context::Create(type).ok() == false);
  }
}

#ifdef hpcplatform_WITH_OPENCV
TEST(hpcplatform, OpenCvBackendIsExplicitlySelected) {
  auto backend = CreateBackend(BackendType::kOpenCV, 0);
  EXPECT_TRUE(backend.ok());
  EXPECT_EQ(backend.value()->device().type, DeviceType::kCPU);
  EXPECT_TRUE(
      backend.value()->SupportsKernelLevel(CpuKernelLevel::kNaive));
  EXPECT_FALSE(
      backend.value()->SupportsKernelLevel(CpuKernelLevel::kOpenMP));
}
#endif

TEST(hpcplatform, OpContextDefaultsToNaiveCpu) {
  auto op_context = OpContext::Null();
  EXPECT_TRUE(op_context.context() == nullptr);
  EXPECT_TRUE(op_context.stream() == nullptr);
  EXPECT_EQ(op_context.cpu_level(), CpuKernelLevel::kNaive);
  EXPECT_TRUE(IsSimdKernelLevel(CpuKernelLevel::kSimdGeneric));
  EXPECT_FALSE(IsSimdKernelLevel(CpuKernelLevel::kOpenMP));
}

TEST(hpcplatform, CpuStreamIsSynchronous) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  EXPECT_TRUE(context->stream().synchronous());
  hpcplatform_EXPECT_OK(context->Synchronize());
}

}  // namespace

