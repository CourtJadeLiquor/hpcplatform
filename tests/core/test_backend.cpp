#include "test_framework.h"

#include "backends/service.h"
#include "mycv/core/context.h"
#include "mycv/ops/op.h"

namespace {

using namespace mycv;

MYCV_TEST(CpuBackendIsExplicitlySelected) {
  auto backend = CreateBackend(BackendType::kNativeCpu, 0);
  MYCV_EXPECT_TRUE(backend.ok());
  MYCV_EXPECT_EQ(backend.value()->type(), BackendType::kNativeCpu);
  MYCV_EXPECT_EQ(backend.value()->device().type, DeviceType::kCPU);
  MYCV_EXPECT_TRUE(backend.value()->SupportsKernelLevel(CpuKernelLevel::kNaive));
  if (backend.value()->SupportsKernelLevel(CpuKernelLevel::kSimdAvx2)) {
    MYCV_EXPECT_TRUE(backend.value()->Describe().find("simd_avx2") !=
                     std::string::npos);
  }
  MYCV_EXPECT_TRUE(backend.value()->Describe().find("naive") !=
                   std::string::npos);
}

MYCV_TEST(UnavailableBackendsFailWithoutFallback) {
  const BackendType unavailable[] = {
#ifndef MYCV_WITH_OPENCV
      BackendType::kOpenCV,
#endif
#ifndef MYCV_WITH_CUDA
      BackendType::kCUDA,
#endif
      BackendType::kNPU};
  for (BackendType type : unavailable) {
    auto backend = CreateBackend(type);
    MYCV_EXPECT_FALSE(backend.ok());
    MYCV_EXPECT_EQ(backend.status().code(), StatusCode::kUnsupported);
    MYCV_EXPECT_TRUE(Context::Create(type).ok() == false);
  }
}

#ifdef MYCV_WITH_OPENCV
MYCV_TEST(OpenCvBackendIsExplicitlySelected) {
  auto backend = CreateBackend(BackendType::kOpenCV, 0);
  MYCV_EXPECT_TRUE(backend.ok());
  MYCV_EXPECT_EQ(backend.value()->device().type, DeviceType::kCPU);
  MYCV_EXPECT_TRUE(backend.value()->SupportsKernelLevel(CpuKernelLevel::kNaive));
  MYCV_EXPECT_FALSE(backend.value()->SupportsKernelLevel(CpuKernelLevel::kOpenMP));
}
#endif

MYCV_TEST(OpContextDefaultsToNaiveCpu) {
  const OpContext op_context;
  MYCV_EXPECT_TRUE(op_context.context == nullptr);
  MYCV_EXPECT_TRUE(op_context.stream == nullptr);
  MYCV_EXPECT_EQ(op_context.cpu_level, CpuKernelLevel::kNaive);
  MYCV_EXPECT_TRUE(IsSimdKernelLevel(CpuKernelLevel::kSimdGeneric));
  MYCV_EXPECT_FALSE(IsSimdKernelLevel(CpuKernelLevel::kOpenMP));
}

MYCV_TEST(CpuStreamIsSynchronous) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  MYCV_EXPECT_TRUE(context->stream().synchronous());
  MYCV_EXPECT_OK(context->Synchronize());
}

}  // namespace

MYCV_TEST_MAIN()
