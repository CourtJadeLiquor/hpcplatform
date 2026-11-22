// TR-1.1: every name<->enum mapper round-trips exactly, and unknown names
// return a non-OK status. Compiled in every build (the mappers are inline in
// public headers), so this guards them in both default and YAML builds.
#include "test_framework.h"

#include <string>

#include "mycv/core/device.h"
#include "mycv/core/kernel_level.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"
#include "mycv/ops/wavelet.h"

namespace {
using namespace mycv;

MYCV_TEST(BackendTypeNamesRoundTrip) {
  for (BackendType type : {BackendType::kNativeCpu, BackendType::kOpenCV,
                           BackendType::kCUDA, BackendType::kNPU}) {
    const std::string name = BackendTypeName(type);
    auto parsed = BackendTypeFromName(name);
    MYCV_EXPECT_TRUE(parsed.ok());
    if (parsed.ok()) {
      MYCV_EXPECT_TRUE(parsed.value() == type);
    }
  }
  MYCV_EXPECT_FALSE(BackendTypeFromName("nope").ok());
}

MYCV_TEST(CpuKernelLevelNamesRoundTrip) {
  for (CpuKernelLevel level :
       {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
        CpuKernelLevel::kSimdGeneric, CpuKernelLevel::kSimdAvx2,
        CpuKernelLevel::kSimdNeon, CpuKernelLevel::kTbb}) {
    const std::string name = CpuKernelLevelName(level);
    auto parsed = CpuKernelLevelFromName(name);
    MYCV_EXPECT_TRUE(parsed.ok());
    if (parsed.ok()) {
      MYCV_EXPECT_TRUE(parsed.value() == level);
    }
  }
  MYCV_EXPECT_FALSE(CpuKernelLevelFromName("nope").ok());
}

MYCV_TEST(OpKindNamesRoundTrip) {
  for (OpKind kind : {OpKind::kResize, OpKind::kGaussianBlur,
                      OpKind::kLaplacianPyramid, OpKind::kFourierTransform,
                      OpKind::kWaveletTransform}) {
    const std::string name = OpKindName(kind);
    auto parsed = OpKindFromName(name);
    MYCV_EXPECT_TRUE(parsed.ok());
    if (parsed.ok()) {
      MYCV_EXPECT_TRUE(parsed.value() == kind);
    }
  }
  MYCV_EXPECT_FALSE(OpKindFromName("nope").ok());
}

MYCV_TEST(InterpolationAndLayoutNamesRoundTrip) {
  for (InterpolationMode mode :
       {InterpolationMode::kNearest, InterpolationMode::kBilinear}) {
    const std::string name = InterpolationModeName(mode);
    auto parsed = InterpolationModeFromName(name);
    MYCV_EXPECT_TRUE(parsed.ok());
    if (parsed.ok()) {
      MYCV_EXPECT_TRUE(parsed.value() == mode);
    }
  }
  MYCV_EXPECT_FALSE(InterpolationModeFromName("nope").ok());

  for (ImageLayout layout : {ImageLayout::kNHWC, ImageLayout::kNCHW}) {
    const std::string name = ImageLayoutName(layout);
    auto parsed = ImageLayoutFromName(name);
    MYCV_EXPECT_TRUE(parsed.ok());
    if (parsed.ok()) {
      MYCV_EXPECT_TRUE(parsed.value() == layout);
    }
  }
  MYCV_EXPECT_FALSE(ImageLayoutFromName("nope").ok());
}

MYCV_TEST(WaveletKindNamesRoundTrip) {
  for (WaveletKind kind : {WaveletKind::kHaar, WaveletKind::kDaubechies4}) {
    const std::string name = WaveletKindName(kind);
    auto parsed = WaveletKindFromName(name);
    MYCV_EXPECT_TRUE(parsed.ok());
    if (parsed.ok()) {
      MYCV_EXPECT_TRUE(parsed.value() == kind);
    }
  }
  MYCV_EXPECT_FALSE(WaveletKindFromName("nope").ok());
}

}  // namespace

MYCV_TEST_MAIN()
