// TR-1.1: every name<->enum mapper round-trips exactly, and unknown names
// return a non-OK status. Compiled in every build (the mappers are inline in
// public headers), so this guards them in both default and YAML builds.
#include <string>

#include "hpcplatform/core/common.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"
#include "hpcplatform/ops/wavelet.h"
#include "test_utils.h"

namespace
{
    using namespace hpcplatform;

    TEST(hpcplatform, BackendTypeNamesRoundTrip)
    {
        for (BackendType type :
             {BackendType::kNativeCpu, BackendType::kOpenCV, BackendType::kCUDA, BackendType::kNPU})
        {
            const std::string name = BackendTypeName(type);
            auto parsed = BackendTypeFromName(name);
            EXPECT_TRUE(parsed.ok());
            if (parsed.ok())
            {
                EXPECT_TRUE(parsed.value() == type);
            }
        }
        EXPECT_FALSE(BackendTypeFromName("nope").ok());
    }

    TEST(hpcplatform, CpuKernelLevelNamesRoundTrip)
    {
        for (CpuKernelLevel level :
             {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
              CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kSimdNeon, CpuKernelLevel::kTbb})
        {
            const std::string name = CpuKernelLevelName(level);
            auto parsed = CpuKernelLevelFromName(name);
            EXPECT_TRUE(parsed.ok());
            if (parsed.ok())
            {
                EXPECT_TRUE(parsed.value() == level);
            }
        }
        EXPECT_FALSE(CpuKernelLevelFromName("nope").ok());
    }

    TEST(hpcplatform, OpKindNamesRoundTrip)
    {
        for (OpKind kind : {OpKind::kResize, OpKind::kGaussianBlur, OpKind::kLaplacianPyramid,
                            OpKind::kFourierTransform, OpKind::kWaveletTransform})
        {
            const std::string name = OpKindName(kind);
            auto parsed = OpKindFromName(name);
            EXPECT_TRUE(parsed.ok());
            if (parsed.ok())
            {
                EXPECT_TRUE(parsed.value() == kind);
            }
        }
        EXPECT_FALSE(OpKindFromName("nope").ok());
    }

    TEST(hpcplatform, InterpolationAndLayoutNamesRoundTrip)
    {
        for (InterpolationMode mode : {InterpolationMode::kNearest, InterpolationMode::kBilinear})
        {
            const std::string name = InterpolationModeName(mode);
            auto parsed = InterpolationModeFromName(name);
            EXPECT_TRUE(parsed.ok());
            if (parsed.ok())
            {
                EXPECT_TRUE(parsed.value() == mode);
            }
        }
        EXPECT_FALSE(InterpolationModeFromName("nope").ok());

        for (ImageLayout layout : {ImageLayout::kNHWC, ImageLayout::kNCHW})
        {
            const std::string name = ImageLayoutName(layout);
            auto parsed = ImageLayoutFromName(name);
            EXPECT_TRUE(parsed.ok());
            if (parsed.ok())
            {
                EXPECT_TRUE(parsed.value() == layout);
            }
        }
        EXPECT_FALSE(ImageLayoutFromName("nope").ok());
    }

    TEST(hpcplatform, WaveletKindNamesRoundTrip)
    {
        for (WaveletKind kind : {WaveletKind::kHaar, WaveletKind::kDaubechies4})
        {
            const std::string name = WaveletKindName(kind);
            auto parsed = WaveletKindFromName(name);
            EXPECT_TRUE(parsed.ok());
            if (parsed.ok())
            {
                EXPECT_TRUE(parsed.value() == kind);
            }
        }
        EXPECT_FALSE(WaveletKindFromName("nope").ok());
    }

}  // namespace
