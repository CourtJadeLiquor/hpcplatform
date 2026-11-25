// AC-1..AC-5 on the CPU path: config-driven operator objects load, run, and
// fail-fast correctly. Only compiled with hpcplatform_WITH_YAML (see CMakeLists.txt);
// AC-6 (stub behavior) is covered by tests/core/test_operator_stub.cpp in the
// default build. CUDA-dependent cases branch on a runtime availability probe
// (CudaBackendAvailable), so the same assertions hold in CUDA and non-CUDA trees.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "hpcplatform/hpcplatform.h"
#include "test_utils.h"

namespace
{
    using namespace hpcplatform;

    // Bit-exact float comparison (raw bytes, so -0 vs +0 also distinguishes).
    bool FloatTensorsBitExact(const Tensor& a, const Tensor& b)
    {
        if (!(a.shape() == b.shape())) return false;
        if (a.dtype() != DataType::kFloat32 || b.dtype() != DataType::kFloat32)
        {
            return false;
        }
        const int64_t n = a.Numel();
        if (n != b.Numel()) return false;
        return std::memcmp(a.Data<float>(), b.Data<float>(),
                           static_cast<std::size_t>(n) * sizeof(float)) == 0;
    }

    Tensor MakeCpuInput(const std::shared_ptr<Context>& context)
    {
        auto input = Tensor::Empty(*context, Shape{1, 8, 8, 3}, DataType::kFloat32).value();
        for (int64_t i = 0; i < input.Numel(); ++i)
        {
            input.Data<float>()[i] = static_cast<float>(i % 17) * 0.5f;
        }
        return input;
    }

    OpContext CpuContext(const std::shared_ptr<Context>& context,
                         CpuKernelLevel level = CpuKernelLevel::kNaive)
    {
        return OpContext(*context.get()).set_stream(nullptr).set_cpu_level(level);
    }

    // Runtime probe: whether the CUDA backend can actually be created in this
    // build on this machine (CUDA compiled in AND a usable device present).
    // Tests that use a "cuda" config backend branch on this instead of assuming
    // the backend is unavailable, so they pass in both CUDA and non-CUDA trees.
    bool CudaBackendAvailable()
    {
        return Context::Create(BackendType::kCUDA).ok();
    }

    constexpr const char* kResizeAndBlurConfig = R"(
operators:
  - name: resize_16
    op: resize
    backend: native_cpu
    params:
      output_height: 16
      output_width: 16
      interpolation: bilinear
      layout: nhwc
  - name: blur_3
    op: gaussian_blur
    backend: native_cpu
    params:
      kernel_size: 3
      sigma: 1.0
      layout: nhwc
)";

    // AC-1: objects load, Run is bit-identical to the direct free-function call,
    // and metadata is correct.
    TEST(hpcplatform, LoadAndRunMatchesDirectCalls)
    {
        auto loaded = LoadOperatorsFromString(kResizeAndBlurConfig);
        EXPECT_TRUE(loaded.ok());
        if (!loaded.ok()) return;
        const OperatorMap& ops = loaded.value();

        auto* resize_op = GetOperator<ResizeOperator>(ops, "resize_16");
        auto* blur_op = GetOperator<GaussianBlurOperator>(ops, "blur_3");
        EXPECT_TRUE(resize_op != nullptr);
        EXPECT_TRUE(blur_op != nullptr);
        if (resize_op == nullptr || blur_op == nullptr) return;

        auto context = Context::Create(BackendType::kNativeCpu).value();
        const Tensor input = MakeCpuInput(context);

        auto direct_resize = Resize(CpuContext(context), input, 16, 16,
                                    InterpolationMode::kBilinear, ImageLayout::kNHWC)
                                 .value();
        auto op_resize = resize_op->Run(input);
        EXPECT_TRUE(op_resize.ok());
        EXPECT_TRUE(FloatTensorsBitExact(op_resize.value(), direct_resize));

        auto direct_blur =
            GaussianBlur(CpuContext(context), input, 3, 1.0, ImageLayout::kNHWC).value();
        auto op_blur = blur_op->Run(input);
        EXPECT_TRUE(op_blur.ok());
        EXPECT_TRUE(FloatTensorsBitExact(op_blur.value(), direct_blur));

        EXPECT_EQ(resize_op->name(), std::string("resize_16"));
        EXPECT_TRUE(resize_op->kind() == OpKind::kResize);
        EXPECT_TRUE(resize_op->backend() == BackendType::kNativeCpu);
        EXPECT_TRUE(resize_op->cpu_level() == CpuKernelLevel::kNaive);
        EXPECT_TRUE(resize_op->context() != nullptr);

        EXPECT_TRUE(blur_op->kind() == OpKind::kGaussianBlur);
    }

    // AC-2: two objects of the same op with different params produce independent,
    // correctly sized outputs.
    TEST(hpcplatform, MultipleObjectsDifferentParams)
    {
        const char* yaml = R"(
operators:
  - name: r16
    op: resize
    backend: native_cpu
    params: {output_height: 16, output_width: 16, interpolation: bilinear, layout: nhwc}
  - name: r32
    op: resize
    backend: native_cpu
    params: {output_height: 32, output_width: 32, interpolation: bilinear, layout: nhwc}
)";
        auto loaded = LoadOperatorsFromString(yaml);
        EXPECT_TRUE(loaded.ok());
        if (!loaded.ok()) return;

        auto* r16 = GetOperator<ResizeOperator>(loaded.value(), "r16");
        auto* r32 = GetOperator<ResizeOperator>(loaded.value(), "r32");
        EXPECT_TRUE(r16 != nullptr);
        EXPECT_TRUE(r32 != nullptr);
        if (r16 == nullptr || r32 == nullptr) return;

        auto context = Context::Create(BackendType::kNativeCpu).value();
        const Tensor input = MakeCpuInput(context);

        auto o16 = r16->Run(input);
        auto o32 = r32->Run(input);
        EXPECT_TRUE(o16.ok());
        EXPECT_TRUE(o32.ok());
        if (o16.ok() && o32.ok())
        {
            // Extra parens hide the braced-init commas from the preprocessor.
            EXPECT_TRUE((o16.value().shape() == Shape{1, 16, 16, 3}));
            EXPECT_TRUE((o32.value().shape() == Shape{1, 32, 32, 3}));
        }
    }

    // AC-3 (a): per-entry config backend binds each object.
    TEST(hpcplatform, PerEntryBackendBindsObject)
    {
        auto loaded = LoadOperatorsFromString(kResizeAndBlurConfig);
        EXPECT_TRUE(loaded.ok());
        if (!loaded.ok()) return;
        for (const auto& [name, op] : loaded.value())
        {
            (void)name;
            EXPECT_TRUE(op->backend() == BackendType::kNativeCpu);
            EXPECT_TRUE(op->context() != nullptr);
        }
    }

    // AC-3 (b): a shared context overrides per-entry backend, even when the config
    // requests a backend unavailable in this build (cuda here). The shared null
    // case returns kInvalidArgument.
    TEST(hpcplatform, SharedContextOverridesConfigBackend)
    {
        const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: cuda
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
        const std::string path = "/tmp/hpcplatform_operator_shared_test.yaml";
        {
            std::ofstream out(path);
            out << yaml;
        }

        auto shared = Context::Create(BackendType::kNativeCpu).value();
        auto loaded = LoadOperators(path, shared);
        EXPECT_TRUE(loaded.ok());
        if (loaded.ok())
        {
            auto* r = GetOperator<ResizeOperator>(loaded.value(), "r");
            EXPECT_TRUE(r != nullptr);
            if (r != nullptr)
            {
                EXPECT_TRUE(r->backend() == BackendType::kNativeCpu);
                EXPECT_TRUE(r->context().get() == shared.get());
            }
        }

        auto null_shared = LoadOperators(path, nullptr);
        std::remove(path.c_str());
        EXPECT_FALSE(null_shared.ok());
        if (!null_shared.ok())
        {
            EXPECT_TRUE(null_shared.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // AC-3 (c): the hpcplatform_BACKEND env var overrides the config backend.
    // Without the env var, the config requests "cuda": this fails fast when CUDA
    // is unavailable in this build or on this machine (no fallback), and binds
    // objects to CUDA when it is available.
    TEST(hpcplatform, EnvVarOverridesBackend)
    {
        const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: cuda
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
        auto without_env = LoadOperatorsFromString(yaml);
        if (CudaBackendAvailable())
        {
            EXPECT_TRUE(without_env.ok());
            if (without_env.ok())
            {
                auto* r = GetOperator<ResizeOperator>(without_env.value(), "r");
                EXPECT_TRUE(r != nullptr);
                if (r != nullptr)
                {
                    EXPECT_TRUE(r->backend() == BackendType::kCUDA);
                }
            }
        }
        else
        {
            EXPECT_FALSE(without_env.ok());
        }

        ::setenv("hpcplatform_BACKEND", "native_cpu", 1);
        auto with_env = LoadOperatorsFromString(yaml);
        ::unsetenv("hpcplatform_BACKEND");
        EXPECT_TRUE(with_env.ok());
        if (with_env.ok())
        {
            auto* r = GetOperator<ResizeOperator>(with_env.value(), "r");
            EXPECT_TRUE(r != nullptr);
            if (r != nullptr)
            {
                EXPECT_TRUE(r->backend() == BackendType::kNativeCpu);
            }
        }
    }

    TEST(hpcplatform, EnvVarCpuLevelApplied)
    {
        const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: native_cpu
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
        ::setenv("hpcplatform_BACKEND", "native_cpu", 1);
        ::setenv("hpcplatform_CPU_LEVEL", "openmp", 1);
        auto loaded = LoadOperatorsFromString(yaml);
        ::unsetenv("hpcplatform_BACKEND");
        ::unsetenv("hpcplatform_CPU_LEVEL");
        EXPECT_TRUE(loaded.ok());
        if (loaded.ok())
        {
            auto* r = GetOperator<ResizeOperator>(loaded.value(), "r");
            EXPECT_TRUE(r != nullptr);
            if (r != nullptr)
            {
                EXPECT_TRUE(r->cpu_level() == CpuKernelLevel::kOpenMP);
            }
        }
    }

    // AC-5 / NFR-6: an env-requested backend that is unavailable fails the whole
    // load rather than silently falling back to the config backend. The load is
    // expected to succeed when the requested CUDA backend is actually available.
    TEST(hpcplatform, EnvVarBackendAvailabilityHonored)
    {
        const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: native_cpu
    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}
)";
        ::setenv("hpcplatform_BACKEND", "cuda", 1);
        auto loaded = LoadOperatorsFromString(yaml);
        ::unsetenv("hpcplatform_BACKEND");
        if (CudaBackendAvailable())
        {
            EXPECT_TRUE(loaded.ok());
            if (loaded.ok())
            {
                auto* r = GetOperator<ResizeOperator>(loaded.value(), "r");
                EXPECT_TRUE(r != nullptr);
                if (r != nullptr)
                {
                    EXPECT_TRUE(r->backend() == BackendType::kCUDA);
                }
            }
        }
        else
        {
            EXPECT_FALSE(loaded.ok());
        }
    }

    TEST(hpcplatform, EnvVarInvalidBackendNameFails)
    {
        ::setenv("hpcplatform_BACKEND", "bogus", 1);
        auto loaded = LoadOperatorsFromString("operators: []");
        ::unsetenv("hpcplatform_BACKEND");
        EXPECT_FALSE(loaded.ok());
    }

    // AC-4 / AC-5 fail-fast error paths: each must return an error with no objects.
    TEST(hpcplatform, UnknownOpFails)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: nope\n    backend: native_cpu\n    params: {}\n");
        EXPECT_FALSE(loaded.ok());
    }

    TEST(hpcplatform, MissingRequiredParamFails)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n"
            "    params: {output_width: 4, interpolation: bilinear, layout: nhwc}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    TEST(hpcplatform, BadParamTypeFails)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n"
            "    params: {output_height: abc, output_width: 4, interpolation: bilinear, layout: "
            "nhwc}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    TEST(hpcplatform, DuplicateNameFails)
    {
        const char* yaml = R"(
operators:
  - name: dup
    op: resize
    backend: native_cpu
    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}
  - name: dup
    op: resize
    backend: native_cpu
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
        auto loaded = LoadOperatorsFromString(yaml);
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    TEST(hpcplatform, YamlSyntaxErrorFails)
    {
        auto loaded = LoadOperatorsFromString("operators: [unterminated");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // Per-entry config backend: "cuda" fails fast when unavailable (no fallback)
    // and binds objects to CUDA when available.
    TEST(hpcplatform, ConfigBackendAvailabilityHonored)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: cuda\n"
            "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: "
            "nhwc}\n");
        if (CudaBackendAvailable())
        {
            EXPECT_TRUE(loaded.ok());
            if (loaded.ok())
            {
                auto* r = GetOperator<ResizeOperator>(loaded.value(), "x");
                EXPECT_TRUE(r != nullptr);
                if (r != nullptr)
                {
                    EXPECT_TRUE(r->backend() == BackendType::kCUDA);
                }
            }
        }
        else
        {
            EXPECT_FALSE(loaded.ok());
        }
    }

    // AC-4: any failure returns an empty map — no partially constructed objects
    // are ever handed back to the caller.
    TEST(hpcplatform, FailureYieldsEmptyMap)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: nope\n    backend: native_cpu\n    params: {}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.value().empty());
        }
    }

    // AC-5: invalid per-entry environment names are reported as kInvalidArgument.
    TEST(hpcplatform, PerEntryInvalidBackendNameFails)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: bogus\n"
            "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: "
            "nhwc}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
            EXPECT_TRUE(loaded.value().empty());
        }
    }

    TEST(hpcplatform, PerEntryInvalidCpuLevelFails)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n    cpu_level: "
            "bogus\n"
            "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: "
            "nhwc}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    TEST(hpcplatform, PerEntryInvalidDeviceIndexFails)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n    device_index: "
            "-1\n"
            "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: "
            "nhwc}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // Hardening: untrusted config text is bounded before it reaches yaml-cpp.
    // Oversized input must fail as kInvalidArgument, not stress the parser.
    TEST(hpcplatform, OversizedConfigFails)
    {
        const std::string huge(1u << 20, ' ');  // 1 MiB of spaces
        auto loaded = LoadOperatorsFromString(huge);
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // Anchors and aliases are rejected: yaml-cpp expands aliases by reference, so
    // a cyclic alias could recurse without bound inside the parser.
    TEST(hpcplatform, AnchorAndAliasConfigFails)
    {
        auto anchored = LoadOperatorsFromString(
            "operators:\n  - &a\n    name: x\n    op: resize\n    backend: native_cpu\n"
            "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: "
            "nhwc}\n");
        EXPECT_FALSE(anchored.ok());
        EXPECT_TRUE(anchored.status().code() == StatusCode::kInvalidArgument);

        auto aliased = LoadOperatorsFromString(
            "defaults: &d {output_height: 2, output_width: 2}\n"
            "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n"
            "    params: *d\n");
        EXPECT_FALSE(aliased.ok());
        EXPECT_TRUE(aliased.status().code() == StatusCode::kInvalidArgument);
    }

    // Flow collections nested deeper than the hardening limit must fail cleanly
    // instead of exhausting the parser's stack.
    TEST(hpcplatform, DeepFlowNestingFails)
    {
        auto loaded = LoadOperatorsFromString("operators: " + std::string(100, '['));
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // Structural caps: more than 1024 entries or more than 256 params on one
    // entry fail as kInvalidArgument.
    TEST(hpcplatform, TooManyOperatorEntriesFails)
    {
        std::string yaml = "operators:\n";
        for (int i = 0; i < 1025; ++i)
        {
            yaml += "  - name: r" + std::to_string(i) +
                    "\n    op: resize\n    backend: native_cpu\n"
                    "    params: {output_height: 2, output_width: 2, interpolation: bilinear, "
                    "layout: nhwc}\n";
        }
        auto loaded = LoadOperatorsFromString(yaml);
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    TEST(hpcplatform, TooManyParamsFails)
    {
        std::string params;
        for (int i = 0; i < 257; ++i)
        {
            if (i > 0) params += ", ";
            params += "k" + std::to_string(i) + ": v" + std::to_string(i);
        }
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n    params: {" +
            params + "}\n");
        EXPECT_FALSE(loaded.ok());
        if (!loaded.ok())
        {
            EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // Quoted scalars may legitimately contain '*', '[', and '{' characters; the
    // pre-parse scan must not reject valid configs.
    TEST(hpcplatform, QuotedSpecialCharactersStillParse)
    {
        auto loaded = LoadOperatorsFromString(
            "operators:\n  - name: \"r*1\"\n    op: resize\n    backend: native_cpu\n"
            "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: "
            "nhwc}\n");
        EXPECT_TRUE(loaded.ok());
        if (loaded.ok())
        {
            EXPECT_TRUE(GetOperator<ResizeOperator>(loaded.value(), "r*1") != nullptr);
        }
    }

    // AC-1 for the transform operators: they load, Run is bit-identical to the
    // direct free-function call, and metadata is correct.
    TEST(hpcplatform, LoadsTransformOperators)
    {
        const char* yaml = R"(
operators:
  - name: pyramid
    op: laplacian_pyramid
    backend: native_cpu
    params: {levels: 2, kernel_size: 3, sigma: 1.0, layout: nhwc}
  - name: spectrum
    op: fourier_transform
    backend: native_cpu
    params: {layout: nhwc}
  - name: bands
    op: wavelet_transform
    backend: native_cpu
    params: {kind: haar, layout: nhwc}
)";
        auto loaded = LoadOperatorsFromString(yaml);
        EXPECT_TRUE(loaded.ok());
        if (!loaded.ok()) return;
        const OperatorMap& ops = loaded.value();

        auto* pyramid = GetOperator<LaplacianPyramidOperator>(ops, "pyramid");
        auto* spectrum = GetOperator<FourierTransformOperator>(ops, "spectrum");
        auto* bands = GetOperator<WaveletTransformOperator>(ops, "bands");
        EXPECT_TRUE(pyramid != nullptr);
        EXPECT_TRUE(spectrum != nullptr);
        EXPECT_TRUE(bands != nullptr);
        if (pyramid == nullptr || spectrum == nullptr || bands == nullptr) return;

        auto context = Context::Create(BackendType::kNativeCpu).value();
        const Tensor input = MakeCpuInput(context);

        auto op_pyramid = pyramid->Run(input);
        auto direct_pyramid =
            BuildLaplacianPyramid(CpuContext(context), input, 2, 3, 1.0, ImageLayout::kNHWC);
        EXPECT_TRUE(op_pyramid.ok());
        EXPECT_TRUE(direct_pyramid.ok());
        if (op_pyramid.ok() && direct_pyramid.ok())
        {
            const auto& op_levels = op_pyramid.value().levels;
            const auto& direct_levels = direct_pyramid.value().levels;
            EXPECT_EQ(op_levels.size(), direct_levels.size());
            if (op_levels.size() == direct_levels.size())
            {
                for (std::size_t i = 0; i < op_levels.size(); ++i)
                {
                    EXPECT_TRUE(FloatTensorsBitExact(op_levels[i], direct_levels[i]));
                }
            }
        }

        auto op_spectrum = spectrum->Run(input);
        auto direct_spectrum = FourierTransform(CpuContext(context), input, ImageLayout::kNHWC);
        EXPECT_TRUE(op_spectrum.ok());
        EXPECT_TRUE(direct_spectrum.ok());
        if (op_spectrum.ok() && direct_spectrum.ok())
        {
            EXPECT_TRUE(
                FloatTensorsBitExact(op_spectrum.value().real, direct_spectrum.value().real));
            EXPECT_TRUE(
                FloatTensorsBitExact(op_spectrum.value().imag, direct_spectrum.value().imag));
        }

        auto op_bands = bands->Run(input);
        auto direct_bands =
            WaveletTransform(CpuContext(context), input, WaveletKind::kHaar, ImageLayout::kNHWC);
        EXPECT_TRUE(op_bands.ok());
        EXPECT_TRUE(direct_bands.ok());
        if (op_bands.ok() && direct_bands.ok())
        {
            EXPECT_TRUE(FloatTensorsBitExact(op_bands.value().ll, direct_bands.value().ll));
            EXPECT_TRUE(FloatTensorsBitExact(op_bands.value().lh, direct_bands.value().lh));
            EXPECT_TRUE(FloatTensorsBitExact(op_bands.value().hl, direct_bands.value().hl));
            EXPECT_TRUE(FloatTensorsBitExact(op_bands.value().hh, direct_bands.value().hh));
        }

        EXPECT_TRUE(pyramid->kind() == OpKind::kLaplacianPyramid);
        EXPECT_TRUE(spectrum->kind() == OpKind::kFourierTransform);
        EXPECT_TRUE(bands->kind() == OpKind::kWaveletTransform);
    }

    // AC-5: each transform descriptor rejects its own invalid parameters.
    TEST(hpcplatform, TransformOperatorParamErrorsFail)
    {
        auto bad_levels = LoadOperatorsFromString(
            "operators:\n  - name: p\n    op: laplacian_pyramid\n"
            "    backend: native_cpu\n"
            "    params: {levels: 0, kernel_size: 3, sigma: 1.0, layout: nhwc}\n");
        EXPECT_FALSE(bad_levels.ok());
        if (!bad_levels.ok())
        {
            EXPECT_TRUE(bad_levels.status().code() == StatusCode::kInvalidArgument);
        }

        auto bad_kernel = LoadOperatorsFromString(
            "operators:\n  - name: p\n    op: laplacian_pyramid\n"
            "    backend: native_cpu\n"
            "    params: {levels: 2, kernel_size: 4, sigma: 1.0, layout: nhwc}\n");
        EXPECT_FALSE(bad_kernel.ok());
        if (!bad_kernel.ok())
        {
            EXPECT_TRUE(bad_kernel.status().code() == StatusCode::kInvalidArgument);
        }

        auto missing_layout = LoadOperatorsFromString(
            "operators:\n  - name: f\n    op: fourier_transform\n"
            "    backend: native_cpu\n    params: {}\n");
        EXPECT_FALSE(missing_layout.ok());
        if (!missing_layout.ok())
        {
            EXPECT_TRUE(missing_layout.status().code() == StatusCode::kInvalidArgument);
        }

        auto bad_kind = LoadOperatorsFromString(
            "operators:\n  - name: w\n    op: wavelet_transform\n"
            "    backend: native_cpu\n    params: {kind: cdf97, layout: nhwc}\n");
        EXPECT_FALSE(bad_kind.ok());
        if (!bad_kind.ok())
        {
            EXPECT_TRUE(bad_kind.status().code() == StatusCode::kInvalidArgument);
        }
    }

    // GetOperator returns nullptr for a missing name or a wrong concrete type, and
    // the base Operator metadata is still reachable.
    TEST(hpcplatform, GetOperatorTypeChecks)
    {
        const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: native_cpu
    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}
)";
        auto loaded = LoadOperatorsFromString(yaml);
        EXPECT_TRUE(loaded.ok());
        if (!loaded.ok()) return;
        const OperatorMap& ops = loaded.value();
        EXPECT_TRUE(GetOperator<ResizeOperator>(ops, "r") != nullptr);
        EXPECT_TRUE(GetOperator<GaussianBlurOperator>(ops, "r") == nullptr);
        EXPECT_TRUE(GetOperator<ResizeOperator>(ops, "missing") == nullptr);
        const Operator* base = ops.at("r").get();
        EXPECT_TRUE(base->kind() == OpKind::kResize);
        EXPECT_TRUE(base->backend() == BackendType::kNativeCpu);
    }

}  // namespace
