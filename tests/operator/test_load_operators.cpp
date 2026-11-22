// AC-1..AC-5 on the CPU path: config-driven operator objects load, run, and
// fail-fast correctly. Only compiled with MYCV_WITH_YAML (see CMakeLists.txt);
// AC-6 (stub behavior) is covered by tests/core/test_operator_stub.cpp in the
// default build. CUDA portions of AC-3 are verified manually in a CUDA tree.
#include "test_framework.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "mycv/mycv.h"

namespace {
using namespace mycv;

// Bit-exact float comparison (raw bytes, so -0 vs +0 also distinguishes).
bool FloatTensorsBitExact(const Tensor& a, const Tensor& b) {
  if (!(a.shape() == b.shape())) return false;
  if (a.dtype() != DataType::kFloat32 || b.dtype() != DataType::kFloat32) {
    return false;
  }
  const int64_t n = a.Numel();
  if (n != b.Numel()) return false;
  return std::memcmp(a.Data<float>(), b.Data<float>(),
                     static_cast<std::size_t>(n) * sizeof(float)) == 0;
}

Tensor MakeCpuInput(const std::shared_ptr<Context>& context) {
  auto input =
      context->AllocateTensor(Shape{1, 8, 8, 3}, DataType::kFloat32).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<float>()[i] = static_cast<float>(i % 17) * 0.5f;
  }
  return input;
}

OpContext CpuContext(const std::shared_ptr<Context>& context,
                     CpuKernelLevel level = CpuKernelLevel::kNaive) {
  return OpContext{context.get(), /*stream=*/nullptr, level};
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
MYCV_TEST(LoadAndRunMatchesDirectCalls) {
  auto loaded = LoadOperatorsFromString(kResizeAndBlurConfig);
  MYCV_EXPECT_TRUE(loaded.ok());
  if (!loaded.ok()) return;
  const OperatorMap& ops = loaded.value();

  auto* resize_op = GetOperator<ResizeOperator>(ops, "resize_16");
  auto* blur_op = GetOperator<GaussianBlurOperator>(ops, "blur_3");
  MYCV_EXPECT_TRUE(resize_op != nullptr);
  MYCV_EXPECT_TRUE(blur_op != nullptr);
  if (resize_op == nullptr || blur_op == nullptr) return;

  auto context = Context::Create(BackendType::kNativeCpu).value();
  const Tensor input = MakeCpuInput(context);

  auto direct_resize =
      Resize(CpuContext(context), input, 16, 16, InterpolationMode::kBilinear,
             ImageLayout::kNHWC)
          .value();
  auto op_resize = resize_op->Run(input);
  MYCV_EXPECT_TRUE(op_resize.ok());
  MYCV_EXPECT_TRUE(FloatTensorsBitExact(op_resize.value(), direct_resize));

  auto direct_blur =
      GaussianBlur(CpuContext(context), input, 3, 1.0, ImageLayout::kNHWC)
          .value();
  auto op_blur = blur_op->Run(input);
  MYCV_EXPECT_TRUE(op_blur.ok());
  MYCV_EXPECT_TRUE(FloatTensorsBitExact(op_blur.value(), direct_blur));

  MYCV_EXPECT_EQ(resize_op->name(), std::string("resize_16"));
  MYCV_EXPECT_TRUE(resize_op->kind() == OpKind::kResize);
  MYCV_EXPECT_TRUE(resize_op->backend() == BackendType::kNativeCpu);
  MYCV_EXPECT_TRUE(resize_op->cpu_level() == CpuKernelLevel::kNaive);
  MYCV_EXPECT_TRUE(resize_op->context() != nullptr);

  MYCV_EXPECT_TRUE(blur_op->kind() == OpKind::kGaussianBlur);
}

// AC-2: two objects of the same op with different params produce independent,
// correctly sized outputs.
MYCV_TEST(MultipleObjectsDifferentParams) {
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
  MYCV_EXPECT_TRUE(loaded.ok());
  if (!loaded.ok()) return;

  auto* r16 = GetOperator<ResizeOperator>(loaded.value(), "r16");
  auto* r32 = GetOperator<ResizeOperator>(loaded.value(), "r32");
  MYCV_EXPECT_TRUE(r16 != nullptr);
  MYCV_EXPECT_TRUE(r32 != nullptr);
  if (r16 == nullptr || r32 == nullptr) return;

  auto context = Context::Create(BackendType::kNativeCpu).value();
  const Tensor input = MakeCpuInput(context);

  auto o16 = r16->Run(input);
  auto o32 = r32->Run(input);
  MYCV_EXPECT_TRUE(o16.ok());
  MYCV_EXPECT_TRUE(o32.ok());
  if (o16.ok() && o32.ok()) {
    // Extra parens hide the braced-init commas from the preprocessor.
    MYCV_EXPECT_TRUE((o16.value().shape() == Shape{1, 16, 16, 3}));
    MYCV_EXPECT_TRUE((o32.value().shape() == Shape{1, 32, 32, 3}));
  }
}

// AC-3 (a): per-entry config backend binds each object.
MYCV_TEST(PerEntryBackendBindsObject) {
  auto loaded = LoadOperatorsFromString(kResizeAndBlurConfig);
  MYCV_EXPECT_TRUE(loaded.ok());
  if (!loaded.ok()) return;
  for (const auto& [name, op] : loaded.value()) {
    (void)name;
    MYCV_EXPECT_TRUE(op->backend() == BackendType::kNativeCpu);
    MYCV_EXPECT_TRUE(op->context() != nullptr);
  }
}

// AC-3 (b): a shared context overrides per-entry backend, even when the config
// requests a backend unavailable in this build (cuda here). The shared null
// case returns kInvalidArgument.
MYCV_TEST(SharedContextOverridesConfigBackend) {
  const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: cuda
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
  const std::string path = "/tmp/mycv_operator_shared_test.yaml";
  {
    std::ofstream out(path);
    out << yaml;
  }

  auto shared = Context::Create(BackendType::kNativeCpu).value();
  auto loaded = LoadOperators(path, shared);
  MYCV_EXPECT_TRUE(loaded.ok());
  if (loaded.ok()) {
    auto* r = GetOperator<ResizeOperator>(loaded.value(), "r");
    MYCV_EXPECT_TRUE(r != nullptr);
    if (r != nullptr) {
      MYCV_EXPECT_TRUE(r->backend() == BackendType::kNativeCpu);
      MYCV_EXPECT_TRUE(r->context().get() == shared.get());
    }
  }

  auto null_shared = LoadOperators(path, nullptr);
  std::remove(path.c_str());
  MYCV_EXPECT_FALSE(null_shared.ok());
  if (!null_shared.ok()) {
    MYCV_EXPECT_TRUE(null_shared.status().code() == StatusCode::kInvalidArgument);
  }
}

// AC-3 (c): the MYCV_BACKEND env var overrides the config backend. Without it,
// an unavailable config backend fails (no fallback).
MYCV_TEST(EnvVarOverridesBackend) {
  const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: cuda
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
  auto without_env = LoadOperatorsFromString(yaml);
  MYCV_EXPECT_FALSE(without_env.ok());

  ::setenv("MYCV_BACKEND", "native_cpu", 1);
  auto with_env = LoadOperatorsFromString(yaml);
  ::unsetenv("MYCV_BACKEND");
  MYCV_EXPECT_TRUE(with_env.ok());
  if (with_env.ok()) {
    auto* r = GetOperator<ResizeOperator>(with_env.value(), "r");
    MYCV_EXPECT_TRUE(r != nullptr);
    if (r != nullptr) {
      MYCV_EXPECT_TRUE(r->backend() == BackendType::kNativeCpu);
    }
  }
}

MYCV_TEST(EnvVarCpuLevelApplied) {
  const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: native_cpu
    params: {output_height: 4, output_width: 4, interpolation: bilinear, layout: nhwc}
)";
  ::setenv("MYCV_BACKEND", "native_cpu", 1);
  ::setenv("MYCV_CPU_LEVEL", "openmp", 1);
  auto loaded = LoadOperatorsFromString(yaml);
  ::unsetenv("MYCV_BACKEND");
  ::unsetenv("MYCV_CPU_LEVEL");
  MYCV_EXPECT_TRUE(loaded.ok());
  if (loaded.ok()) {
    auto* r = GetOperator<ResizeOperator>(loaded.value(), "r");
    MYCV_EXPECT_TRUE(r != nullptr);
    if (r != nullptr) {
      MYCV_EXPECT_TRUE(r->cpu_level() == CpuKernelLevel::kOpenMP);
    }
  }
}

// AC-5 / NFR-6: an env-requested backend that is unavailable fails the whole
// load rather than silently falling back to the config backend.
MYCV_TEST(EnvVarUnavailableBackendFails) {
  const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: native_cpu
    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}
)";
  ::setenv("MYCV_BACKEND", "cuda", 1);
  auto loaded = LoadOperatorsFromString(yaml);
  ::unsetenv("MYCV_BACKEND");
  MYCV_EXPECT_FALSE(loaded.ok());
}

MYCV_TEST(EnvVarInvalidBackendNameFails) {
  ::setenv("MYCV_BACKEND", "bogus", 1);
  auto loaded = LoadOperatorsFromString("operators: []");
  ::unsetenv("MYCV_BACKEND");
  MYCV_EXPECT_FALSE(loaded.ok());
}

// AC-4 / AC-5 fail-fast error paths: each must return an error with no objects.
MYCV_TEST(UnknownOpFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: nope\n    backend: native_cpu\n    params: {}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
}

MYCV_TEST(MissingRequiredParamFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n"
      "    params: {output_width: 4, interpolation: bilinear, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
  }
}

MYCV_TEST(BadParamTypeFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n"
      "    params: {output_height: abc, output_width: 4, interpolation: bilinear, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
  }
}

MYCV_TEST(DuplicateNameFails) {
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
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
  }
}

MYCV_TEST(YamlSyntaxErrorFails) {
  auto loaded = LoadOperatorsFromString("operators: [unterminated");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
  }
}

MYCV_TEST(UnavailableBackendFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: resize\n    backend: cuda\n"
      "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
}

// AC-4: any failure returns an empty map — no partially constructed objects
// are ever handed back to the caller.
MYCV_TEST(FailureYieldsEmptyMap) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: nope\n    backend: native_cpu\n    params: {}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.value().empty());
  }
}

// AC-5: invalid per-entry environment names are reported as kInvalidArgument.
MYCV_TEST(PerEntryInvalidBackendNameFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: resize\n    backend: bogus\n"
      "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
    MYCV_EXPECT_TRUE(loaded.value().empty());
  }
}

MYCV_TEST(PerEntryInvalidCpuLevelFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n    cpu_level: bogus\n"
      "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
  }
}

MYCV_TEST(PerEntryInvalidDeviceIndexFails) {
  auto loaded = LoadOperatorsFromString(
      "operators:\n  - name: x\n    op: resize\n    backend: native_cpu\n    device_index: -1\n"
      "    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(loaded.ok());
  if (!loaded.ok()) {
    MYCV_EXPECT_TRUE(loaded.status().code() == StatusCode::kInvalidArgument);
  }
}

// AC-1 for the transform operators: they load, Run is bit-identical to the
// direct free-function call, and metadata is correct.
MYCV_TEST(LoadsTransformOperators) {
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
  MYCV_EXPECT_TRUE(loaded.ok());
  if (!loaded.ok()) return;
  const OperatorMap& ops = loaded.value();

  auto* pyramid = GetOperator<LaplacianPyramidOperator>(ops, "pyramid");
  auto* spectrum = GetOperator<FourierTransformOperator>(ops, "spectrum");
  auto* bands = GetOperator<WaveletTransformOperator>(ops, "bands");
  MYCV_EXPECT_TRUE(pyramid != nullptr);
  MYCV_EXPECT_TRUE(spectrum != nullptr);
  MYCV_EXPECT_TRUE(bands != nullptr);
  if (pyramid == nullptr || spectrum == nullptr || bands == nullptr) return;

  auto context = Context::Create(BackendType::kNativeCpu).value();
  const Tensor input = MakeCpuInput(context);

  auto op_pyramid = pyramid->Run(input);
  auto direct_pyramid =
      BuildLaplacianPyramid(CpuContext(context), input, 2, 3, 1.0,
                            ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(op_pyramid.ok());
  MYCV_EXPECT_TRUE(direct_pyramid.ok());
  if (op_pyramid.ok() && direct_pyramid.ok()) {
    const auto& op_levels = op_pyramid.value().levels;
    const auto& direct_levels = direct_pyramid.value().levels;
    MYCV_EXPECT_EQ(op_levels.size(), direct_levels.size());
    if (op_levels.size() == direct_levels.size()) {
      for (std::size_t i = 0; i < op_levels.size(); ++i) {
        MYCV_EXPECT_TRUE(
            FloatTensorsBitExact(op_levels[i], direct_levels[i]));
      }
    }
  }

  auto op_spectrum = spectrum->Run(input);
  auto direct_spectrum =
      FourierTransform(CpuContext(context), input, ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(op_spectrum.ok());
  MYCV_EXPECT_TRUE(direct_spectrum.ok());
  if (op_spectrum.ok() && direct_spectrum.ok()) {
    MYCV_EXPECT_TRUE(FloatTensorsBitExact(op_spectrum.value().real,
                                          direct_spectrum.value().real));
    MYCV_EXPECT_TRUE(FloatTensorsBitExact(op_spectrum.value().imag,
                                          direct_spectrum.value().imag));
  }

  auto op_bands = bands->Run(input);
  auto direct_bands =
      WaveletTransform(CpuContext(context), input, WaveletKind::kHaar,
                       ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(op_bands.ok());
  MYCV_EXPECT_TRUE(direct_bands.ok());
  if (op_bands.ok() && direct_bands.ok()) {
    MYCV_EXPECT_TRUE(
        FloatTensorsBitExact(op_bands.value().ll, direct_bands.value().ll));
    MYCV_EXPECT_TRUE(
        FloatTensorsBitExact(op_bands.value().lh, direct_bands.value().lh));
    MYCV_EXPECT_TRUE(
        FloatTensorsBitExact(op_bands.value().hl, direct_bands.value().hl));
    MYCV_EXPECT_TRUE(
        FloatTensorsBitExact(op_bands.value().hh, direct_bands.value().hh));
  }

  MYCV_EXPECT_TRUE(pyramid->kind() == OpKind::kLaplacianPyramid);
  MYCV_EXPECT_TRUE(spectrum->kind() == OpKind::kFourierTransform);
  MYCV_EXPECT_TRUE(bands->kind() == OpKind::kWaveletTransform);
}

// AC-5: each transform descriptor rejects its own invalid parameters.
MYCV_TEST(TransformOperatorParamErrorsFail) {
  auto bad_levels = LoadOperatorsFromString(
      "operators:\n  - name: p\n    op: laplacian_pyramid\n"
      "    backend: native_cpu\n"
      "    params: {levels: 0, kernel_size: 3, sigma: 1.0, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(bad_levels.ok());
  if (!bad_levels.ok()) {
    MYCV_EXPECT_TRUE(bad_levels.status().code() ==
                     StatusCode::kInvalidArgument);
  }

  auto bad_kernel = LoadOperatorsFromString(
      "operators:\n  - name: p\n    op: laplacian_pyramid\n"
      "    backend: native_cpu\n"
      "    params: {levels: 2, kernel_size: 4, sigma: 1.0, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(bad_kernel.ok());
  if (!bad_kernel.ok()) {
    MYCV_EXPECT_TRUE(bad_kernel.status().code() ==
                     StatusCode::kInvalidArgument);
  }

  auto missing_layout = LoadOperatorsFromString(
      "operators:\n  - name: f\n    op: fourier_transform\n"
      "    backend: native_cpu\n    params: {}\n");
  MYCV_EXPECT_FALSE(missing_layout.ok());
  if (!missing_layout.ok()) {
    MYCV_EXPECT_TRUE(missing_layout.status().code() ==
                     StatusCode::kInvalidArgument);
  }

  auto bad_kind = LoadOperatorsFromString(
      "operators:\n  - name: w\n    op: wavelet_transform\n"
      "    backend: native_cpu\n    params: {kind: cdf97, layout: nhwc}\n");
  MYCV_EXPECT_FALSE(bad_kind.ok());
  if (!bad_kind.ok()) {
    MYCV_EXPECT_TRUE(bad_kind.status().code() ==
                     StatusCode::kInvalidArgument);
  }
}

// GetOperator returns nullptr for a missing name or a wrong concrete type, and
// the base Operator metadata is still reachable.
MYCV_TEST(GetOperatorTypeChecks) {
  const char* yaml = R"(
operators:
  - name: r
    op: resize
    backend: native_cpu
    params: {output_height: 2, output_width: 2, interpolation: bilinear, layout: nhwc}
)";
  auto loaded = LoadOperatorsFromString(yaml);
  MYCV_EXPECT_TRUE(loaded.ok());
  if (!loaded.ok()) return;
  const OperatorMap& ops = loaded.value();
  MYCV_EXPECT_TRUE(GetOperator<ResizeOperator>(ops, "r") != nullptr);
  MYCV_EXPECT_TRUE(GetOperator<GaussianBlurOperator>(ops, "r") == nullptr);
  MYCV_EXPECT_TRUE(GetOperator<ResizeOperator>(ops, "missing") == nullptr);
  const Operator* base = ops.at("r").get();
  MYCV_EXPECT_TRUE(base->kind() == OpKind::kResize);
  MYCV_EXPECT_TRUE(base->backend() == BackendType::kNativeCpu);
}

}  // namespace

MYCV_TEST_MAIN()
