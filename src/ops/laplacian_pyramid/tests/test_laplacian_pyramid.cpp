// Tests validate Laplacian pyramid level counts, reconstruction error, and the
// composed resize/gaussian_blur behavior.
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "core/backend.h"
#include "hpcplatform/hpcplatform.h"
#include "hpcplatform/core/transfer.h"
namespace {
using namespace hpcplatform;
constexpr int kKernelSize = 5;
constexpr double kSigma = 1.2;
OpContext CpuOpContext(Context& context) {
  return OpContext(context);
}
std::shared_ptr<Context> GetCpuContext() {
  static auto context = Context::Create(BackendType::kNativeCpu);
  return context.ok() ? context.value() : nullptr;
}
Tensor MakeImage(int64_t n, int64_t h, int64_t w, int64_t c,
                 ImageLayout layout) {
  const Shape shape =
      layout == ImageLayout::kNHWC ? Shape{n, h, w, c} : Shape{n, c, h, w};
  auto context = GetCpuContext();
  if (!context) return Tensor{};
  auto image = Tensor::Empty(*context, shape, DataType::kFloat32);
  EXPECT_TRUE(image.ok());
  if (!image.ok()) return Tensor{};
  float* data = image.value().Data<float>();
  for (int64_t i = 0; i < image.value().Numel(); ++i) {
    data[i] = static_cast<float>((i * 13) % 97) * 0.01f;
  }
  return image.value();
}
Tensor Subtract(const Tensor& fine, const Tensor& coarse) {
  auto context = GetCpuContext();
  if (!context) return Tensor{};
  auto output = Tensor::Empty(*context, fine.shape(), DataType::kFloat32);
  EXPECT_TRUE(output.ok());
  if (!output.ok()) return Tensor{};
  const float* a = fine.Data<float>();
  const float* b = coarse.Data<float>();
  float* out = output.value().Data<float>();
  for (int64_t i = 0; i < output.value().Numel(); ++i)
    out[i] = a[i] - b[i];
  return output.value();
}
bool BitExact(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  return std::memcmp(a.data(), b.data(), a.ByteSize()) == 0;
}
bool Near(const Tensor& a, const Tensor& b, double tolerance) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  const float* lhs = a.Data<float>();
  const float* rhs = b.Data<float>();
  for (int64_t i = 0; i < a.Numel(); ++i) {
    const double expected = lhs[i];
    const double actual = rhs[i];
    if (std::fabs(expected - actual) >
        tolerance * std::fmax(1.0, std::fabs(expected))) {
      return false;
    }
  }
  return true;
}
Result<Tensor> CopyTo(Context& destination, const Tensor& source) {
  auto output = Tensor::Empty(destination, source.shape(), source.dtype());
  if (!output.ok()) return output.status();
  hpcplatform_RETURN_IF_ERROR(Copy(source, &output.value()));
  return output;
}
}  // namespace
TEST(hpcplatform, LaplacianPyramidShapesAndLevelCount) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  struct Case {
    ImageLayout layout;
    int64_t n;
    int64_t c;
  };
  const Case cases[] = {{ImageLayout::kNHWC, 1, 3}, {ImageLayout::kNCHW, 2, 3}};
  for (const Case& item : cases) {
    auto input = MakeImage(item.n, 9, 11, item.c, item.layout);
    auto result = BuildLaplacianPyramid(CpuOpContext(*context), input, 3,
                                        kKernelSize, kSigma, item.layout);
    EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    EXPECT_EQ(result.value().levels.size(), size_t{4});
    if (result.value().levels.size() != 4) continue;
    // 9x11 -> 5x6 -> 3x3 -> 2x2; only the last stage is 2x2, so the request
    // is legal and every band keeps the dtype/backend of the input.
    const int64_t expected_h[] = {9, 5, 3, 2};
    const int64_t expected_w[] = {11, 6, 3, 2};
    bool shapes_ok = true;
    for (size_t i = 0; i < 4 && shapes_ok; ++i) {
      const Tensor& level = result.value().levels[i];
      shapes_ok = level.dtype() == DataType::kFloat32 &&
                  level.backend() == BackendType::kNativeCpu &&
                  level.IsContiguous();
      const Shape expected =
          item.layout == ImageLayout::kNHWC
              ? Shape{item.n, expected_h[i], expected_w[i], item.c}
              : Shape{item.n, item.c, expected_h[i], expected_w[i]};
      shapes_ok = shapes_ok && level.shape() == expected;
    }
    EXPECT_TRUE(shapes_ok);
  }
}
TEST(hpcplatform, LaplacianPyramidConstantImageHasZeroBands) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = Tensor::Empty(*context, Shape{1, 7, 9, 2}, DataType::kFloat32);
  EXPECT_TRUE(input.ok());
  if (!input.ok()) return;
  std::fill(input.value().Data<float>(),
            input.value().Data<float>() + input.value().Numel(), 0.25f);
  auto result = BuildLaplacianPyramid(CpuOpContext(*context), input.value(), 2,
                                      kKernelSize, 1.0);
  EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  // Blur/downsample/upsample all preserve a constant image, so every band is
  // zero and the coarse residual is the constant itself.
  bool bands_zero = true;
  for (size_t i = 0; i + 1 < result.value().levels.size() && bands_zero; ++i) {
    const float* band = result.value().levels[i].Data<float>();
    for (int64_t j = 0; j < result.value().levels[i].Numel() && bands_zero;
         ++j) {
      bands_zero = std::fabs(band[j]) <= 1e-6f;
    }
  }
  EXPECT_TRUE(bands_zero);
  const Tensor& coarsest = result.value().levels.back();
  bool coarse_constant = true;
  for (int64_t i = 0; i < coarsest.Numel() && coarse_constant; ++i) {
    coarse_constant = std::fabs(coarsest.Data<float>()[i] - 0.25f) <= 1e-6f;
  }
  EXPECT_TRUE(coarse_constant);
}
TEST(hpcplatform, LaplacianPyramidMatchesReferenceComposition) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  const OpContext op_context = CpuOpContext(*context);
  auto input = MakeImage(1, 8, 8, 2, ImageLayout::kNHWC);
  auto result = BuildLaplacianPyramid(op_context, input, 3, kKernelSize, kSigma,
                                      ImageLayout::kNHWC);
  EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  // Independent composition written with the public single-level ops.
  Tensor gaussian = input;
  for (int level = 0; level < 3; ++level) {
    const int64_t h = gaussian.shape().Dim(1);
    const int64_t w = gaussian.shape().Dim(2);
    auto blurred = GaussianBlur(op_context, gaussian, kKernelSize, kSigma,
                                ImageLayout::kNHWC);
    EXPECT_TRUE(blurred.ok());
    if (!blurred.ok()) return;
    auto next = Resize(op_context, blurred.value(), (h + 1) / 2, (w + 1) / 2,
                       InterpolationMode::kNearest, ImageLayout::kNHWC);
    EXPECT_TRUE(next.ok());
    if (!next.ok()) return;
    auto up = Resize(op_context, next.value(), h, w,
                     InterpolationMode::kBilinear, ImageLayout::kNHWC);
    EXPECT_TRUE(up.ok());
    if (!up.ok()) return;
    const Tensor expected = Subtract(gaussian, up.value());
    EXPECT_TRUE(BitExact(expected, result.value().levels[level]));
    gaussian = next.value();
  }
  EXPECT_TRUE(BitExact(gaussian, result.value().levels.back()));
}
TEST(hpcplatform, LaplacianPyramidCpuLevelsAreBitExact) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = MakeImage(1, 17, 19, 3, ImageLayout::kNHWC);
  auto reference = BuildLaplacianPyramid(CpuOpContext(*context), input, 2,
                                         kKernelSize, kSigma);
  EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  for (CpuKernelLevel level :
       {CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
        CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kTbb}) {
    if (!context->SupportsKernelLevel(level)) continue;
    OpContext level_context = CpuOpContext(*context);
    level_context.set_cpu_level(level);
    auto actual =
        BuildLaplacianPyramid(level_context, input, 2, kKernelSize, kSigma);
    // Backend-wide capability is broader than this operator's matrix (no SIMD
    // until resize grows one); an unavailable level is skipped.
    if (!actual.ok() && actual.status().code() == StatusCode::kUnsupported) {
      continue;
    }
    EXPECT_TRUE(actual.ok());
    if (!actual.ok()) continue;
    EXPECT_EQ(actual.value().levels.size(),
                   reference.value().levels.size());
    bool identical = true;
    for (size_t i = 0; i < actual.value().levels.size() && identical; ++i) {
      identical =
          BitExact(actual.value().levels[i], reference.value().levels[i]);
    }
    EXPECT_TRUE(identical);
  }
}
TEST(hpcplatform, LaplacianPyramidRejectsInvalidParameters) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  const OpContext op_context = CpuOpContext(*context);
  auto valid = MakeImage(1, 8, 8, 3, ImageLayout::kNHWC);
  auto bad_levels = BuildLaplacianPyramid(op_context, valid, 0);
  EXPECT_EQ(bad_levels.status().code(), StatusCode::kInvalidArgument);
  auto even_kernel = BuildLaplacianPyramid(op_context, valid, 1, 4, 1.0);
  EXPECT_EQ(even_kernel.status().code(), StatusCode::kInvalidArgument);
  auto bad_sigma = BuildLaplacianPyramid(op_context, valid, 1, 5, 0.0);
  EXPECT_EQ(bad_sigma.status().code(), StatusCode::kInvalidArgument);
  auto null_context = BuildLaplacianPyramid(OpContext::Null(), valid, 1);
  EXPECT_EQ(null_context.status().code(), StatusCode::kInvalidArgument);
  auto u8 = Tensor::Empty(*context, Shape{1, 8, 8, 3}, DataType::kUInt8);
  EXPECT_TRUE(u8.ok());
  if (u8.ok()) {
    // Signed bands cannot be stored in u8.
    auto unsupported = BuildLaplacianPyramid(op_context, u8.value(), 1);
    EXPECT_EQ(unsupported.status().code(), StatusCode::kUnsupported);
  }
  auto rank5 =
      Tensor::Empty(*context, Shape{1, 1, 2, 2, 2}, DataType::kFloat32);
  EXPECT_TRUE(rank5.ok());
  if (rank5.ok()) {
    auto bad_rank = BuildLaplacianPyramid(op_context, rank5.value(), 1);
    EXPECT_EQ(bad_rank.status().code(), StatusCode::kInvalidArgument);
  }
  auto tiny = Tensor::Empty(*context, Shape{1, 1, 1}, DataType::kFloat32);
  EXPECT_TRUE(tiny.ok());
  if (tiny.ok()) {
    auto too_deep = BuildLaplacianPyramid(op_context, tiny.value(), 1);
    EXPECT_EQ(too_deep.status().code(), StatusCode::kInvalidArgument);
  }
}
#ifdef hpcplatform_WITH_OPENCV
namespace {
std::shared_ptr<Context> GetOpenCvContext() {
  static auto result = Context::Create(BackendType::kOpenCV);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP laplacian_pyramid OpenCV tests: "
                << result.status().ToString() << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}
}  // namespace
TEST(hpcplatform, LaplacianPyramidOpenCvMatchesCpuFloat) {
  auto cpu = Context::Create(BackendType::kNativeCpu).value();
  auto opencv = GetOpenCvContext();
  if (!opencv) return;
  auto input = MakeImage(1, 24, 32, 3, ImageLayout::kNHWC);
  auto reference =
      BuildLaplacianPyramid(CpuOpContext(*cpu), input, 2, kKernelSize, kSigma);
  EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  auto device = CopyTo(*opencv, input);
  EXPECT_TRUE(device.ok());
  if (!device.ok()) return;
  auto actual = BuildLaplacianPyramid(CpuOpContext(*opencv), device.value(), 2,
                                      kKernelSize, kSigma);
  EXPECT_TRUE(actual.ok());
  if (!actual.ok()) return;
  EXPECT_EQ(actual.value().levels.size(), reference.value().levels.size());
  bool equivalent = true;
  for (size_t i = 0; i < actual.value().levels.size() && equivalent; ++i) {
    auto back = CopyTo(*cpu, actual.value().levels[i]);
    EXPECT_TRUE(back.ok());
    if (!back.ok()) return;
    equivalent = Near(reference.value().levels[i], back.value(), 1e-4);
  }
  EXPECT_TRUE(equivalent);
}
#endif
#ifdef hpcplatform_WITH_CUDA
namespace {
std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP laplacian_pyramid CUDA tests: "
                << result.status().ToString() << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}
}  // namespace
TEST(hpcplatform, LaplacianPyramidCudaMatchesCpuFloat) {
  auto cpu = Context::Create(BackendType::kNativeCpu).value();
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto input = MakeImage(1, 24, 32, 3, ImageLayout::kNHWC);
  auto reference =
      BuildLaplacianPyramid(CpuOpContext(*cpu), input, 2, kKernelSize, kSigma);
  EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  auto device = CopyTo(*cuda, input);
  EXPECT_TRUE(device.ok());
  if (!device.ok()) return;
  auto actual = BuildLaplacianPyramid(CpuOpContext(*cuda), device.value(), 2,
                                      kKernelSize, kSigma);
  EXPECT_TRUE(actual.ok());
  if (!actual.ok()) return;
  EXPECT_EQ(actual.value().levels.size(), reference.value().levels.size());
  bool equivalent = true;
  for (size_t i = 0; i < actual.value().levels.size() && equivalent; ++i) {
    auto back = CopyTo(*cpu, actual.value().levels[i]);
    EXPECT_TRUE(back.ok());
    if (!back.ok()) return;
    equivalent = Near(reference.value().levels[i], back.value(), 1e-4);
  }
  EXPECT_TRUE(equivalent);
}
#endif