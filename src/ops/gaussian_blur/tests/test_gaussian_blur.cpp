// 测试接口用于验证 gaussian_blur 的参数校验、各 CPU
// 内核版本位精确一致性，以及与 naive 基准的对拍。
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/backend.h"
#include "framework/image_utils.h"
#include "mycv/mycv.h"
#include "test_framework.h"

using namespace mycv;
using ops_internal::ImageByteOffset;
using ops_internal::ImageInfo;
using ops_internal::ParseImage;

namespace {

constexpr int kKernelSize = 5;
constexpr double kSigma = 1.2;

// T -> DataType selector (core/dtype.h exposes no trait template).
template <typename T>
constexpr DataType DTypeOf() {
  if constexpr (std::is_same_v<T, uint8_t>) {
    return DataType::kUInt8;
  } else if constexpr (std::is_same_v<T, float>) {
    return DataType::kFloat32;
  } else {
    return DataType::kFloat64;
  }
}

OpContext MakeOpContext() {
  static auto context = Context::Create(BackendType::kNativeCpu);
  static const bool reported = [] {
    if (!context.ok())
      std::cout << "SKIP gaussian_blur tests: " << context.status().ToString()
                << '\n';
    return true;
  }();
  (void)reported;
  if (!context.ok()) return OpContext{};
  OpContext op_context;
  op_context.context = context.value().get();
  op_context.cpu_level = CpuKernelLevel::kNaive;
  return op_context;
}

std::shared_ptr<Context> GetCpuContext() {
  static auto context = Context::Create(BackendType::kNativeCpu);
  return context.ok() ? context.value() : nullptr;
}

Result<Tensor> CopyTo(Context& destination, const Tensor& source) {
  auto output = destination.AllocateTensor(source.shape(), source.dtype());
  if (!output.ok()) return output.status();
  MYCV_RETURN_IF_ERROR(destination.Copy(&output.value(), source));
  return output;
}

std::vector<CpuKernelLevel> AvailableLevels() {
  static auto context = Context::Create(BackendType::kNativeCpu);
  std::vector<CpuKernelLevel> levels;
  if (!context.ok()) return levels;
  for (CpuKernelLevel level :
       {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
        CpuKernelLevel::kSimdGeneric, CpuKernelLevel::kSimdAvx2,
        CpuKernelLevel::kTbb}) {
    if (context.value()->backend_ptr()->SupportsKernelLevel(level)) {
      levels.push_back(level);
    }
  }
  return levels;
}

// Deterministic logical-pixel value shared by every layout/rank, so tensors
// built for different layouts hold the same image content.
double PixelValue(int64_t n, int64_t y, int64_t x, int64_t c) {
  const uint64_t hash = (static_cast<uint64_t>(n) * 73856093ULL) ^
                        (static_cast<uint64_t>(y) * 19349663ULL) ^
                        (static_cast<uint64_t>(x) * 83492791ULL) ^
                        (static_cast<uint64_t>(c) * 2971215073ULL);
  return static_cast<double>(hash % 251ULL) +
         0.25 * static_cast<double>((x + c) % 4);
}

// Builds a rank-3 (NHWC {h,w,c} / NCHW {c,h,w}) or rank-4 (NHWC {n,h,w,c} /
// NCHW {n,c,h,w}) image tensor filled with PixelValue per logical pixel.
template <typename T>
Tensor MakeImage(int64_t n, int64_t h, int64_t w, int64_t c, ImageLayout layout,
                 int rank) {
  const Shape shape =
      rank == 3
          ? (layout == ImageLayout::kNHWC ? Shape({h, w, c}) : Shape({c, h, w}))
          : (layout == ImageLayout::kNHWC ? Shape({n, h, w, c})
                                          : Shape({n, c, h, w}));
  auto context = Context::Create(BackendType::kNativeCpu);
  auto tensor = context.value()->AllocateTensor(shape, DTypeOf<T>());
  T* data = static_cast<T*>(tensor.value().data());
  const ImageInfo info{rank, n, h, w, c, layout};
  // Rank-3 holds a single image; the n coordinate does not exist there.
  const int64_t images = rank == 4 ? n : 1;
  for (int64_t ni = 0; ni < images; ++ni) {
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        for (int64_t ci = 0; ci < c; ++ci) {
          const size_t offset =
              ops_internal::ImageByteOffset(tensor.value(), info, ni, y, x, ci);
          data[offset / sizeof(T)] = static_cast<T>(PixelValue(ni, y, x, ci));
        }
      }
    }
  }
  return tensor.value();
}

double ReadPixel(const Tensor& tensor, const ops_internal::ImageInfo& info,
                 int64_t n, int64_t y, int64_t x, int64_t c) {
  const size_t offset = ops_internal::ImageByteOffset(tensor, info, n, y, x, c);
  const char* base = static_cast<const char*>(tensor.data());
  switch (tensor.dtype()) {
    case DataType::kUInt8:
      return *reinterpret_cast<const uint8_t*>(base + offset);
    case DataType::kFloat32:
      return *reinterpret_cast<const float*>(base + offset);
    default:
      return *reinterpret_cast<const double*>(base + offset);
  }
}

// Relative error |a-b| <= tol * max(1, |a|); same rule as the backend tests
// (docs/qa/testing.md §4.2). tol = 0 means exact equality.
bool ExpectPixel(double expected, double actual, double tolerance) {
  const double diff = std::fabs(expected - actual);
  return diff <= tolerance * std::fmax(1.0, std::fabs(expected));
}

ImageInfo InfoOf(const Tensor& tensor, ImageLayout layout) {
  auto info = ops_internal::ParseImage(tensor.shape(), layout);
  return info.ok() ? info.value() : ImageInfo{};
}

}  // namespace

MYCV_TEST(GaussianBlurRejectsBadKernelSize) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto image = MakeImage<uint8_t>(1, 8, 8, 3, ImageLayout::kNHWC, 4);
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, 4, kSigma).ok());
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, 1, kSigma).ok());
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, -3, kSigma).ok());
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, 3, 0.0).ok());
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, 3, -1.0).ok());
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, 3,
                                 std::numeric_limits<double>::infinity())
                        .ok());
  MYCV_EXPECT_FALSE(GaussianBlur(op_context, image, 3, std::nan("")).ok());
}

MYCV_TEST(GaussianBlurRejectsUnsupportedDType) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  const Shape shape({4, 4, 3});
  auto context = Context::Create(BackendType::kNativeCpu);
  auto f16 = context.value()->AllocateTensor(shape, DataType::kFloat16);
  if (!f16.ok()) return;
  auto result = GaussianBlur(op_context, f16.value(), kKernelSize, kSigma,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_FALSE(result.ok());
}

MYCV_TEST(GaussianBlurCpuLevelsAreBitExact) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  const auto levels = AvailableLevels();
  if (levels.empty()) return;
  struct LayoutCase {
    int rank;
    ImageLayout layout;
  };
  for (const LayoutCase& Case :
       {LayoutCase{3, ImageLayout::kNHWC}, LayoutCase{3, ImageLayout::kNCHW},
        LayoutCase{4, ImageLayout::kNHWC}, LayoutCase{4, ImageLayout::kNCHW}}) {
    auto reference_tensor =
        MakeImage<float>(1, 9, 11, 4, Case.layout, Case.rank);
    auto reference = GaussianBlur(op_context, reference_tensor, kKernelSize,
                                  kSigma, Case.layout);
    MYCV_EXPECT_TRUE(reference.ok());
    if (!reference.ok()) return;
    for (CpuKernelLevel level : levels) {
      if (level == CpuKernelLevel::kNaive) continue;
      OpContext level_context = op_context;
      level_context.cpu_level = level;
      auto result = GaussianBlur(level_context, reference_tensor, kKernelSize,
                                 kSigma, Case.layout);
      MYCV_EXPECT_TRUE(result.ok());
      if (!result.ok()) continue;
      MYCV_EXPECT_TRUE(std::memcmp(reference.value().data(),
                                   result.value().data(),
                                   reference.value().ByteSize()) == 0);
    }
  }
  // u8 path: rounding rule (half away from zero) must be level-independent.
  auto reference_tensor =
      MakeImage<uint8_t>(1, 9, 11, 4, ImageLayout::kNHWC, 4);
  auto reference = GaussianBlur(op_context, reference_tensor, kKernelSize,
                                kSigma, ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  for (CpuKernelLevel level : levels) {
    if (level == CpuKernelLevel::kNaive) continue;
    OpContext level_context = op_context;
    level_context.cpu_level = level;
    auto result = GaussianBlur(level_context, reference_tensor, kKernelSize,
                               kSigma, ImageLayout::kNHWC);
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    MYCV_EXPECT_TRUE(std::memcmp(reference.value().data(),
                                 result.value().data(),
                                 reference.value().ByteSize()) == 0);
  }
}

MYCV_TEST(GaussianBlurLayoutsAgree) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  const int64_t n = 2, h = 6, w = 7, c = 3;
  auto reference = GaussianBlur(
      op_context, MakeImage<float>(n, h, w, c, ImageLayout::kNHWC, 4),
      kKernelSize, kSigma, ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  const ImageInfo reference_info =
      InfoOf(reference.value(), ImageLayout::kNHWC);
  for (const auto& [rank, layout] :
       {std::pair<int, ImageLayout>{3, ImageLayout::kNHWC},
        std::pair<int, ImageLayout>{3, ImageLayout::kNCHW},
        std::pair<int, ImageLayout>{4, ImageLayout::kNCHW}}) {
    auto result =
        GaussianBlur(op_context, MakeImage<float>(n, h, w, c, layout, rank),
                     kKernelSize, kSigma, layout);
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    const ImageInfo info = InfoOf(result.value(), layout);
    const int64_t images = rank == 4 ? n : 1;
    bool all_equal = true;
    for (int64_t ni = 0; ni < images && all_equal; ++ni) {
      for (int64_t y = 0; y < h && all_equal; ++y) {
        for (int64_t x = 0; x < w && all_equal; ++x) {
          for (int64_t ci = 0; ci < c && all_equal; ++ci) {
            all_equal = ExpectPixel(
                ReadPixel(reference.value(), reference_info, ni, y, x, ci),
                ReadPixel(result.value(), info, ni, y, x, ci), 0.0);
          }
        }
      }
    }
    MYCV_EXPECT_TRUE(all_equal);
  }
}

MYCV_TEST(GaussianBlurConstantImageStaysConstant) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto context = Context::Create(BackendType::kNativeCpu);
  const Shape shape({1, 5, 6, 3});
  auto image = context.value()->AllocateTensor(shape, DataType::kUInt8);
  MYCV_EXPECT_TRUE(image.ok());
  if (!image.ok()) return;
  std::memset(image.value().data(), 37, image.value().ByteSize());
  auto result = GaussianBlur(op_context, image.value(), kKernelSize, kSigma,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  const uint8_t* data = static_cast<const uint8_t*>(result.value().data());
  bool constant = true;
  for (int64_t i = 0; i < result.value().Numel() && constant; ++i) {
    constant = data[i] == 37;
  }
  MYCV_EXPECT_TRUE(constant);
}

namespace {

double Variance(const Tensor& tensor) {
  const double mean = [&] {
    double sum = 0.0;
    for (int64_t i = 0; i < tensor.Numel(); ++i) {
      sum += static_cast<const double*>(tensor.data())[i];
    }
    return sum / static_cast<double>(tensor.Numel());
  }();
  double sum = 0.0;
  for (int64_t i = 0; i < tensor.Numel(); ++i) {
    const double delta = static_cast<const double*>(tensor.data())[i] - mean;
    sum += delta * delta;
  }
  return sum / static_cast<double>(tensor.Numel());
}

}  // namespace

MYCV_TEST(GaussianBlurImpulseResponseMatchesSeparableKernel) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  constexpr int64_t kSize = 9;
  constexpr int kK = 5;
  constexpr double kSigmaTest = 1.2;
  auto context = GetCpuContext();
  if (!context) return;
  auto image =
      context->AllocateTensor(Shape{1, kSize, kSize, 1}, DataType::kFloat64);
  MYCV_EXPECT_TRUE(image.ok());
  if (!image.ok()) return;
  std::memset(image.value().data(), 0, image.value().ByteSize());
  auto* pixels = static_cast<double*>(image.value().data());
  pixels[(kSize / 2) * kSize + kSize / 2] = 1.0;

  auto result = GaussianBlur(op_context, image.value(), kK, kSigmaTest,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;

  // Independent oracle: normalised 1-D kernel, response = k[i] * k[j].
  const int radius = kK / 2;
  std::vector<double> kernel(kK);
  double sum = 0.0;
  for (int i = 0; i < kK; ++i) {
    const double offset = static_cast<double>(i - radius);
    kernel[i] = std::exp(-0.5 * offset * offset / (kSigmaTest * kSigmaTest));
    sum += kernel[i];
  }
  for (double& weight : kernel)
    weight /= sum;

  const double* output = static_cast<const double*>(result.value().data());
  bool matches = true;
  for (int64_t y = 0; y < kSize && matches; ++y) {
    for (int64_t x = 0; x < kSize && matches; ++x) {
      const int dy = static_cast<int>(y) - kSize / 2;
      const int dx = static_cast<int>(x) - kSize / 2;
      const double expected = (std::abs(dy) <= radius && std::abs(dx) <= radius)
                                  ? kernel[dy + radius] * kernel[dx + radius]
                                  : 0.0;
      matches = std::fabs(expected - output[y * kSize + x]) <= 1e-12;
    }
  }
  MYCV_EXPECT_TRUE(matches);
  MYCV_EXPECT_EQ(result.value().dtype(), DataType::kFloat64);
}

MYCV_TEST(GaussianBlurRank2Float64ConstantStaysConstant) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto context = GetCpuContext();
  if (!context) return;
  auto image = context->AllocateTensor(Shape{7, 7}, DataType::kFloat64);
  MYCV_EXPECT_TRUE(image.ok());
  if (!image.ok()) return;
  auto* pixels = static_cast<double*>(image.value().data());
  for (int64_t i = 0; i < image.value().Numel(); ++i)
    pixels[i] = 0.5;

  auto result =
      GaussianBlur(op_context, image.value(), 3, 1.0, ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  // Rank-2 HW keeps its shape and dtype.
  MYCV_EXPECT_TRUE(result.value().shape() == Shape({7, 7}));
  MYCV_EXPECT_EQ(result.value().dtype(), DataType::kFloat64);
  const double* output = static_cast<const double*>(result.value().data());
  bool constant = true;
  for (int64_t i = 0; i < result.value().Numel() && constant; ++i) {
    constant = std::fabs(output[i] - 0.5) <= 1e-12;
  }
  MYCV_EXPECT_TRUE(constant);
}

MYCV_TEST(GaussianBlurLargerSigmaReducesVariance) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto context = GetCpuContext();
  if (!context) return;
  auto image =
      context->AllocateTensor(Shape{1, 24, 24, 1}, DataType::kFloat64).value();
  auto* pixels = static_cast<double*>(image.data());
  uint64_t state = 12345;
  for (int64_t i = 0; i < image.Numel(); ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    pixels[i] = static_cast<double>((state >> 33) % 256) / 255.0;
  }
  const double original = Variance(image);
  auto mild = GaussianBlur(op_context, image, 3, 0.5, ImageLayout::kNHWC);
  auto strong = GaussianBlur(op_context, image, 9, 3.0, ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(mild.ok());
  MYCV_EXPECT_TRUE(strong.ok());
  if (!mild.ok() || !strong.ok()) return;
  const double mild_variance = Variance(mild.value());
  const double strong_variance = Variance(strong.value());
  MYCV_EXPECT_TRUE(mild_variance < original);
  MYCV_EXPECT_TRUE(strong_variance < mild_variance);
}

MYCV_TEST(GaussianBlurVectorizedLevelsRejectStridedInput) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto context = GetCpuContext();
  if (!context) return;
  auto base =
      context->AllocateTensor(Shape{4, 12, 4}, DataType::kFloat32).value();
  for (int64_t i = 0; i < base.Numel(); ++i) {
    base.Data<float>()[i] = static_cast<float>(i % 13);
  }
  auto strided = base.Slice(1, 1, 9);
  MYCV_EXPECT_TRUE(strided.ok());
  if (!strided.ok()) return;

  // Scalar path handles views.
  auto naive =
      GaussianBlur(op_context, strided.value(), 3, 1.0, ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(naive.ok());

  for (CpuKernelLevel level :
       {CpuKernelLevel::kSimdGeneric, CpuKernelLevel::kSimdAvx2}) {
    if (!context->backend_ptr()->SupportsKernelLevel(level)) continue;
    OpContext level_context = op_context;
    level_context.cpu_level = level;
    auto result = GaussianBlur(level_context, strided.value(), 3, 1.0,
                               ImageLayout::kNHWC);
    MYCV_EXPECT_EQ(result.status().code(), StatusCode::kUnsupported);
  }
}

#ifdef MYCV_WITH_CUDA
namespace {

std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP GaussianBlur CUDA tests: "
                << result.status().ToString() << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}

}  // namespace

MYCV_TEST(GaussianBlurCudaMatchesCpu) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto cuda_context = GetCudaContext();
  if (!cuda_context) return;
  auto source_tensor = MakeImage<float>(1, 24, 32, 3, ImageLayout::kNHWC, 4);
  auto reference = GaussianBlur(op_context, source_tensor, kKernelSize, kSigma,
                                ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  OpContext cuda_op = op_context;
  cuda_op.context = cuda_context.get();
  auto cpu_context = GetCpuContext();
  if (!cpu_context) return;
  auto copied = CopyTo(*cuda_context, source_tensor);
  MYCV_EXPECT_TRUE(copied.ok());
  if (!copied.ok()) return;
  auto result = GaussianBlur(cuda_op, copied.value(), kKernelSize, kSigma,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  auto back = CopyTo(*cpu_context, result.value());
  MYCV_EXPECT_TRUE(back.ok());
  if (!back.ok()) return;
  const ImageInfo info = InfoOf(reference.value(), ImageLayout::kNHWC);
  const ImageInfo result_info = InfoOf(back.value(), ImageLayout::kNHWC);
  bool equivalent = true;
  for (int64_t y = 0; y < info.h && equivalent; ++y) {
    for (int64_t x = 0; x < info.w && equivalent; ++x) {
      for (int64_t c = 0; c < info.c && equivalent; ++c) {
        equivalent =
            ExpectPixel(ReadPixel(reference.value(), info, 0, y, x, c),
                        ReadPixel(back.value(), result_info, 0, y, x, c), 1e-4);
      }
    }
  }
  MYCV_EXPECT_TRUE(equivalent);
}

MYCV_TEST(GaussianBlurCudaMatchesCpuU8) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto cuda_context = GetCudaContext();
  if (!cuda_context) return;
  auto cpu_context = GetCpuContext();
  if (!cpu_context) return;
  auto source_tensor = MakeImage<uint8_t>(1, 17, 23, 3, ImageLayout::kNHWC, 4);
  auto reference = GaussianBlur(op_context, source_tensor, kKernelSize, kSigma,
                                ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  OpContext cuda_op = op_context;
  cuda_op.context = cuda_context.get();
  auto copied = CopyTo(*cuda_context, source_tensor);
  MYCV_EXPECT_TRUE(copied.ok());
  if (!copied.ok()) return;
  auto result = GaussianBlur(cuda_op, copied.value(), kKernelSize, kSigma,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  auto back = CopyTo(*cpu_context, result.value());
  MYCV_EXPECT_TRUE(back.ok());
  if (!back.ok()) return;
  // u8 cross-backend: ±1 absolute (documented in README.md).
  bool within_one = true;
  for (int64_t i = 0; i < back.value().Numel() && within_one; ++i) {
    within_one =
        std::fabs(static_cast<double>(back.value().Data<uint8_t>()[i]) -
                  static_cast<double>(reference.value().Data<uint8_t>()[i])) <=
        1.0;
  }
  MYCV_EXPECT_TRUE(within_one);
}
#endif

#ifdef MYCV_WITH_OPENCV
namespace {

std::shared_ptr<Context> GetOpenCvContext() {
  static auto result = Context::Create(BackendType::kOpenCV);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP GaussianBlur OpenCV tests: "
                << result.status().ToString() << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}

}  // namespace

MYCV_TEST(GaussianBlurOpenCvMatchesCpu) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto opencv_context = GetOpenCvContext();
  if (!opencv_context) return;
  auto source_tensor = MakeImage<uint8_t>(1, 24, 32, 3, ImageLayout::kNHWC, 4);
  auto reference = GaussianBlur(op_context, source_tensor, kKernelSize, kSigma,
                                ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  OpContext opencv_op = op_context;
  opencv_op.context = opencv_context.get();
  auto cpu_context = GetCpuContext();
  if (!cpu_context) return;
  auto copied = CopyTo(*opencv_context, source_tensor);
  MYCV_EXPECT_TRUE(copied.ok());
  if (!copied.ok()) return;
  auto result = GaussianBlur(opencv_op, copied.value(), kKernelSize, kSigma,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  auto back = CopyTo(*cpu_context, result.value());
  MYCV_EXPECT_TRUE(back.ok());
  if (!back.ok()) return;
  const ImageInfo info = InfoOf(reference.value(), ImageLayout::kNHWC);
  const ImageInfo result_info = InfoOf(back.value(), ImageLayout::kNHWC);
  bool equivalent = true;
  for (int64_t y = 0; y < info.h && equivalent; ++y) {
    for (int64_t x = 0; x < info.w && equivalent; ++x) {
      for (int64_t c = 0; c < info.c && equivalent; ++c) {
        // u8 cross-backend: ±1 absolute (host/device + fixed-point internals).
        equivalent =
            std::fabs(ReadPixel(reference.value(), info, 0, y, x, c) -
                      ReadPixel(back.value(), result_info, 0, y, x, c)) <= 1.0;
      }
    }
  }
  MYCV_EXPECT_TRUE(equivalent);
}

MYCV_TEST(GaussianBlurOpenCvMatchesCpuFloat) {
  auto op_context = MakeOpContext();
  if (op_context.context == nullptr) return;
  auto opencv_context = GetOpenCvContext();
  if (!opencv_context) return;
  auto cpu_context = GetCpuContext();
  if (!cpu_context) return;
  auto source_tensor = MakeImage<float>(1, 17, 23, 3, ImageLayout::kNHWC, 4);
  auto reference = GaussianBlur(op_context, source_tensor, kKernelSize, kSigma,
                                ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  OpContext opencv_op = op_context;
  opencv_op.context = opencv_context.get();
  auto copied = CopyTo(*opencv_context, source_tensor);
  MYCV_EXPECT_TRUE(copied.ok());
  if (!copied.ok()) return;
  auto result = GaussianBlur(opencv_op, copied.value(), kKernelSize, kSigma,
                             ImageLayout::kNHWC);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  auto back = CopyTo(*cpu_context, result.value());
  MYCV_EXPECT_TRUE(back.ok());
  if (!back.ok()) return;
  const ImageInfo info = InfoOf(reference.value(), ImageLayout::kNHWC);
  const ImageInfo result_info = InfoOf(back.value(), ImageLayout::kNHWC);
  bool equivalent = true;
  for (int64_t y = 0; y < info.h && equivalent; ++y) {
    for (int64_t x = 0; x < info.w && equivalent; ++x) {
      for (int64_t c = 0; c < info.c && equivalent; ++c) {
        equivalent =
            ExpectPixel(ReadPixel(reference.value(), info, 0, y, x, c),
                        ReadPixel(back.value(), result_info, 0, y, x, c), 1e-4);
      }
    }
  }
  MYCV_EXPECT_TRUE(equivalent);
}
#endif

MYCV_TEST_MAIN();
