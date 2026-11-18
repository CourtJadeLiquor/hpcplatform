#include "test_framework.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>

#include "mycv/mycv.h"

namespace {

using namespace mycv;

std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok()) std::cout << "SKIP CUDA tests: " << result.status().ToString() << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}

OpContext Ops(Context& context, CpuKernelLevel level = CpuKernelLevel::kNaive) {
  OpContext result;
  result.context = &context;
  result.stream = &context.stream();
  result.cpu_level = level;
  return result;
}

Result<Tensor> CopyTo(Context& destination, const Tensor& source) {
  auto output = destination.AllocateTensor(source.shape(), source.dtype());
  if (!output.ok()) return output.status();
  MYCV_RETURN_IF_ERROR(destination.Copy(&output.value(), source));
  return output;
}

void ExpectEquivalent(const Tensor& expected, const Tensor& actual,
                      Context& native, double tolerance = 1e-4) {
  MYCV_EXPECT_TRUE(actual.shape() == expected.shape());
  MYCV_EXPECT_EQ(actual.dtype(), expected.dtype());
  if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype()) return;
  auto host = CopyTo(native, actual);
  MYCV_EXPECT_TRUE(host.ok());
  if (!host.ok()) return;
  for (int64_t i = 0; i < expected.Numel(); ++i) {
    if (expected.dtype() == DataType::kFloat32) {
      const double lhs = expected.Data<float>()[i];
      const double rhs = host.value().Data<float>()[i];
      MYCV_EXPECT_TRUE(std::fabs(lhs - rhs) <= tolerance * std::max(1.0, std::fabs(lhs)));
    } else if (expected.dtype() == DataType::kFloat64) {
      const double lhs = expected.Data<double>()[i];
      const double rhs = host.value().Data<double>()[i];
      MYCV_EXPECT_TRUE(std::fabs(lhs - rhs) <= tolerance * std::max(1.0, std::fabs(lhs)));
    } else {
      MYCV_EXPECT_EQ(std::memcmp(expected.data(), host.value().data(), expected.ByteSize()), 0);
      return;
    }
  }
}

MYCV_TEST(CudaCopyRoundTripAndMemoryLifetime) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto input = native->AllocateTensor(Shape{5, 7}, DataType::kFloat32).value();
  for (int i = 0; i < 35; ++i) input.Data<float>()[i] = static_cast<float>(i) * 0.5f;
  auto device = CopyTo(*cuda, input);
  MYCV_EXPECT_TRUE(device.ok());
  if (!device.ok()) return;
  MYCV_EXPECT_EQ(device.value().buffer()->kind(), MemoryKind::kDevice);
  MYCV_EXPECT_FALSE(cuda->stream().synchronous());
  auto output = CopyTo(*native, device.value());
  MYCV_EXPECT_TRUE(output.ok());
  if (output.ok()) {
    MYCV_EXPECT_EQ(std::memcmp(input.data(), output.value().data(), input.ByteSize()), 0);
  }
}

MYCV_TEST(CudaTensorCanOutliveItsContext) {
  if (!GetCudaContext()) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto input = native->AllocateTensor(Shape{8}, DataType::kFloat32).value();
  for (int i = 0; i < 8; ++i) input.Data<float>()[i] = static_cast<float>(i + 2);
  Tensor device;
  {
    auto local_result = Context::Create(BackendType::kCUDA);
    MYCV_EXPECT_TRUE(local_result.ok());
    if (!local_result.ok()) return;
    auto local = local_result.value();
    auto allocated = CopyTo(*local, input);
    MYCV_EXPECT_TRUE(allocated.ok());
    if (!allocated.ok()) return;
    device = allocated.value();
  }
  auto output = CopyTo(*native, device);
  MYCV_EXPECT_TRUE(output.ok());
  if (output.ok()) {
    MYCV_EXPECT_EQ(std::memcmp(input.data(), output.value().data(), input.ByteSize()), 0);
  }
}

MYCV_TEST(CudaMemoryStatsReturnToZeroWithCaching) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  {
    auto first = cuda->AllocateTensor(Shape{64}, DataType::kFloat32).value();
    auto second = cuda->AllocateTensor(Shape{32}, DataType::kUInt8).value();
    MYCV_EXPECT_EQ(cuda->memory().Stats().allocation_count, static_cast<size_t>(2));
    MYCV_EXPECT_EQ(cuda->memory().Stats().allocated_bytes, first.ByteSize() + second.ByteSize());
  }
  MYCV_EXPECT_EQ(cuda->memory().Stats().allocation_count, static_cast<size_t>(0));
  MYCV_EXPECT_EQ(cuda->memory().Stats().allocated_bytes, static_cast<size_t>(0));
  auto reused = cuda->AllocateTensor(Shape{64}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(reused.ok());
}

MYCV_TEST(CudaRejectsCpuOptimizationLevel) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto input = cuda->AllocateTensor(Shape{2}, DataType::kFloat32).value();
  auto output = Add(Ops(*cuda, CpuKernelLevel::kSimdAvx2), input, input);
  MYCV_EXPECT_EQ(output.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(CudaOperatorsMatchNativeCpu) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto input = native->AllocateTensor(Shape{4, 5, 3}, DataType::kFloat32).value();
  auto rhs = native->AllocateTensor(Shape{4, 5, 3}, DataType::kFloat32).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<float>()[i] = static_cast<float>(i % 17) * 0.25f;
    rhs.Data<float>()[i] = static_cast<float>(i % 5 + 1) * 0.5f;
  }
  auto device_input = CopyTo(*cuda, input).value();
  auto device_rhs = CopyTo(*cuda, rhs).value();

  auto reference = Add(Ops(*native), input, rhs).value();
  auto actual = Add(Ops(*cuda), device_input, device_rhs);
  MYCV_EXPECT_TRUE(actual.ok());
  if (actual.ok()) ExpectEquivalent(reference, actual.value(), *native);

  reference = Mul(Ops(*native), input, rhs).value();
  actual = Mul(Ops(*cuda), device_input, device_rhs);
  MYCV_EXPECT_TRUE(actual.ok());
  if (actual.ok()) ExpectEquivalent(reference, actual.value(), *native);

  auto scalar = native->AllocateTensor(Shape{}, DataType::kFloat32).value();
  scalar.Data<float>()[0] = 2.0f;
  auto device_scalar = CopyTo(*cuda, scalar).value();
  auto reference_scalar = Mul(Ops(*native), input, scalar).value();
  auto actual_scalar = Mul(Ops(*cuda), device_input, device_scalar);
  MYCV_EXPECT_TRUE(actual_scalar.ok());
  if (actual_scalar.ok()) ExpectEquivalent(reference_scalar, actual_scalar.value(), *native);

  auto reference_cast = Cast(Ops(*native), input, DataType::kFloat64).value();
  auto actual_cast = Cast(Ops(*cuda), device_input, DataType::kFloat64);
  MYCV_EXPECT_TRUE(actual_cast.ok());
  if (actual_cast.ok()) ExpectEquivalent(reference_cast, actual_cast.value(), *native);

  auto reference_norm = Normalize(Ops(*native), input, {0.5, 1.0, 1.5}, {2, 3, 4}).value();
  auto actual_norm = Normalize(Ops(*cuda), device_input, {0.5, 1.0, 1.5}, {2, 3, 4});
  MYCV_EXPECT_TRUE(actual_norm.ok());
  if (actual_norm.ok()) ExpectEquivalent(reference_norm, actual_norm.value(), *native);

  auto reference_resize = Resize(Ops(*native), input, 3, 4).value();
  auto actual_resize = Resize(Ops(*cuda), device_input, 3, 4);
  MYCV_EXPECT_TRUE(actual_resize.ok());
  if (actual_resize.ok()) ExpectEquivalent(reference_resize, actual_resize.value(), *native);

  auto reference_crop = Crop(Ops(*native), input, 1, 1, 2, 3).value();
  auto actual_crop = Crop(Ops(*cuda), device_input, 1, 1, 2, 3);
  MYCV_EXPECT_TRUE(actual_crop.ok());
  if (actual_crop.ok()) ExpectEquivalent(reference_crop, actual_crop.value(), *native);

  auto reference_transpose = Transpose(Ops(*native), input, {2, 0, 1}).value();
  auto actual_transpose = Transpose(Ops(*cuda), device_input, {2, 0, 1});
  MYCV_EXPECT_TRUE(actual_transpose.ok());
  if (actual_transpose.ok()) ExpectEquivalent(reference_transpose, actual_transpose.value(), *native);

  const std::array<double, 6> matrix{1, 0, 0.5, 0, 1, -0.25};
  auto reference_warp = WarpAffine(Ops(*native), input, matrix, 4, 5,
                                   InterpolationMode::kBilinear, -1.0).value();
  auto actual_warp = WarpAffine(Ops(*cuda), device_input, matrix, 4, 5,
                                InterpolationMode::kBilinear, -1.0);
  MYCV_EXPECT_TRUE(actual_warp.ok());
  if (actual_warp.ok()) ExpectEquivalent(reference_warp, actual_warp.value(), *native);
}

MYCV_TEST(CudaSupportsStridedInputsAndNchw) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto base = native->AllocateTensor(Shape{3, 5}, DataType::kFloat32).value();
  for (int i = 0; i < 15; ++i) base.Data<float>()[i] = static_cast<float>(i);
  auto slice = base.Slice(1, 1, 4).value();
  auto device_base = CopyTo(*cuda, base).value();
  auto device_slice = device_base.Slice(1, 1, 4).value();
  auto expected = Add(Ops(*native), slice, slice).value();
  auto actual = Add(Ops(*cuda), device_slice, device_slice);
  MYCV_EXPECT_TRUE(actual.ok());
  if (actual.ok()) ExpectEquivalent(expected, actual.value(), *native);

  auto nchw = native->AllocateTensor(Shape{1, 2, 2, 2}, DataType::kFloat32).value();
  for (int i = 0; i < 8; ++i) nchw.Data<float>()[i] = static_cast<float>(i);
  auto device_nchw = CopyTo(*cuda, nchw).value();
  auto expected_resize = Resize(Ops(*native), nchw, 4, 4,
                                InterpolationMode::kNearest, ImageLayout::kNCHW).value();
  auto actual_resize = Resize(Ops(*cuda), device_nchw, 4, 4,
                              InterpolationMode::kNearest, ImageLayout::kNCHW);
  MYCV_EXPECT_TRUE(actual_resize.ok());
  if (actual_resize.ok()) ExpectEquivalent(expected_resize, actual_resize.value(), *native);
}

MYCV_TEST(CudaCastPreservesIntegerAndFloat16Semantics) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto input = native->AllocateTensor(Shape{5}, DataType::kFloat32).value();
  const float values[] = {-2.0f, 0.9f, 1.9f, 300.0f, 3.1415926f};
  for (int i = 0; i < 5; ++i) input.Data<float>()[i] = values[i];
  auto device = CopyTo(*cuda, input).value();
  auto expected = Cast(Ops(*native), input, DataType::kUInt8).value();
  auto bytes = Cast(Ops(*cuda), device, DataType::kUInt8);
  MYCV_EXPECT_TRUE(bytes.ok());
  if (bytes.ok()) ExpectEquivalent(expected, bytes.value(), *native, 0.0);

  auto half = Cast(Ops(*cuda), device, DataType::kFloat16);
  MYCV_EXPECT_TRUE(half.ok());
  if (!half.ok()) return;
  auto restored = Cast(Ops(*cuda), half.value(), DataType::kFloat32);
  MYCV_EXPECT_TRUE(restored.ok());
  if (restored.ok()) ExpectEquivalent(input, restored.value(), *native, 0.001);
}

}  // namespace

MYCV_TEST_MAIN()
