#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

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

MYCV_TEST(CudaCloneReusesTheOwningContext) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto host = native->AllocateTensor(Shape{16}, DataType::kFloat32).value();
  for (int i = 0; i < 16; ++i) host.Data<float>()[i] = static_cast<float>(i);
  auto uploaded = CopyTo(*cuda, host);
  MYCV_EXPECT_TRUE(uploaded.ok());
  if (!uploaded.ok()) return;
  Tensor device = uploaded.value();

  const size_t allocations_before = cuda->memory().Stats().allocation_count;
  auto clone = device.Clone();
  MYCV_EXPECT_TRUE(clone.ok());
  if (!clone.ok()) return;
  // Clone must allocate from the tensor's own context, not a throwaway one.
  MYCV_EXPECT_EQ(cuda->memory().Stats().allocation_count, allocations_before + 1);
  MYCV_EXPECT_EQ(clone.value().backend(), BackendType::kCUDA);

  auto downloaded = CopyTo(*native, clone.value());
  MYCV_EXPECT_TRUE(downloaded.ok());
  if (downloaded.ok()) {
    for (int i = 0; i < 16; ++i) {
      MYCV_EXPECT_EQ(downloaded.value().Data<float>()[i], static_cast<float>(i));
    }
  }
}

MYCV_TEST(CudaRejectsNonContiguousDeviceCopy) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto source = cuda->AllocateTensor(Shape{4, 4}, DataType::kFloat32).value();
  auto destination = cuda->AllocateTensor(Shape{4, 2}, DataType::kFloat32).value();

  auto narrowed_result = source.Slice(1, 0, 2);
  MYCV_EXPECT_TRUE(narrowed_result.ok());
  if (!narrowed_result.ok()) return;
  auto status = cuda->Copy(&destination, narrowed_result.value());
  // Explicit refusal beats a per-element synchronous GPU copy.
  MYCV_EXPECT_EQ(status.code(), StatusCode::kUnsupported);
}

// --- End-to-end pipeline: resize -> gaussian_blur. The point of this test is
// not timing but the invariant that the whole chain executes on the GPU
// without staging through host memory.

constexpr int64_t kPipeH = 128;
constexpr int64_t kPipeW = 160;
constexpr int64_t kPipeC = 3;
constexpr int64_t kResizeH = 96;
constexpr int64_t kResizeW = 112;
constexpr int kBlurK = 3;
constexpr double kBlurSigma = 1.0;

Result<Tensor> RunImageChain(const OpContext& ops, const Tensor& input,
                             std::vector<Tensor>* stages) {
  auto resized = Resize(ops, input, kResizeH, kResizeW);
  if (!resized.ok()) return resized.status();
  if (stages != nullptr) stages->push_back(resized.value());

  auto blurred = GaussianBlur(ops, resized.value(), kBlurK, kBlurSigma);
  if (!blurred.ok()) return blurred.status();
  if (stages != nullptr) stages->push_back(blurred.value());
  return blurred;
}

// Mean absolute error normalised by the stage value range. cv::warpAffine and
// the native kernel disagree by ~1/128 px on border pixels, so a per-element
// tolerance would fail there; the mean over the stage does not.
double StageMeanRelative(const Tensor& actual_host,
                         const Tensor& reference_host) {
  if (actual_host.Numel() != reference_host.Numel() ||
      actual_host.dtype() != reference_host.dtype()) {
    return 1.0;
  }
  const int64_t count = reference_host.Numel();
  double sum_abs = 0.0;
  double range = 0.0;
  const auto accumulate = [&](double a, double e) {
    sum_abs += std::fabs(a - e);
    range = std::max(range, std::fabs(e));
  };
  if (reference_host.dtype() == DataType::kUInt8) {
    const uint8_t* a = actual_host.Data<uint8_t>();
    const uint8_t* r = reference_host.Data<uint8_t>();
    for (int64_t i = 0; i < count; ++i) accumulate(a[i], r[i]);
  } else if (reference_host.dtype() == DataType::kFloat32) {
    const float* a = actual_host.Data<float>();
    const float* r = reference_host.Data<float>();
    for (int64_t i = 0; i < count; ++i) accumulate(a[i], r[i]);
  } else {
    return 1.0;
  }
  return count > 0 ? sum_abs / static_cast<double>(count) /
                          std::max(1.0, range)
                    : 0.0;
}

MYCV_TEST(CudaPipelineNeverTouchesHost) {
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto cuda_result = Context::Create(BackendType::kCUDA);
  if (!cuda_result.ok()) {
    std::cout << "SKIP CudaPipelineNeverTouchesHost: "
              << cuda_result.status().ToString() << '\n';
    return;
  }
  auto cuda = cuda_result.value();

  auto input =
      native->AllocateTensor(Shape{kPipeH, kPipeW, kPipeC}, DataType::kUInt8)
          .value();
  uint8_t* pixels = input.Data<uint8_t>();
  for (int64_t y = 0; y < kPipeH; ++y) {
    for (int64_t x = 0; x < kPipeW; ++x) {
      for (int64_t c = 0; c < kPipeC; ++c) {
        const double v =
            64.0 + 120.0 * x / kPipeW + 60.0 * y / kPipeH + 20.0 * c;
        pixels[(y * kPipeW + x) * kPipeC + c] =
            static_cast<uint8_t>(std::min(255.0, std::max(0.0, v)));
      }
    }
  }

  // Exactly one upload before the chain. Everything below must stay on device.
  auto uploaded = CopyTo(*cuda, input);
  MYCV_EXPECT_TRUE(uploaded.ok());
  if (!uploaded.ok()) return;

  const size_t allocations_before = cuda->memory().Stats().allocation_count;

  std::vector<Tensor> device_stages;
  auto output = RunImageChain(Ops(*cuda), uploaded.value(), &device_stages);
  MYCV_EXPECT_TRUE(output.ok());
  if (!output.ok()) return;

  // Two operators -> two device allocations on this context. The blur's f64
  // intermediate also lands here but is released before the call returns, so
  // it does not show up in the steady-state count. Any hidden host staging
  // buffer (or an extra live device temporary) would fail the test.
  MYCV_EXPECT_EQ(device_stages.size(), static_cast<size_t>(2));
  MYCV_EXPECT_EQ(cuda->memory().Stats().allocation_count,
                 allocations_before + 2);

  // Every intermediate is resident on the GPU.
  for (const Tensor& stage : device_stages) {
    MYCV_EXPECT_EQ(stage.backend(), BackendType::kCUDA);
    MYCV_EXPECT_EQ(stage.device().type, DeviceType::kGPU);
    MYCV_EXPECT_EQ(stage.buffer()->kind(), MemoryKind::kDevice);
  }

  // Correctness against the native CPU chain, after a single copy-back per
  // stage. Same tolerance as the pipeline benchmark.
  std::vector<Tensor> reference_stages;
  auto reference = RunImageChain(Ops(*native), input, &reference_stages);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  MYCV_EXPECT_EQ(reference_stages.size(), device_stages.size());

  for (size_t i = 0; i < device_stages.size(); ++i) {
    auto downloaded = CopyTo(*native, device_stages[i]);
    MYCV_EXPECT_TRUE(downloaded.ok());
    if (!downloaded.ok()) return;
    MYCV_EXPECT_TRUE(StageMeanRelative(downloaded.value(), reference_stages[i]) <=
                     1e-3);
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
  auto input = cuda->AllocateTensor(Shape{2, 2}, DataType::kFloat32).value();
  auto output = Resize(Ops(*cuda, CpuKernelLevel::kSimdAvx2), input, 2, 2);
  MYCV_EXPECT_EQ(output.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(CudaOperatorsMatchNativeCpu) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto input = native->AllocateTensor(Shape{4, 5, 3}, DataType::kFloat32).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<float>()[i] = static_cast<float>(i % 17) * 0.25f;
  }
  auto device_input = CopyTo(*cuda, input).value();

  auto reference_resize = Resize(Ops(*native), input, 3, 4).value();
  auto actual_resize = Resize(Ops(*cuda), device_input, 3, 4);
  MYCV_EXPECT_TRUE(actual_resize.ok());
  if (actual_resize.ok()) ExpectEquivalent(reference_resize, actual_resize.value(), *native);

  auto reference_blur = GaussianBlur(Ops(*native), input, 3, 1.0).value();
  auto actual_blur = GaussianBlur(Ops(*cuda), device_input, 3, 1.0);
  MYCV_EXPECT_TRUE(actual_blur.ok());
  if (actual_blur.ok()) ExpectEquivalent(reference_blur, actual_blur.value(), *native);
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
  auto expected = Resize(Ops(*native), slice, 2, 3).value();
  auto actual = Resize(Ops(*cuda), device_slice, 2, 3);
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

MYCV_TEST(CudaWrapAdoptsExternalDeviceMemory) {
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto native = Context::Create(BackendType::kNativeCpu).value();

  // Build reference data on the host, then upload to a device allocation that
  // mycv did NOT make — stand-in for a pointer from another runtime
  // (cv::cuda::GpuMat, a raw cudaMalloc, an NPU buffer mapped to device, ...).
  constexpr int64_t kCount = 8;
  auto host = native->AllocateTensor(Shape{2, 4}, DataType::kFloat32).value();
  for (int64_t i = 0; i < kCount; ++i) {
    host.Data<float>()[i] = static_cast<float>(i) * 1.5f;
  }

  void* device_ptr = nullptr;
  MYCV_EXPECT_EQ(cudaMalloc(&device_ptr, host.ByteSize()), cudaSuccess);
  if (device_ptr == nullptr) return;
  MYCV_EXPECT_EQ(cudaMemcpy(device_ptr, host.data(), host.ByteSize(),
                            cudaMemcpyHostToDevice),
                 cudaSuccess);
  // The external runtime produced this memory on the default stream; make it
  // visible to mycv's non-blocking stream before the kernel reads it.
  MYCV_EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const size_t allocations_before = cuda->memory().Stats().allocation_count;
  auto freed = std::make_shared<int>(0);

  {
    auto wrapped =
        cuda->Wrap(device_ptr, host.shape(), host.dtype(), MemoryKind::kDevice,
                   [freed](void* p, size_t) {
                     cudaFree(p);
                     ++(*freed);
                   });
    MYCV_EXPECT_TRUE(wrapped.ok());
    if (!wrapped.ok()) {
      cudaFree(device_ptr);
      return;
    }
    const Tensor& t = wrapped.value();
    // Wrap aliases the external device pointer and records no manager
    // allocation — the whole point of zero-copy adoption.
    MYCV_EXPECT_EQ(t.buffer()->kind(), MemoryKind::kDevice);
    MYCV_EXPECT_EQ(t.data(), device_ptr);
    MYCV_EXPECT_EQ(cuda->memory().Stats().allocation_count, allocations_before);
    MYCV_EXPECT_EQ(*freed, 0);

    // Zero-copy device input: feed the adopted pointer straight into a GPU op.
    // Identity resize reads the borrowed device memory via the kernel.
    auto result = Resize(Ops(*cuda), t, 2, 4);
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) return;

    auto downloaded = CopyTo(*native, result.value());
    MYCV_EXPECT_TRUE(downloaded.ok());
    if (downloaded.ok()) {
      for (int64_t i = 0; i < kCount; ++i) {
        MYCV_EXPECT_EQ(downloaded.value().Data<float>()[i],
                       host.Data<float>()[i]);
      }
    }
  }
  // Last reference released -> the adopted cudaFree deleter ran exactly once,
  // and nothing the manager tracked leaked.
  MYCV_EXPECT_EQ(*freed, 1);
  MYCV_EXPECT_EQ(cuda->memory().Stats().allocation_count, allocations_before);
}

}  // namespace

MYCV_TEST_MAIN()
