// Per-operator benchmark for laplacian_pyramid (docs/architecture.md §6.1/§7.2).
//
// The pyramid is a composition of gaussian_blur and resize, so its "version"
// is the CpuKernelLevel forwarded to those two ops. This sweep crosses input
// size, pyramid depth and every backend/level available in the build:
//
//   ./build-release-full/src/ops/laplacian_pyramid/laplacian_pyramid_benchmark
//
// Columns:
//   op,backend,level,dtype,levels,kernel,sigma,in_h,in_w,c,ms_per_call,mpix_per_s,speedup
// speedup is relative to the naive CPU row of the same configuration.
//
// Accelerated rows are measured before the CPU rows on purpose: several
// seconds of saturated CPU work lower the GPU clock on shared-power machines,
// which made the last CUDA configuration look ~5x slower than it is in
// isolation (verified by running the same config first/last).

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/service.h"
#include "mycv/mycv.h"

using namespace mycv;

namespace {

constexpr int kIterations = 5;
constexpr int kWarmup = 1;
constexpr int kKernelSize = 5;
constexpr double kSigma = 1.2;

struct Config {
  int64_t height;
  int64_t width;
  int64_t channels;
  int levels;
};

struct Measurement {
  double ms_per_call = 0.0;
  double mpix_per_second = 0.0;
};

struct ConfigResult {
  Config config;
  Measurement cuda;
  bool has_cuda = false;
  Measurement opencv;
  bool has_opencv = false;
  std::vector<std::pair<std::string, Measurement>> cpu;  // level name -> run
};

const char* DTypeLabel(DataType dtype) {
  return dtype == DataType::kUInt8 ? "u8" : "f32";
}

void FillInput(Tensor* tensor) {
  float* data = tensor->Data<float>();
  for (int64_t i = 0; i < tensor->Numel(); ++i) {
    data[i] = static_cast<float>((i * 37) % 251) / 251.0f;
  }
}

Measurement Measure(const OpContext& context, const Tensor& input,
                    const Config& config, bool* ok, int passes) {
  *ok = false;
  Measurement best;
  best.ms_per_call = 1e300;
  const double megapixels =
      static_cast<double>(config.height * config.width) / 1e6;
  for (int pass = 0; pass < passes; ++pass) {
    for (int i = 0; i < kWarmup; ++i) {
      auto warmup = BuildLaplacianPyramid(context, input, config.levels,
                                          kKernelSize, kSigma);
      if (!warmup.ok()) return {};
    }
    const auto start = std::chrono::steady_clock::now();
    double checksum = 0.0;
    for (int i = 0; i < kIterations; ++i) {
      auto pyramid = BuildLaplacianPyramid(context, input, config.levels,
                                           kKernelSize, kSigma);
      if (!pyramid.ok()) return {};
      for (const Tensor& level : pyramid.value().levels) {
        // Metadata only: accelerator levels live in device memory.
        checksum += static_cast<double>(level.Numel());
      }
    }
    const auto stop = std::chrono::steady_clock::now();
    *ok = true;
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            stop - start)
            .count() /
        kIterations;
    if (checksum == -1.0) std::cerr << "";
    if (ms < best.ms_per_call) best = {ms, megapixels / (ms / 1e3)};
  }
  return best;
}

void PrintRow(const char* backend, const std::string& level, DataType dtype,
              const Config& config, const Measurement& run, double baseline) {
  std::cout << "laplacian_pyramid," << backend << ',' << level << ','
            << DTypeLabel(dtype) << ',' << config.levels << ',' << kKernelSize
            << ',' << std::fixed << std::setprecision(2) << kSigma << ','
            << config.height << ',' << config.width << ',' << config.channels
            << ',' << std::fixed << std::setprecision(4) << run.ms_per_call
            << ',' << run.mpix_per_second << ','
            << (baseline > 0.0 ? baseline / run.ms_per_call : 0.0) << '\n';
}

}  // namespace

int main() {
  static auto cpu = Context::Create(BackendType::kNativeCpu);
  if (!cpu.ok()) {
    std::cerr << "FAIL: " << cpu.status().ToString() << '\n';
    return 1;
  }

  const Config configs[] = {
      {512, 512, 3, 2},
      {512, 512, 3, 3},
      {1024, 1024, 3, 2},
      {1024, 1024, 3, 3},
  };
  std::vector<ConfigResult> results;
  results.reserve(sizeof(configs) / sizeof(configs[0]));
  for (const Config& config : configs) {
    ConfigResult result;
    result.config = config;
    results.push_back(std::move(result));
  }

  std::cout << "op,backend,level,dtype,levels,kernel,sigma,in_h,in_w,c,"
               "ms_per_call,mpix_per_s,speedup\n";

  auto opencv = Context::Create(BackendType::kOpenCV);
  auto cuda = Context::Create(BackendType::kCUDA);

  // Phase 1: accelerator rows first, while the GPU clock is untouched.
  for (ConfigResult& result : results) {
    const Config& config = result.config;
    auto input =
        cpu.value()->AllocateTensor(Shape{config.height, config.width,
                                          config.channels},
                                    DataType::kFloat32);
    if (!input.ok()) continue;
    FillInput(&input.value());

    const auto measure_backend =
        [&](const std::shared_ptr<Context>& backend, int passes,
            Measurement* out) -> bool {
      if (!backend) return false;
      OpContext accelerated;
      accelerated.context = backend.get();
      accelerated.stream = &backend->stream();
      auto device =
          backend->AllocateTensor(input.value().shape(), DataType::kFloat32);
      if (!device.ok()) return false;
      if (!backend->Copy(&device.value(), input.value()).ok()) return false;
      bool ok = false;
      *out = Measure(accelerated, device.value(), config, &ok, passes);
      return ok;
    };

    // Two CUDA passes: the first batch absorbs any clock ramp.
    if (cuda.ok()) {
      result.has_cuda = measure_backend(cuda.value(), 2, &result.cuda);
    }
    if (opencv.ok()) {
      result.has_opencv = measure_backend(opencv.value(), 1, &result.opencv);
    }
  }

  // Phase 2: CPU versions, level-major so each level sees the same machine
  // state across configurations.
  for (CpuKernelLevel level :
       {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
        CpuKernelLevel::kSimdGeneric, CpuKernelLevel::kSimdAvx2,
        CpuKernelLevel::kTbb}) {
    if (!cpu.value()->backend_ptr()->SupportsKernelLevel(level)) continue;
    for (ConfigResult& result : results) {
      const Config& config = result.config;
      auto input =
          cpu.value()->AllocateTensor(Shape{config.height, config.width,
                                            config.channels},
                                      DataType::kFloat32);
      if (!input.ok()) continue;
      FillInput(&input.value());
      OpContext context;
      context.context = cpu.value().get();
      context.stream = &cpu.value()->stream();
      context.cpu_level = level;
      bool ok = false;
      const Measurement run = Measure(context, input.value(), config, &ok, 1);
      // Backend-wide capability is broader than this operator's matrix (no
      // SIMD until resize grows one), so an unavailable level is skipped.
      if (!ok) continue;
      result.cpu.emplace_back(CpuKernelLevelName(level), run);
    }
  }

  // Phase 3: print per configuration with the naive CPU row as baseline.
  for (const ConfigResult& result : results) {
    double baseline = 0.0;
    for (const auto& entry : result.cpu) {
      if (entry.first == "naive") baseline = entry.second.ms_per_call;
    }
    for (const auto& entry : result.cpu) {
      PrintRow("cpu", entry.first, DataType::kFloat32, result.config,
               entry.second, baseline);
    }
    if (result.has_opencv) {
      PrintRow("opencv", "naive", DataType::kFloat32, result.config,
               result.opencv, baseline);
    }
    if (result.has_cuda) {
      PrintRow("cuda", "naive", DataType::kFloat32, result.config, result.cuda,
               baseline);
    }
  }
  return 0;
}
