#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "mycv/mycv.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Measurement {
  mycv::CpuKernelLevel level;
  double milliseconds;
};

}  // namespace

int main() {
  using namespace mycv;
  auto context = Context::Create(BackendType::kNativeCpu).value();
  constexpr int64_t kElements = 1 << 20;
  constexpr int kIterations = 30;
  auto lhs = context->AllocateTensor(Shape{kElements}, DataType::kFloat32).value();
  auto rhs = context->AllocateTensor(Shape{kElements}, DataType::kFloat32).value();
  for (int64_t i = 0; i < kElements; ++i) {
    lhs.Data<float>()[i] = static_cast<float>(i % 101) * 0.01f;
    rhs.Data<float>()[i] = static_cast<float>(i % 17) * 0.02f;
  }

  std::cout << "op,level,elements,iterations,ms_per_iter,gib_per_s,speedup\n";
  for (const char* op_name : {"add", "mul"}) {
    std::vector<Measurement> measurements;
    for (CpuKernelLevel level : {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
                                 CpuKernelLevel::kTbb, CpuKernelLevel::kSimdGeneric,
                                 CpuKernelLevel::kSimdAvx2}) {
      if (!context->backend().SupportsKernelLevel(level)) continue;
      OpContext op_context;
      op_context.context = context.get();
      op_context.stream = &context->stream();
      op_context.cpu_level = level;
      const auto run = [&] {
        return std::string(op_name) == "add" ? Add(op_context, lhs, rhs)
                                             : Mul(op_context, lhs, rhs);
      };
      auto warmup = run();
      if (!warmup.ok()) continue;
      const auto start = Clock::now();
      float checksum = 0;
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        auto output = run();
        if (!output.ok()) return 1;
        checksum += output.value().Data<float>()[static_cast<int64_t>(iteration)];
      }
      const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start);
      measurements.push_back({level, elapsed.count() / kIterations});
      if (checksum == -1.0f) std::cerr << "unreachable\n";
    }
    const double baseline = measurements.empty() ? 1.0 : measurements.front().milliseconds;
    for (const Measurement& measurement : measurements) {
      const double seconds = measurement.milliseconds / 1000.0;
      const double gib_per_s = (3.0 * kElements * sizeof(float)) /
                               (seconds * 1024.0 * 1024.0 * 1024.0);
      std::cout << op_name << ',' << CpuKernelLevelName(measurement.level) << ','
                << kElements << ',' << kIterations << ',' << std::fixed
                << std::setprecision(4) << measurement.milliseconds << ','
                << gib_per_s << ',' << baseline / measurement.milliseconds << '\n';
    }
  }
  return 0;
}
