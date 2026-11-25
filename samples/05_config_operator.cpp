// Config-driven operator sample: load operator objects from a YAML config, hold
// several of them, and drive the execution flow yourself. The library does NOT
// build a computation graph; it only hands back configured operator objects.
//
// It also demonstrates how to time the three phases separately:
//   1. loading    - done once, outside the hot loop (YAML parse + object build)
//   2. each Run   - accumulated per operator across iterations
//   3. pipeline   - one full iteration (both operators plus glue)
//
// Usage:
//   ./05_config_operator                 # uses the inline config below
//   ./05_config_operator path/to.yaml    # loads operators from a config file
//
// Tip: set hpcplatform_BACKEND (e.g. "native_cpu" or, with a CUDA build, "cuda") to
// override the backend for every loaded operator, ignoring the config backend.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include "hpcplatform/hpcplatform.h"

namespace
{

    constexpr int kIterations = 50;

    // Sample-local timing helpers built on <chrono>. The library deliberately
    // ships no timing utilities; application code should use its own clock.
    double NanosecondsToMilliseconds(int64_t nanoseconds)
    {
        return static_cast<double>(nanoseconds) / 1.0e6;
    }

    double AverageMilliseconds(int64_t total_nanoseconds, int iterations)
    {
        return NanosecondsToMilliseconds(total_nanoseconds) / iterations;
    }

    // RAII timer that adds its scope's duration to a caller-owned accumulator, so
    // per-operator timings can be summed across iterations without any allocation.
    class ScopedTimer
    {
    public:
        explicit ScopedTimer(int64_t* accumulator_nanoseconds)
            : accumulator_(accumulator_nanoseconds), start_(std::chrono::steady_clock::now())
        {
        }

        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;

        ~ScopedTimer()
        {
            if (accumulator_ != nullptr)
            {
                *accumulator_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start_)
                                     .count();
            }
        }

    private:
        int64_t* accumulator_ = nullptr;
        std::chrono::steady_clock::time_point start_;
    };

    // A self-contained config so the sample runs with no external file dependency.
    constexpr const char* kInlineConfig = R"(
operators:
  - name: downscale
    op: resize
    backend: native_cpu
    cpu_level: naive
    params:
      output_height: 2
      output_width: 2
      interpolation: bilinear
      layout: nhwc
  - name: smooth
    op: gaussian_blur
    backend: native_cpu
    params:
      kernel_size: 3
      sigma: 1.0
      layout: nhwc
)";

    void PrintOperator(const hpcplatform::Operator& op)
    {
        std::cout << "  - " << op.name() << "  kind=" << hpcplatform::OpKindName(op.kind())
                  << "  backend=" << hpcplatform::BackendTypeName(op.backend())
                  << "  cpu_level=" << hpcplatform::CpuKernelLevelName(op.cpu_level()) << '\n';
    }

}  // namespace

int main(int argc, char** argv)
{
    using namespace hpcplatform;

    // Load once. Parsing the config and building the objects is a one-time cost:
    // keep it out of the hot loop and measure it on its own.
    const auto load_start = std::chrono::steady_clock::now();
    Result<OperatorMap> loaded =
        (argc > 1) ? LoadOperators(argv[1]) : LoadOperatorsFromString(kInlineConfig);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start)
            .count();
    if (!loaded.ok())
    {
        std::cerr << "LoadOperators failed: " << loaded.status().ToString() << '\n';
        return 1;
    }
    OperatorMap operators = std::move(loaded.value());

    std::cout << "loaded " << operators.size() << " operator object(s):\n";
    for (const auto& [name, op] : operators)
    {
        (void)name;
        PrintOperator(*op);
    }

    // Fetch typed handles. GetOperator returns nullptr on a missing name or a
    // concrete-type mismatch, so the caller can decide what to do.
    auto* downscale = GetOperator<ResizeOperator>(operators, "downscale");
    auto* smooth = GetOperator<GaussianBlurOperator>(operators, "smooth");
    if (downscale == nullptr || smooth == nullptr)
    {
        std::cerr << "expected operators 'downscale' and 'smooth' are missing\n";
        return 1;
    }

    // Prepare an input on the CPU. In a real pipeline this would come from a
    // decoder / previous stage; the operator objects are reusable across calls.
    auto context = Context::Create(BackendType::kNativeCpu).value();
    auto input = Tensor::Empty(*context, Shape{1, 4, 4, 3}, DataType::kFloat32).value();
    for (int64_t i = 0; i < input.Numel(); ++i)
    {
        input.Data<float>()[i] = static_cast<float>(i % 5);
    }

    // Warm up: the first Run may allocate lazily (and, on accelerators, trigger
    // initialization). Synchronize so the loop below measures steady state. The
    // resulting shapes are printed here, outside the timing loop.
    {
        auto warm_resized = downscale->Run(input);
        if (!warm_resized.ok())
        {
            std::cerr << "resize failed: " << warm_resized.status().ToString() << '\n';
            return 1;
        }
        auto warm_blurred = smooth->Run(warm_resized.value());
        if (!warm_blurred.ok())
        {
            std::cerr << "gaussian_blur failed: " << warm_blurred.status().ToString() << '\n';
            return 1;
        }
        (void)downscale->context()->Synchronize();
        std::cout << "input   shape: " << input.shape().ToString() << '\n';
        std::cout << "resized shape: " << warm_resized.value().shape().ToString() << '\n';
        std::cout << "blurred shape: " << warm_blurred.value().shape().ToString() << '\n';
    }

    // Steady-state loop: accumulate wall time per phase. ScopedTimer adds each
    // scope's duration to the counter it is given.
    int64_t resize_ns = 0;
    int64_t blur_ns = 0;
    int64_t pipeline_ns = 0;
    for (int i = 0; i < kIterations; ++i)
    {
        ScopedTimer pipeline_timer(&pipeline_ns);

        Tensor resized;
        {
            ScopedTimer resize_timer(&resize_ns);
            auto result = downscale->Run(input);
            if (!result.ok())
            {
                std::cerr << "resize failed: " << result.status().ToString() << '\n';
                return 1;
            }
            resized = result.value();
            // On an asynchronous backend, wait for the device before the timer stops;
            // the CPU backend returns synchronously.
            (void)downscale->context()->Synchronize();
        }

        Tensor blurred;
        {
            ScopedTimer blur_timer(&blur_ns);
            auto result = smooth->Run(resized);
            if (!result.ok())
            {
                std::cerr << "gaussian_blur failed: " << result.status().ToString() << '\n';
                return 1;
            }
            blurred = result.value();
            (void)smooth->context()->Synchronize();
        }
        (void)blurred;
    }

    std::cout << "\n--- timing (average over " << kIterations << " iterations) ---\n";
    std::cout << "load (one-time) : " << load_ms << " ms\n";
    std::cout << "resize / run    : " << AverageMilliseconds(resize_ns, kIterations) << " ms\n";
    std::cout << "blur / run      : " << AverageMilliseconds(blur_ns, kIterations) << " ms\n";
    std::cout << "pipeline / iter : " << AverageMilliseconds(pipeline_ns, kIterations) << " ms\n";
    return 0;
}
