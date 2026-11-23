#ifndef hpcplatform_CORE_TIMING_H_
#define hpcplatform_CORE_TIMING_H_

#include <chrono>
#include <cstdint>

namespace hpcplatform {

// Monotonic clock for wall-clock measurements. steady_clock is not affected by
// system time adjustments (NTP, manual clock changes), so it is the right
// choice for benchmarking.
using SteadyClock = std::chrono::steady_clock;

// Converts a nanosecond count to fractional milliseconds.
inline double ToMilliseconds(int64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1'000'000.0;
}

// Elapsed-time stopwatch. Starts on construction; call Reset() to begin a new
// measurement. Intended for coarse phases such as loading a config or a whole
// pipeline iteration.
class Stopwatch {
 public:
  Stopwatch() : start_(SteadyClock::now()) {}

  void Reset() { start_ = SteadyClock::now(); }

  // Elapsed time since construction or the last Reset().
  int64_t ElapsedNanoseconds() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               SteadyClock::now() - start_)
        .count();
  }

  double ElapsedMilliseconds() const {
    return ToMilliseconds(ElapsedNanoseconds());
  }

 private:
  SteadyClock::time_point start_;
};

// RAII timer that adds its scope's duration to a caller-owned accumulator, so
// per-operator timings can be summed across iterations without any allocation.
// Pass the address of a mutable int64_t; nullptr disables accumulation, which
// lets the same code run with timing switched off.
class ScopedTimer {
 public:
  explicit ScopedTimer(int64_t* accumulator_nanoseconds)
      : accumulator_(accumulator_nanoseconds), start_(SteadyClock::now()) {}

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  ~ScopedTimer() {
    if (accumulator_ != nullptr) {
      *accumulator_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           SteadyClock::now() - start_)
                           .count();
    }
  }

 private:
  int64_t* accumulator_ = nullptr;
  SteadyClock::time_point start_;
};

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_TIMING_H_
