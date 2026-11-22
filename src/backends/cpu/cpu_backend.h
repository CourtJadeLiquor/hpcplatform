#ifndef MYCV_SRC_BACKENDS_CPU_CPU_BACKEND_H_
#define MYCV_SRC_BACKENDS_CPU_CPU_BACKEND_H_

#include <string>

#include "backends/cpu/cpu_memory_manager.h"
#include "backends/cpu/cpu_stream.h"
#include "core/backend.h"

namespace mycv {

class CpuBackend final : public Backend {
 public:
  explicit CpuBackend(int device_index)
      : device_{DeviceType::kCPU, device_index},
        memory_(device_),
        stream_(device_) {}

  BackendType type() const override { return BackendType::kNativeCpu; }
  Device device() const override { return device_; }
  MemoryManager& memory() override { return memory_; }
  // Grants mutable access to the owned stream; the backend itself is not
  // modified, so the accessor is const and casts away the member's constness.
  Stream& stream() const override { return const_cast<CpuStream&>(stream_); }
  Status Synchronize() override { return stream_.Synchronize(); }

  bool SupportsKernelLevel(CpuKernelLevel level) const override;
  std::string Describe() const override;

 private:
  Device device_;
  CpuMemoryManager memory_;
  CpuStream stream_;
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_CPU_CPU_BACKEND_H_
