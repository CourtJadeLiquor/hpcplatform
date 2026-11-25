#ifndef hpcplatform_CORE_MEMORY_H_
#define hpcplatform_CORE_MEMORY_H_

#include <cstddef>
#include <functional>
#include <memory>

#include "hpcplatform/core/device.h"
#include "hpcplatform/core/error.h"

namespace hpcplatform {

class Stream;

// Memory category; determines the allocator and copy path.
enum class MemoryKind : uint8_t {
  kHost = 0,        // Standard pageable host memory
  kHostPinned = 1,  // Pinned host memory (suitable for DMA)
  kDevice = 2,      // Device memory (GPU / NPU)
};

// Snapshot of memory statistics at a point in time.
struct MemoryStats {
  size_t allocated_bytes = 0;   // Bytes in current live allocations
  size_t peak_bytes = 0;        // Historical peak bytes
  size_t allocation_count = 0;  // Number of live allocations
};

// Opaque, device-agnostic allocation handle. Memory is released through the
// deleter supplied by the owning MemoryManager, which lets each backend use its
// own allocator (aligned malloc, cv::Mat, cudaMalloc, caching pool, ...).
class Buffer {
 public:
  using Deleter = std::function<void(void* data, size_t bytes)>;

  Buffer(void* data, size_t bytes, Device device, MemoryKind kind,
         Deleter deleter)
      : data_(data),
        bytes_(bytes),
        device_(device),
        kind_(kind),
        deleter_(std::move(deleter)) {}

  ~Buffer() {
    if (deleter_ && data_ != nullptr) deleter_(data_, bytes_);
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  // Raw pointer to the allocated memory.
  void* data() { return data_; }
  const void* data() const { return data_; }
  // Number of allocated bytes.
  size_t size() const { return bytes_; }
  // Device that owns the memory.
  Device device() const { return device_; }
  // Memory category.
  MemoryKind kind() const { return kind_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
  Device device_;
  MemoryKind kind_ = MemoryKind::kHost;
  Deleter deleter_;
};

// Unified allocation / copy interface. Every backend provides its own
// implementation; nothing is auto-selected at runtime.
class MemoryManager {
 public:
  virtual ~MemoryManager() = default;

  // Returns the manager's device.
  virtual Device device() const = 0;

  // alignment must be a power of two and a multiple of sizeof(void*).
  virtual Result<std::shared_ptr<Buffer>> Allocate(size_t bytes,
                                                   size_t alignment = 64) = 0;

  // Synchronous copy between two buffers owned by this manager.
  // Cross-device transfers are resolved by the transfer functions in
  // hpcplatform/core/transfer.h.
  virtual Status Copy(Buffer* dst, size_t dst_offset, const Buffer* src,
                      size_t src_offset, size_t bytes,
                      Stream* stream = nullptr) = 0;

  // Returns current usage, peak usage, and live allocation count.
  virtual MemoryStats Stats() const = 0;
};

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_MEMORY_H_
