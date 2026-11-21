#ifndef MYCV_CORE_MEMORY_H_
#define MYCV_CORE_MEMORY_H_

#include <cstddef>
#include <functional>
#include <memory>

#include "mycv/core/device.h"
#include "mycv/core/error.h"

namespace mycv {

class Stream;

// 内存类别，决定分配器与拷贝路径。
enum class MemoryKind : uint8_t {
  kHost = 0,        // 普通可分页主机内存
  kHostPinned = 1,  // 页锁定主机内存（适合 DMA）
  kDevice = 2,      // 设备内存（GPU / NPU）
};

// 某一时刻的内存统计快照。
struct MemoryStats {
  size_t allocated_bytes = 0;   // 当前活动字节数
  size_t peak_bytes = 0;        // 历史峰值字节数
  size_t allocation_count = 0;  // 当前活动分配块数
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

  // 已分配内存的裸指针。
  void* data() { return data_; }
  const void* data() const { return data_; }
  // 分配的字节数。
  size_t size() const { return bytes_; }
  // 内存所属设备。
  Device device() const { return device_; }
  // 内存类别。
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

  // 返回管理器所属设备。
  virtual Device device() const = 0;

  // alignment must be a power of two and a multiple of sizeof(void*).
  virtual Result<std::shared_ptr<Buffer>> Allocate(size_t bytes,
                                                   size_t alignment = 64) = 0;

  // Synchronous copy between two buffers owned by this manager.
  // Cross-device transfers are resolved by Context::Copy.
  virtual Status Copy(Buffer* dst, size_t dst_offset, const Buffer* src,
                      size_t src_offset, size_t bytes,
                      Stream* stream = nullptr) = 0;

  // 返回当前活动内存、峰值与分配块数。
  virtual MemoryStats Stats() const = 0;
};

}  // namespace mycv

#endif  // MYCV_CORE_MEMORY_H_
