#ifndef MYCV_CORE_CONTEXT_H_
#define MYCV_CORE_CONTEXT_H_

#include <memory>
#include <string>

#include "mycv/core/device.h"
#include "mycv/core/error.h"
#include "mycv/core/memory.h"
#include "mycv/core/shape.h"
#include "mycv/core/stream.h"
#include "mycv/core/tensor.h"

namespace mycv {

// The per-device provider interface behind every Context. Internal type: the
// definition lives in src/core/backend.h and is never exposed through the
// public API; Context::Create is the only construction path
// (docs/architecture.md §6.2).
class Backend;

// Entry point of the library: binds a backend to a device and exposes the
// unified memory manager / stream used to allocate and schedule tensors.
class Context {
 public:
  // No fallback: an unavailable backend returns kUnsupported.
  static Result<std::shared_ptr<Context>> Create(BackendType backend,
                                                 int device_index = 0);

  explicit Context(std::shared_ptr<Backend> backend)
      : backend_(std::move(backend)) {}

  // 返回绑定的后端类型。
  BackendType backend_type() const;
  // 返回当前设备（类型 + 序号）。
  Device device() const;
  // 返回后端与设备的人类可读描述，用于日志。
  std::string Describe() const;

  // Internal access to the owning backend. The Backend definition is internal
  // (src/core/backend.h), so public callers cannot use the pointee.
  const std::shared_ptr<Backend>& backend_ptr() const { return backend_; }

  // 返回该 Context 的内存管理器。
  MemoryManager& memory();
  const MemoryManager& memory() const;
  // 返回该 Context 的执行流。
  Stream& stream();

  // 等待该 Context 上所有已提交工作完成。
  Status Synchronize();

  // 在当前 Context 上分配连续张量；alignment 需为 2 的幂且是 sizeof(void*)
  // 的倍数。
  Result<Tensor> AllocateTensor(const Shape& shape, DataType dtype,
                                size_t alignment = 64) {
    return Tensor::Create(*this, shape, dtype, alignment);
  }

  // Wraps externally owned memory as a Tensor without allocating or copying.
  // `data` must hold at least shape.Numel() * DTypeSize(dtype) bytes. When
  // `deleter` is empty the Tensor borrows the memory (caller keeps it alive);
  // otherwise the deleter is invoked when the last reference dies, so memory
  // from another runtime (cv::Mat, cudaMalloc, NPU, ...) can be adopted.
  Result<Tensor> Wrap(void* data, const Shape& shape, DataType dtype,
                      MemoryKind kind = MemoryKind::kHost,
                      Buffer::Deleter deleter = nullptr);

  // Copies logical elements, including non-contiguous views. The destination
  // context selects the copy path; copies are synchronized before returning.
  Status Copy(Tensor* dst, const Tensor& src, Stream* stream = nullptr);

 private:
  std::shared_ptr<Backend> backend_;
};

}  // namespace mycv

#endif  // MYCV_CORE_CONTEXT_H_
