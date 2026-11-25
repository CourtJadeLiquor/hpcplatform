# CUDA 外部内存可见性：Wrap 进来的数据为什么是错的

> 相关代码：`src/backends/cuda/cuda_stream.cpp`、`include/hpcplatform/core/context.h`（`Wrap`）、`tests/consistency/test_cuda_backend.cpp`（`CudaWrapAdoptsExternalDeviceMemory`）
> 这是本项目真实踩过并写进测试的坑。如果你要把 `cudaMalloc` / `cv::cuda::GpuMat` / 其他运行时的 device 指针喂给 hpcplatform，必读。

---

## 1. 现象

```cpp
void* dptr;
cudaMalloc(&dptr, bytes);
cudaMemcpy(dptr, host_ptr, bytes, cudaMemcpyHostToDevice);  // 默认流

auto t = internal::TensorFactory::WrapExternal(
    *cuda_ctx, dptr, shape, dtype, MemoryKind::kDevice,
    [](void* p, size_t) { cudaFree(p); });
auto out = Add(OpContext(*cuda_ctx), t, t);   // 读出来的是垃圾！
```

`WrapExternal` 成功、指针也对（`t.data() == dptr`）、`MemoryStats` 也没多分配——但 `Add` 的结果是未初始化的垃圾值。

---

## 2. 根因：non-blocking 流与默认流没有隐式顺序

hpcplatform 的 CUDA 流创建时带 `cudaStreamNonBlocking` 标志（`src/backends/cuda/cuda_stream.cpp`）。这意味着：

- hpcplatform 的流与 CUDA 默认流（stream 0）**不隐式同步**；
- 你在默认流上 `cudaMemcpy` 产生的数据，和 hpcplatform 流上 launch 的内核之间，**没有 CUDA 自动建立的 happens-before 关系**。

虽然 `cudaMemcpy`（同步版本）会阻塞 CPU 直到拷贝完成，但那只是 **host 侧**的等待；在 GPU 的**流序**层面，默认流的写与 non-blocking 流的读之间没有强制顺序。在某些系统/时序下可能碰巧对，但**不保证、不可移植**——在本项目的验证机上它就是错的。

> 对比：如果数据是通过 `Upload`（H2D）产生的，它用的就是 hpcplatform 自己的流，且返回前同步——同一条流内的操作天然有序，所以不需要额外处理。问题只出在"外部流产生的数据"。

---

## 3. 责任在谁：调用方

库**不会**替你同步外部内存，因为它根本不知道外部数据是在哪条流上产生的（可能是默认流，可能是另一个库的流，可能是图形 API 的队列）。这是调用方的责任：

> 把外部 device 内存交给 `TensorFactory::WrapExternal` / 算子之前，必须先确保它对 hpcplatform 的流可见。

---

## 4. 两种正确做法

### 做法 A：全局设备同步（最简单）

```cpp
cudaMemcpy(dptr, host_ptr, bytes, cudaMemcpyHostToDevice);
cudaDeviceSynchronize();          // 等所有流上的所有操作完成
auto t = internal::TensorFactory::WrapExternal(*cuda_ctx, dptr, ...);
auto out = Add(OpContext(*cuda_ctx), t, t);   // 正确
```

`cudaDeviceSynchronize()` 等整个设备空闲，保证数据可见。简单但重（会等无关工作）。适合功能正确性优先的场景。

### 做法 B：事件等待（更精确，只同步这条依赖）

```cpp
cudaEvent_t ready;
cudaEventCreate(&ready);
cudaEventRecord(ready, producer_stream);   // 在产生数据的流上记录事件
cudaStreamWaitEvent(hpcplatform_raw_stream, ready, 0);  // hpcplatform 流等该事件
// 然后 Wrap + 调用算子
```

只让 hpcplatform 流等这一次外部写，不阻塞其他流。需要拿到 hpcplatform 流的原生句柄（当前未直接暴露，可通过后端扩展）。

---

## 5. 测试如何守护这个契约

`tests/consistency/test_cuda_backend.cpp` 中的 `CudaWrapAdoptsExternalDeviceMemory`：

1. 用裸 `cudaMalloc` + `cudaMemcpy`（默认流）准备外部 device 数据；
2. 调用 `cudaDeviceSynchronize()` 建立可见性；
3. `Wrap` 收养该指针（deleter 为 `cudaFree`）；
4. 断言 `MemoryStats` 没多分配、`buffer()->kind() == kDevice`；
5. 喂给 GPU `Add`，拷回 host 验证结果是 `2×` 输入；
6. 作用域结束后断言 deleter（`cudaFree`）恰好触发一次。

这同时覆盖了：零拷贝输入路径正确、外部内存可见性契约、收养语义的释放。无 GPU 时该测试 SKIP，不影响 `ctest` 全绿。

---

## 6. 与"寿命"的区别（别搞混）

这是两个独立的保证，都要满足：

| 维度 | 保证什么 | 谁负责 |
| --- | --- | --- |
| **寿命** | 指针在 Tensor 活着时不被释放 | deleter（收养）或调用方（借用） |
| **可见性** | 数据在流序上对内核可读 | 调用方同步外部流 |

寿命对了但可见性不对 → 读到垃圾（本文）；可见性对了但寿命不对 → 读到已释放内存（use-after-free）。两者都要。

---

## 7. 一句话总结

> hpcplatform 的 CUDA 流是 non-blocking 的，和默认流不隐式同步。外部流产生的 device 内存在 `Wrap`/喂算子前，调用方必须先 `cudaDeviceSynchronize()` 或用事件建立顺序——库不替你做，因为它不知道外部流的存在。
