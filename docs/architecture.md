# mycv 架构设计

***

## 1. 设计原则

1. **显式优先于隐式**：不做自动探测、自动回退、静默转换；后端、内核级别、布局都由调用方显式选择。
2. **可复现**：同一算子的 CPU 各版本必须位精确；只有跨后端才允许有限容差。
3. **描述与存储分离**：`Tensor` 只是元数据（shape/strides/offset），内存由 `Buffer`/`MemoryManager` 管理。
4. **后端可插拔、核心零第三方依赖**：`core` 不依赖 OpenCV/CUDA/TBB；可选能力全部由编译期开关门控。
5. **无异常**：在 `-fno-exceptions` 约束下使用 `Status`/`Result` 错误模型。
6. **错误码三分类**：参数错误 `kInvalidArgument`、编译期未启用 `kUnsupported`、注册表缺内核 `kNotImplemented`，绝不静默降级。
7. **开闭原则**：新增算子以"新增文件夹 + 一行式登记"为主，尽量不改动算子间共享的实现文件。
8. **算子自治**：每个算子文件夹是完备解决方案，包含入口、全部版本实现、构建、测试、基准与说明文档。

***

## 2. 总体架构

### 2.1 分层

```
                 include/mycv/**                ← 唯一公共 API 面（调用者只见这里）
                        ↑
                     src/core/                  ← Tensor/Shape/Buffer/MemoryManager/Stream/Context
                        ↑                       ← 内部契约：Backend 接口、Copy 语义
                  src/framework/                ← 注册表、图像工具、dtype 分发、剖析与统计
                   ↑         ↑
        src/backends/       src/ops/
        （能力服务）          （算子自治，每算子一个文件夹）
```

算子自己的后端实现文件（`*_cuda.cu`、`*_opencv.cpp`）位于算子文件夹内，按需引用对应后端的服务头。

### 2.2 依赖规则

- **R1** 公共头（`include/mycv/**`）不得包含任何 `src/**`。可选能力用 `#ifdef MYCV_WITH_*` 门控。
- **R2** `src/core/**` 不得包含 `framework`、`backends`、`ops`。`Backend` 作为 core 的内部契约定义在 `src/core/backend.h`。
- **R3** `src/framework/**` 依赖 `core` 与公共头中的稳定类型（`ops/op.h`、`ops/image.h`），不得包含 `backends` 或任何具体算子实现。
- **R4** `src/backends/**` 只依赖 `core`/`framework`，不得包含 `src/ops/**`。
- **R5** `src/ops/<op>/` 的入口与 CPU 实现只依赖公共头、`core`、`framework`、`ops/common` 与本文件夹内部头；只有 `*_cuda.cu` / `*_opencv.cpp` 等后端实现文件可以包含对应后端的服务头。
- **R6** 依赖单向无环。高层策略（算子）依赖低层能力（后端服务），后端永远不知道具体算子的存在。

***

## 3. 目录结构

```
include/mycv/
  core/                  # shape / dtype / error / device / memory / stream / tensor / tensor_iterator / context
  ops/                   # image.h / op.h / resize.h / gaussian_blur.h / laplacian_pyramid.h / fourier.h / wavelet.h
  interop/opencv.h       # cv::Mat 与 Tensor 的数据适配器
  mycv.h                 # 伞形头

src/
  core/                  # 数据结构与内部契约
    backend.h            # Backend 接口（core 内部契约）
    copy.cpp             # Backend::CopyBuffer / CopyTensor / Context::Copy
    copy_internal.h      # CopyTensor / ResolveOwningBackend 声明
  framework/             # 框架工具层
    op_registry.{h,cpp}  # 三元组内核注册表
    image_utils.h        # 图像布局解析与索引计算
    dtype_dispatch.h     # u8/f32/f64 分发
    compiler.h           # 编译器暗示宏
    profiling.h          # 可选算子计时宏（默认零开销）
    stats.h              # 按 MemoryKind 聚合的内存快照
  backends/              # 能力服务层，不含算子实现
    backend_internal.h   # 各后端创建函数与 CUDA 拷贝辅助
    factory.cpp          # CreateBackend
    cpu/                 # CPU 内存管理器与流
    cuda/                # CUDA 内存管理器、流、设备辅助
    opencv/              # OpenCV 张量访问适配
    npu/                 # 占位后端
  ops/
    register_ops.cpp     # 唯一的算子注册清单
    CMakeLists.txt       # 每算子一行 add_subdirectory
    common/op_common.h   # ValidateInvocation / AllocateOutput
    resize/              # 自治算子文件夹
    gaussian_blur/
    laplacian_pyramid/
    fourier/
    wavelet/

tests/
  test_framework.h       # 零依赖测试框架
  core/                  # shape / tensor / views / memory / backend / copy
  backends/              # CUDA / OpenCV 后端级测试
  perf/                  # 算子性能评估（按目标开启 profiling）
samples/                 # 公共 API 使用样例
```

***

## 4. 核心数据模型

### 4.1 Tensor：描述符

`Tensor` 持有 `shape`、`dtype`、`strides`、`offset`、`Buffer` 与弱引用的所属后端。创建方式：

- `Tensor::Create(backend, shape, dtype)`：由后端分配连续内存。
- `Tensor::Wrap(backend, data, shape, dtype, kind, deleter)`：接管外部内存，不分配、不拷贝；`deleter` 为空表示借用。

`Slice` / `Reshape` / `Transpose` 等视图只改元数据，共享同一个 `Buffer`。

### 4.2 Buffer 与 MemoryManager

`Buffer` 是引用计数的内存块，记录字节数、设备、`MemoryKind` 与可选 `deleter`。`MemoryManager` 负责按后端分配/释放，并暴露 `Stats()`（活动字节、峰值、分配次数）。CPU 使用缓存分配器；设备内存由对应后端管理。

### 4.3 Context 与 Backend

`Context` 是用户可见的门面：创建设备上下文、分配张量、包装外部内存、同步拷贝。它持有 `shared_ptr<Backend>`，但公共头只前向声明 `Backend`，用户无法接触后端能力。

`Backend` 是 core 的内部契约，职责限定为：内存管理、流、`CopyBuffer`、后端 API 封装与数据转换。后端不实现具体算子。

***

## 5. 后端服务层

`src/backends/` 是能力提供者，不是算子容器：

| 服务           | 位置                                                 | 使用者                |
| ------------ | -------------------------------------------------- | ------------------ |
| 后端接口与拷贝默认实现  | `src/core/backend.h`、`src/core/copy.cpp`           | `Context`、`Tensor` |
| 后端创建         | `src/backends/factory.cpp`                         | `Context::Create`  |
| CUDA 设备/启动辅助 | `src/backends/cuda/cuda_device_utils.h`            | 各算子 `*_cuda.cu`    |
| OpenCV 张量访问  | `src/backends/opencv/opencv_tensor_access.{h,cpp}` | 各算子 `*_opencv.cpp` |
| 图像布局与索引      | `src/framework/image_utils.h`                      | 图像算子所有版本           |

设计要点：

- 后端只暴露窄接口与稳定的服务头，算子依赖服务而非实现细节（依赖倒置）。
- 非 CPU 后端只接受 `kNaive` 级别的内核；`CopyBuffer` 不做主机中转。
- `BackendType`/`Device` 保持公共，因为 `Context::Create` 需要；但它们只是枚举/值类型，不暴露能力。
- `interop/opencv.h` 的 `CopyFromCvMat`/`CopyToCvMat` 属于对外数据适配器，不是后端能力，因此保留在公共 API。

***

## 6. 算子自治结构

### 6.1 文件夹契约

每个算子文件夹是一个完备解决方案：

```
src/ops/<op>/
  <op>.cpp                 # 入口：校验 → 查注册表 → 分配输出 → 调内核
  <op>_internal.h          # 参数结构体、内核原型、注册声明
  <op>_cpu.h               # CPU 共享骨架（可选）
  <op>_cpu_naive.cpp       # 参考实现
  <op>_cpu_omp.cpp         # MYCV_WITH_OPENMP
  <op>_cpu_tbb.cpp         # MYCV_WITH_TBB
  <op>_cpu_simd.cpp        # MYCV_ENABLE_SIMD
  <op>_cpu_avx2.cpp        # MYCV_COMPILER_HAS_AVX2（独立 OBJECT 库）
  <op>_cuda.cu             # MYCV_WITH_CUDA
  <op>_opencv.cpp          # MYCV_WITH_OPENCV
  CMakeLists.txt           # 逐文件宏门控 + 挂测试与基准
  tests/                   # 正确性与位精确对拍
  benchmark/               # 各版本耗时
  README.md                # 版本矩阵、容差约定、基准数据
```

契约要点：

- **文件唯一归属**：算子文件夹内的文件不被其他算子触碰。
- **版本矩阵**：README 维护 `(后端, CpuKernelLevel)` 与实现文件、对拍方式的对应表。
- **参数结构体归位**：`ResizeParams`、`GaussianBlurParams` 等定义在各自 `*_internal.h`。
- **入口骨架固定**：校验 → 查注册表 → 分配输出 → 调内核；公共头只声明 API。
- **版本不齐不是错误**：某组合没有实现时，注册表返回 `kNotImplemented`，`SupportsKernelLevel` 返回 false。
- **AVX2 构建**：AVX2 翻译单元编成独立 OBJECT 库（`-mavx2` + `MYCV_ARCH_AVX2`），再嵌入 `mycv`，避免污染整个静态库的可移植性。

### 6.2 内核纪律

- CPU 各版本之间位精确（`memcmp` 对拍 naive 基准）。
- 跨后端使用 `1e-4` 相对容差；u8 规则见 [`qa/testing.md`](./qa/testing.md)。
- 归约型算子（如高斯）必须保持与 naive 一致的累加顺序，SIMD/并行只能改变任务划分，不能改变单像素求和顺序。
- `CopyBuffer` 的同步语义、非 CPU 后端级别限制、Context 生命周期规则见 [`qa/context_copy.md`](./qa/context_copy.md)。

***

## 7. 注册与分发

### 7.1 三元组键

内核注册键为：

```
(OpKind, BackendType, CpuKernelLevel) → OpKernelFn
```

`FindKernel` 对未命中严格区分三种情况：参数不合法、后端未编译、内核未实现，分别映射到三个错误码。

### 7.2 显式注册清单

`src/ops/register_ops.cpp` 是唯一的注册入口：

```cpp
void RegisterAllKernels() {
  RegisterResizeKernels();
  RegisterGaussianBlurKernels();
  RegisterLaplacianPyramidKernels();
  RegisterFourierKernels();
  RegisterWaveletKernels();
}
```

每个算子的 `Register<Op>Kernels()` 定义在本算子文件夹内，一次性注册它的全部版本（CPU 各级 + CUDA + OpenCV）。

**不使用静态自注册**：静态库中未被引用的翻译单元可能根本不会被链接器拉入，自注册对象的构造函数可能不执行，导致内核静默缺失；补 `--whole-archive` 等链接器魔法与"显式优先"原则冲突。显式清单多一行登记，换来行为完全可预期。

***

## 8. 框架工具层

### 8.1 归置规则

- **逐像素热路径辅助**：`inline` 定义在 `src/framework/*.h`，随使用方内联。例如 `ImageByteOffset`。
- **一次性冷路径函数**：声明在头文件，定义落 `.cpp`，改实现只重编一个 TU。当前 `image_utils.h` 仍整体 header-only，新增冷路径函数时按此规则拆分。
- 不为了"统一"把热函数改成跨 TU 调用，除非有测量结果支撑。

### 8.2 剖析与统计

- `profiling.h`：`MYCV_PROFILE_SCOPE` / `MYCV_PROFILE_OP`，默认展开为空语句；只对需要剖析的目标定义 `MYCV_ENABLE_PROFILING=1`。
- `stats.h`：`SnapshotMemory(Context&)` 按 `MemoryKind` 聚合活动字节、峰值与分配次数，用于峰值和泄漏检查。
- 算子 `benchmark/` 直接使用 `std::chrono::steady_clock`，不强依赖剖析宏。
- 性能评估用例见 `tests/perf/test_operator_perf.cpp`。

### 8.3 dtype 分发

`dtype_dispatch.h` 提供 `DispatchImageDType`（u8/f32/f64）。类型分支集中在入口，内核实现保持单一类型模板，避免在每个版本里重复 switch。

***

## 9. 构建与开关

- 可选后端：`MYCV_WITH_CUDA`、`MYCV_WITH_OPENCV`、`MYCV_WITH_TBB`、`MYCV_WITH_OPENMP`、`MYCV_WITH_SIMD_GENERIC`。
- 运行时开关：`MYCV_ENABLE_SIMD`、`MYCV_ENABLE_PROFILING`。
- 各算子通过自己的 `CMakeLists.txt` 逐文件门控；未启用时对应翻译单元不参与编译。
- 推荐至少维护三棵构建树：默认（无可选后端）、全开、ASan。

***

## 10. 扩展指南

**新增算子 X**：

1. `include/mycv/ops/x.h` —— 公共 API。
2. `include/mycv/ops/op.h` —— `OpKind` 追加枚举值（append-only）。
3. `src/ops/x/` —— 自治文件夹（入口、内部头、各版本实现、CMake、tests、benchmark、README）。
4. `src/ops/register_ops.cpp` —— 加一行 `RegisterXKernels();`。
5. `src/ops/CMakeLists.txt` —— 加一行 `add_subdirectory(x)`。
6. `include/mycv/mycv.h` —— 加一行 include。

**给已有算子加后端实现**：算子文件夹加一个实现文件 + `Register<Op>Kernels()` 加一行宏门控注册 + 算子 CMake 加一行。core/backends/其他算子零改动。

**新增后端 B**：`src/backends/b/`（内存管理器 + 流）+ `src/backends/factory.cpp` 加一个 `case` + 构建开关。各算子自愿增量添加 `*_b.cpp`，不添加也不破坏现有构建。

***

## 11. 关键设计模式

| 模式              | 落点                                        | 作用                          |
| --------------- | ----------------------------------------- | --------------------------- |
| Strategy        | 每算子多版本内核 + `CpuKernelLevel`               | 同一语义的不同实现可替换                |
| Factory         | `CreateBackend`                           | 屏蔽后端构造细节                    |
| Registry        | `op_registry.{h,cpp}`                     | 三元组到内核函数的延迟绑定               |
| Facade          | `Context`                                 | 对调用者隐藏后端与内存管理               |
| Adapter         | `interop/opencv.h`、`opencv_tensor_access` | 外部 `cv::Mat` 与 `Tensor` 互操作 |
| Template Method | `<op>_cpu.h` 共享骨架                         | 固定遍历/边界/累加骨架，版本只填热点         |
| RAII            | `Buffer`、`Result`、`Stream`                | 资源与错误生命周期自动收敛               |
| 编译期多态           | `MYCV_WITH_*` / `MYCV_ENABLE_*`           | 零运行时开销的可选能力                 |

***

## 12. UML 类图

类图使用 Mermaid `classDiagram`（Markdown 原生渲染，无需额外运行时）。按模块拆分为三张图，避免单图过大。

### 12.1 core 数据模型

```mermaid
classDiagram
  direction LR

  class Shape {
    -int rank_
    -int64_t[] dims_
    +Rank() int
    +Dim(index) int64_t
    +Numel() int64_t
    +IsScalar() bool
    +Validate() Status
    +ToString() string
  }

  class Device {
    +DeviceType type
    +int index
    +ToString() string
  }

  class Buffer {
    -void* data_
    -size_t bytes_
    -Device device_
    -MemoryKind kind_
    -Deleter deleter_
    +data() void*
    +size() size_t
    +device() Device
    +kind() MemoryKind
  }

  class MemoryManager {
    <<abstract>>
    +device() Device
    +Allocate(bytes, alignment) Result
    +Copy(dst, src, stream) Status
    +Stats() MemoryStats
  }

  class Stream {
    <<abstract>>
    +device() Device
    +Synchronize() Status
    +synchronous() bool
  }

  class Backend {
    <<abstract>>
    +type() BackendType
    +device() Device
    +memory() MemoryManager
    +stream() Stream
    +Synchronize() Status
    +CopyBuffer(dst, src, bytes, stream) Status
    +SupportsDType(dtype) bool
    +SupportsKernelLevel(level) bool
    +Describe() string
  }

  class Context {
    -shared_ptr~Backend~ backend_
    +Create(backend, deviceIndex) Result
    +AllocateTensor(shape, dtype) Result
    +Wrap(data, shape, dtype) Result
    +Copy(dst, src, stream) Status
    +memory() MemoryManager
    +stream() Stream
    +Synchronize() Status
  }

  class Tensor {
    -shared_ptr~Buffer~ buffer_
    -weak_ptr~Backend~ owner_
    -Shape shape_
    -DataType dtype_
    -int64_t[] strides_
    -size_t offset_
    +Create(backend, shape, dtype) Result
    +Wrap(backend, data, shape, dtype) Result
    +View(shape) Result
    +Slice(dim, begin, end) Result
    +Permute(order) Result
    +Reshape(shape) Result
    +Clone() Result
    +IsContiguous() bool
  }

  class Status {
    -StatusCode code_
    -string message_
    +ok() bool
    +code() StatusCode
    +message() string
  }

  class Result {
    <<template T>>
    -bool ok_
    -T value_
    -Status status_
    +ok() bool
    +value() T
    +status() Status
  }

  class MemoryStats {
    +size_t allocated_bytes
    +size_t peak_bytes
    +size_t allocation_count
  }

  class BackendType {
    <<enumeration>>
    kNativeCpu
    kOpenCV
    kCUDA
    kNPU
  }

  class DataType {
    <<enumeration>>
    kUInt8
    kFloat32
    kFloat64
  }

  class MemoryKind {
    <<enumeration>>
    kHost
    kHostPinned
    kDevice
  }

  Tensor *-- Buffer : owns
  Tensor --> Shape
  Tensor --> Device
  Tensor o-- Backend : weak owner
  Context o-- Backend : binds
  Context --> MemoryManager : memory()
  Context --> Stream : stream()
  Backend o-- MemoryManager
  Backend o-- Stream
  MemoryManager ..> Buffer : allocates
  MemoryManager ..> MemoryStats : reports
  Buffer --> Device
  Buffer --> MemoryKind
  Result ..> Status : carries
  Context ..> Result : returns
  Tensor ..> Result : returns
```

### 12.2 后端实现

```mermaid
classDiagram
  direction TB

  class Backend {
    <<abstract>>
    +type() BackendType
    +device() Device
    +memory() MemoryManager
    +stream() Stream
    +CopyBuffer(dst, src, bytes, stream) Status
    +SupportsKernelLevel(level) bool
  }

  class MemoryManager {
    <<abstract>>
    +Allocate(bytes, alignment) Result
    +Copy(dst, src, stream) Status
    +Stats() MemoryStats
  }

  class Stream {
    <<abstract>>
    +Synchronize() Status
    +synchronous() bool
  }

  class CpuBackend {
    -Device device_
    -CpuMemoryManager memory_
    -CpuStream stream_
    +SupportsKernelLevel(level) bool
    +Describe() string
  }

  class CudaBackend {
    -Device device_
    -CudaMemoryManager memory_
    -CudaStream stream_
    -string device_name_
    +Initialize() Status
    +CopyBuffer(dst, src, bytes, stream) Status
    +SupportsKernelLevel(level) bool
  }

  class OpenCvBackend {
    -Device device_
    -OpenCvMemoryManager memory_
    -CpuStream stream_
    +SupportsDType(dtype) bool
    +SupportsKernelLevel(level) bool
  }

  class NpuBackend {
    <<placeholder>>
    +CreateNpuBackend(index) Result
  }

  class CpuMemoryManager {
    -Device device_
    -shared_ptr~CpuAllocatorState~ state_
    +Allocate(bytes, alignment) Result
    +Copy(dst, src, stream) Status
    +Stats() MemoryStats
  }

  class CudaMemoryManager {
    -Device device_
    -shared_ptr~CudaAllocatorState~ state_
    +Allocate(bytes, alignment) Result
    +Copy(dst, src, stream) Status
    +Stats() MemoryStats
  }

  class OpenCvMemoryManager {
    -Device device_
    -shared_ptr~OpenCvAllocatorState~ state_
    +Allocate(bytes, alignment) Result
    +Copy(dst, src, stream) Status
    +Stats() MemoryStats
  }

  class CpuStream {
    -Device device_
    +Synchronize() Status
    +synchronous() bool
  }

  class CudaStream {
    -Device device_
    -cudaStream_t stream_
    +Initialize() Status
    +Synchronize() Status
    +synchronous() bool
    +handle() cudaStream_t
  }

  class CpuAllocatorState {
    +vector~CpuBlock~ free_blocks
    +atomic~size_t~ allocated_bytes
    +atomic~size_t~ peak_bytes
    +Release(data, bytes) void
  }

  class CudaAllocatorState {
    +int device_index
    +vector~CudaBlock~ free_blocks
    +atomic~size_t~ allocated_bytes
    +atomic~size_t~ peak_bytes
    +Release(raw, data, bytes) void
  }

  class OpenCvAllocatorState {
    +atomic~size_t~ allocated_bytes
    +atomic~size_t~ peak_bytes
  }

  class BackendFactory {
    <<utility>>
    +CreateBackend(type, deviceIndex) Result
  }

  Backend <|-- CpuBackend
  Backend <|-- CudaBackend
  Backend <|-- OpenCvBackend

  MemoryManager <|.. CpuMemoryManager
  MemoryManager <|.. CudaMemoryManager
  MemoryManager <|.. OpenCvMemoryManager

  Stream <|.. CpuStream
  Stream <|.. CudaStream

  CpuBackend *-- CpuMemoryManager
  CpuBackend *-- CpuStream
  CudaBackend *-- CudaMemoryManager
  CudaBackend *-- CudaStream
  OpenCvBackend *-- OpenCvMemoryManager
  OpenCvBackend *-- CpuStream

  CpuMemoryManager o-- CpuAllocatorState
  CudaMemoryManager o-- CudaAllocatorState
  OpenCvMemoryManager o-- OpenCvAllocatorState

  BackendFactory ..> Backend : creates
  BackendFactory ..> NpuBackend : placeholder
```

### 12.3 算子、注册表与分发

```mermaid
classDiagram
  direction LR

  class OpContext {
    +Context* context
    +Stream* stream
    +CpuKernelLevel cpu_level
  }

  class Context {
    +AllocateTensor(shape, dtype) Result
    +Copy(dst, src, stream) Status
  }

  class Tensor {
    +shape() Shape
    +dtype() DataType
    +IsContiguous() bool
  }

  class OpKind {
    <<enumeration>>
    kResize
    kGaussianBlur
    kLaplacianPyramid
    kFourierTransform
    kWaveletTransform
  }

  class CpuKernelLevel {
    <<enumeration>>
    kNaive
    kOpenMP
    kSimdGeneric
    kSimdAvx2
    kSimdNeon
    kTbb
  }

  class BackendType {
    <<enumeration>>
    kNativeCpu
    kOpenCV
    kCUDA
  }

  class OpKernelFn {
    <<typedef>>
    Status(OpContext, Tensor, Tensor*, void*, Tensor*)
  }

  class KernelRegistry {
    <<utility>>
    +RegisterKernel(op, backend, level, fn) void
    +FindKernel(op, backend, level) Result
    +HasKernel(op, backend, level) bool
    +HasAnyKernel(backend, level) bool
  }

  class ResizeModule {
    <<module>>
    +Resize(context, input, h, w) Result
    +RegisterResizeKernels() void
  }

  class GaussianBlurModule {
    <<module>>
    +GaussianBlur(context, input, ksize, sigma) Result
    +RegisterGaussianBlurKernels() void
  }

  class LaplacianPyramidModule {
    <<module>>
    +BuildLaplacianPyramid(context, input, levels) Result
    +RegisterLaplacianPyramidKernels() void
  }

  class FourierModule {
    <<module>>
    +FourierTransform(context, input) Result
    +RegisterFourierKernels() void
  }

  class WaveletModule {
    <<module>>
    +WaveletTransform(context, input, kind) Result
    +RegisterWaveletKernels() void
  }

  OpContext --> Context
  OpContext --> Stream
  OpContext --> CpuKernelLevel
  OpKernelFn ..> OpContext
  OpKernelFn ..> Tensor
  KernelRegistry ..> OpKernelFn : stores
  KernelRegistry ..> OpKind : key
  KernelRegistry ..> BackendType : key
  KernelRegistry ..> CpuKernelLevel : key
  ResizeModule ..> KernelRegistry : registers / finds
  GaussianBlurModule ..> KernelRegistry
  LaplacianPyramidModule ..> KernelRegistry
  FourierModule ..> KernelRegistry
  WaveletModule ..> KernelRegistry
  ResizeModule ..> Tensor : input / output
  GaussianBlurModule ..> Tensor
  LaplacianPyramidModule ..> Tensor
  FourierModule ..> Tensor
  WaveletModule ..> Tensor
```

