# 版本演进

> 只记录**已经稳定**的版本。进行中的设计、未落地的计划不写入本文；有稳定改动时再增补对应版本小节。
> 当前架构说明见 [`../architecture.md`](../architecture.md)。

## 版本概览

| 版本 | 状态      | 主题                            |
| -- | ------- | ----------------------------- |
| v1 | 历史稳定版   | 集中式算子与后端内核                   |
| v2 | 历史稳定版   | 算子自治、后端服务化、框架工具层              |
| v3 | 当前稳定版   | SOLID 架构：Context ISP / OpContext class / transfer 自由函数层 / TensorConvert trait |

***

## v1

**定位**：以算子功能覆盖为主，所有算子共享少量集中式注册与后端内核文件。

**公共 API 面**：`add` / `mul` / `cast` / `normalize` / `resize` / `crop` / `transpose` / `warp_affine`。

**结构特征**：

- 算子注册集中在 `src/ops/op_registry.cpp`。
- 参数结构体集中在 `src/ops/op_params.h`。
- CUDA / OpenCV 内核分别集中在 `src/backends/cuda/cuda_kernels.cu`、`src/backends/opencv/opencv_kernels.cpp`。
- 工具函数散落在 `src/ops/` 顶层。

**主要问题**：新增一个算子需要改动多个共享热点文件（注册、参数、两个后端内核、根 CMake），协作冲突成本高；后端内核文件反向包含算子内部头，依赖方向混乱。

***

## v2

**重构重点**：

1. **算子自治**：每个算子一个独立文件夹，入口、全部版本实现、构建、测试、基准、README 自包含。
2. **后端服务化**：`backends` 只提供内存、流、拷贝、设备封装与数据转换，不再承载算子实现。
3. **依赖倒置**：`Backend` 接口归入 `core`；后端实现依赖 core，core 不反向依赖后端；算子只依赖后端服务头。
4. **框架工具层**：注册表、图像工具、dtype 分发、性能剖析与内存统计集中到 `src/framework/`。
5. **范围收敛**：删除元素级算子，保留并强化图像/变换类多版本算子。

### API 变更

**新增**

- `GaussianBlur(...)`：可分离高斯平滑，支持 HW/HWC/CHW/NHWC 与 u8/f32/f64。
- `LaplacianPyramid(...)` 与 `FourierTransform(...)`、`WaveletTransform(...)` 等算子公共头。
- `Tensor::Empty(context, ...)`：唯一的公共分配入口；分配是后端细节，不再挂在 `Context` 上。
- `Copy` / `Upload` / `Download`：`transfer.h` 中的自由函数，执行者由张量所属后端推断。
- `internal::TensorFactory::WrapExternal(...)`：底层外部内存绑定工厂（借用/接管由 deleter 决定），仅供库内适配器使用。

**删除**

- 公共头与实现：`add` / `mul` / `cast` / `normalize` / `crop` / `transpose` / `warp_affine`。
- `Context::backend()` 公共访问器：后端能力不再对调用者暴露。
- 公共头 `include/hpcplatform/backends/backend.h`：`Backend` 转为 core 内部契约。

**行为/兼容性变更**

- `OpKind` 收缩并重编号为 `kResize=1`、`kGaussianBlur=2` 等，随后恢复 append-only 纪律。
- 公共头不再包含任何 `src/**` 内容。

### 代码移动

| 原位置                                | 新位置                                 |
| ---------------------------------- | ----------------------------------- |
| `src/ops/op_registry.{h,cpp}`      | `src/framework/op_registry.{h,cpp}` |
| `src/ops/image_utils.h`            | `src/framework/image_utils.h`       |
| `src/ops/op_common.h`              | `src/ops/common/op_common.h`        |
| `src/ops/cpu_resize.h`             | `src/ops/resize/resize_cpu.h`       |
| `include/hpcplatform/backends/backend.h`  | `src/core/backend.h`                |
| `src/backends/service.h`           | `src/core/backend.h`                |
| `src/backends/backend_copy.cpp`    | `src/core/copy.cpp`                 |
| `src/backends/backend_factory.cpp` | `src/backends/factory.cpp`          |
| `src/backends/backends_internal.h` | `src/backends/backend_internal.h`   |
| `src/ops/registry.cpp`             | `src/ops/register_ops.cpp`          |
| CUDA 内核（集中文件）                      | 各算子 `*_cuda.cu`                     |
| OpenCV 内核（集中文件）                    | 各算子 `*_opencv.cpp`                  |

### 内部设施

- 新增 `src/framework/{compiler,dtype_dispatch,profiling,stats}.h`。
- 新增 `src/core/copy.h`：`CopyTensorData` / `ResolveOwningBackend` 内部声明。
- 新增 `tests/perf/test_operator_perf.cpp`：基于 profiling/stats 的算子性能评估样例。
- `tests/` 重组为 `core/`、`backends/`、`perf/` 与各算子内部 `tests/`。

### 构建与测试

- 可选后端由 `hpcplatform_WITH_CUDA`、`hpcplatform_WITH_OPENCV`、`hpcplatform_WITH_YAML` 以及 `hpcplatform_ENABLE_TBB`、`hpcplatform_ENABLE_OPENMP`、`hpcplatform_ENABLE_SIMD` 门控（均为默认 ON）。
- AVX2 翻译单元使用独立 OBJECT 库，避免 `-mavx2` 影响整个静态库。
- 测试基线（历史）：默认 / 全后端 / ASan 三棵树。**已废弃**：现为单一全特性构建树（`build_debug` / `build_release`）+ GoogleTest + CTest，见 [`docs/qa/testing.md`](../qa/testing.md)。

***

## v3

**版本定位**：SOLID 架构落地（`0.2.0`）。把 v2 中仍挂在 `Context` 上的分配与传输职责剥离，把 `OpContext` 从裸 struct 升为类型安全的 class，把外部类型转换从 Context 方法收敛为编译期 trait。

**核心改动（v2 → v3 的 API 变更）**

| 方向 | 变化 | 原因 |
| --- | --- | --- |
| `Context::AllocateTensor(shape, dtype)` | → `Tensor::Empty(context, shape, dtype)` | ISP：Context 不再承担分配，Tensor 是分配的唯一公共入口 |
| `context->Copy(src, &dst)` | → 自由函数 `Copy(&dst, src)` | Context 不再持有传输；执行者由目标张量所属后端推断 |
| `context->Upload(host, &dev)` | → 自由函数 `Upload(host, &dev)` | 同上；`Upload`/`Download` 只在入口处做内存种类校验 |
| `context->Download(dev, &host)` | → 自由函数 `Download(dev, &host)` | 同上 |
| `context->memory()` 公共访问器 | → 私有；新增 `Context::GetMemoryStats()` | ISP：外部无法绕开 `Tensor::Empty` 直接操作 MemoryManager |
| `OpContext` 裸 struct（`OpContext{ptr, ptr, level}`） | → class，factory：`OpContext(Context&)` / `OpContext::From(Context&)` / `OpContext::Null()` + fluent setter | 强制绑定 Context + stream；禁止无参构造与 aggregate brace-init |
| `OpContext.context` / `.stream` 裸字段 | → accessor `context()` / `stream()` | 封装：外部无法绕过类型安全直接改指针 |
| `Tensor::Create(context, ...)` 公共 | → 私有，通过 `Tensor::Empty` 入口 | 统一分配入口，消除"同一件事多个入口"的困惑 |
| `context->Wrap(data, shape, dtype)` | → `internal::TensorFactory::WrapExternal(context, ...)` | ISP：Wrap 是库内底层工厂，不属于公共 API |
| `CopyFromCvMat` / `CopyToCvMat` | → `TensorConvert<cv::Mat>` 特化（`Tensor::From` / `Tensor::Into`） | OCP：新增第三方类型只需加特化，不改 core |

**验证基线**

- 构建：`cmake --build build -j8` 100% 通过，26 个 target 全部成功。
- 测试：16/17 通过（94%）；3 个预存失败（`test_load_operators` 的 `EnvVarOverridesBackend` / `EnvVarUnavailableBackendFails` / `UnavailableBackendFails`）与本次重构无关。

**设计原则落地**

- **ISP**：Context 只负责后端绑定 / 设备查询 / stream / Synchronize / 能力探测 / 统计；`Tensor` 负责分配；`transfer.h` 负责跨设备传输。
- **OCP**：新增后端/第三方类型只加文件，不改已有核心头；`TensorConvert<T>` trait 是编译期扩展点。
- **单一职责**：OpContext 只管执行参数（Context 引用 + 可选 stream 覆盖 + CPU level），不再承载分配/传输。

***

## 记录约定

后续每个稳定版本按以下顺序补充，保持简短：

1. **版本定位**：一句话说明该版本主题。
2. **API 变更**：新增 / 删除 / 行为变更。
3. **代码移动**：仅记录跨目录或改名。
4. **重构重点**：影响架构或协作方式的变化。
5. **验证基线**：构建树与测试结果。
