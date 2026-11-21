# mycv v2 架构设计：算子自治与服务化后端

> 文档定位：**本项目唯一的架构设计文档**，取代旧版 `PLAN.md` 与 `docs/architecture/P0~P5` 系列。
> **v2 算子自治结构已全部落地**：迁移七步与验收门均已通过（见 §9），§5~§8 描述的是当前真实代码。§2 保留 v1 的问题诊断作为重构依据，其中引用的 v1 文件已删除。
> 配套阅读：[`qa/`](./qa/) 从学习者/贡献者视角解释各模块易错点（路径已刷新为 v2 布局）。

---

## 1. 承继的原则（v1 已验证、v2 继续有效）

以下原则来自 v1 的实践沉淀，本次重构不推翻，是后续所有设计的底座：

1. **显式优先于隐式**：不做自动探测、自动回退、静默转换；一切能力由调用方显式选择。
2. **可复现**：CPU 各版本内核位精确对拍；跨后端才允许有限容差。
3. **描述与存储分离**：`Tensor` 只是视图便签（shape/strides/offset），内存由 `Buffer`/`MemoryManager` 管理。
4. **后端可插拔、核心零第三方依赖**：core 不依赖 OpenCV/CUDA/TBB；可选后端全部编译期开关门控（`MYCV_WITH_*` / `MYCV_ENABLE_*`）。
5. **无异常**：`-fno-exceptions` 约束下的 `Status`/`Result` 错误模型（见 [`qa/error_model.md`](./qa/error_model.md)）。
6. **错误码三分类**：参数错误 `kInvalidArgument`、编译期未启用 `kUnsupported`、注册表缺内核 `kNotImplemented`，绝不静默降级（见 [`qa/op_dispatch.md`](./qa/op_dispatch.md)）。

---

## 2. 问题诊断：为什么必须重构

### 2.1 现状证据：新增一个算子要改 7 处、跨 4 个共享文件

以 v1 中新增 `crop` 的真实改动为例：

| # | 文件（v1，均已删除或改造） | 改动性质 |
| --- | --- | --- |
| 1 | `include/mycv/ops/crop.h` | 新增（无冲突） |
| 2 | `src/ops/crop/` 目录 | 新增（无冲突） |
| 3 | `src/ops/op_registry.cpp` | **共享文件**：前向声明 + `EnsureRegistered` 注册 |
| 4 | `src/ops/op_params.h` | **共享文件**：追加 `CropParams` |
| 5 | `src/backends/cuda/cuda_kernels.cu` | **共享文件**：追加 kernel + 封装 + 注册（800 行文件） |
| 6 | `src/backends/opencv/opencv_kernels.cpp` | **共享文件**：追加 `OpenCvCrop` |
| 7 | 根 `CMakeLists.txt` | **共享文件**：源列表登记 |

问题不在"处数多"，而在 **#3~#7 是所有算子共享的热点文件**。协作开发时，两个人各加一个算子（或一人加算子、一人修 bug），必然在同一批文件冲突；后端内核文件动辄数百行，冲突解决成本高、漏合风险大。

### 2.2 SOLID 逐条诊断

| 原则 | 现状问题 | 证据 |
| --- | --- | --- |
| **S** 单一职责 | 一个后端内核文件承载所有算子实现 | v1 的 `cuda_kernels.cu` 单文件 8 个算子约 800 行；`opencv_kernels.cpp` 同时承担 8 个算子 + Mat 转换辅助 + 异常桥接（两者均已删除，能力抽入 `cuda_device_utils.h` / `opencv_tensor_access.{h,cpp}`） |
| **O** 开闭 | 加算子必须**修改**共享注册文件与后端文件，而非只**新增** | §2.1 表 #3~#7 |
| **L** 里氏替换 | 尚可：各后端对同一算子的语义一致性由跨后端对拍保证 | [`qa/testing.md`](./qa/testing.md) §4 |
| **I** 接口隔离 | 尚可：`OpKernelFn` 五参签名虽宽（图像算子不用 `other`），但它是注册表统一值类型的必要代价；算子内部仍可定义更窄的私有函数再适配 | [framework/op_registry.h](../src/framework/op_registry.h) |
| **D** 依赖倒置 | **依赖方向混乱**：后端内核文件反过来包含算子内部头（`ops/image_utils.h`、`ops/op_params.h`），形成 backends→ops 边；而算子又通过注册表依赖框架，双向纠缠 | v1 的 `cuda_kernels.cu` L12-14、`opencv_kernels.cpp` L9-12（文件已删除）；现状由 §4.2 R2/R3 约束 |

### 2.3 次要问题

- **工具函数无统一归置规则**：`image_utils.h` 等 header-only 工具散在 `src/ops/` 顶层，热路径内联与冷路径函数混在同一文件，缺少成文的归置规则（由本文档 §6.4 补齐）。
- **缺少可观测性设施**：无统一的耗时统计、内存/显存占用快照工具，算子优化只能靠手工计时。
- **范围发散**：8 个算子摊薄了维护精力，核心价值（图像算子的多版本优化与多后端一致性）反而不突出。

---

## 3. 设计目标与非目标

**目标**

1. **开闭原则为第一优先**：新增算子 = 新增一个文件夹 + 若干一行式登记；不修改任何算子间共享的实现文件。
2. **算子自治**：每个算子文件夹是完备解决方案——公共入口、全部版本实现（含各后端）、构建脚本、单元测试、基准测试、说明文档。
3. **后端服务化**：backends 只提供"能力服务"（内存、流、拷贝、后端 API 封装、数据转换函数），被算子直接使用，但**不暴露给最终调用者**。
4. **框架工具层统一规划**：注册表、图像工具、dtype 分发、剖析/统计（内存与显存占用、耗时）集中到 `src/framework/`，并明确"热路径内联 / 冷路径落 .cpp"的归置规则。
5. **范围收缩**：只保留并重点实现 **resize** 与新增的 **gaussian_blur（高斯平滑）** 两个多版本算子。

**非目标**（明确不做，防止设计漂移）

- 不做动态库插件化 / 运行时算子自动发现（链接器语义决定的静态注册方案见 §6.3）。
- 不做自动调优（`CpuKernelLevel` 仍由调用方显式指定）。
- 不做多语言绑定、异步执行图。

---

## 4. 总体架构与依赖规则

### 4.1 分层图

```
                 include/mycv/**                ← 唯一公共 API 面（调用者只见这里）
                        ↑
                     src/core/               ← Tensor/Shape/Buffer/MemoryManager/Stream/Context
                        ↑
                  src/framework/            ← 注册表、图像工具、dtype 分发、剖析与统计
                   ↑         ↑
        src/backends/       src/ops/
        （服务提供者）        （算子自治，每算子一个文件夹）
              ↑
   src/ops/<op>/<op>_cuda.cu 等后端实现文件
   （位于算子文件夹内，可引用对应后端的服务头）
```

### 4.2 依赖规则（评审时逐条检查）

- **R1** 公共头（`include/mycv/**`）不得包含任何 `src/**`。可选能力（如 `interop/opencv.h` 的 Mat 适配器）用 `#ifdef MYCV_WITH_*` 门控。
- **R2** `src/backends/**` 只能依赖 `include/mycv/**` 与 `src/framework/**`，**禁止**包含 `src/ops/**`（消除 v1 的 backends→ops 边）。
- **R3** `src/ops/<op>/` 的**后端实现文件**（`*_cuda.cu`、`*_opencv.cpp`）可以包含对应后端的服务头（`src/backends/cuda/**`、`src/backends/opencv/**`）；CPU 实现文件不得包含任何后端头。
- **R4** `src/ops/<op>/` 的入口与 CPU 实现只依赖 `include/mycv/**`、`src/framework/**`、`src/ops/common/**` 与本文件夹内部头。
- **R5** 依赖单向无环：`core ← framework ← {backends, ops}`；`ops → backends(服务头)` 只存在于 R3 允许的文件中。

---

## 5. 目录结构（v2 已落地）

> 标记：`【新】` 本次新增，`【移】` 移动/改名，`【删】` v1 文件已删除。本节即当前仓库的真实结构（`build*` 目录除外）。

```
include/mycv/
  core/                  # 不变：shape/dtype/error/device/memory/stream/tensor/tensor_iterator/context
  ops/
    image.h              # 不变：InterpolationMode / ImageLayout
    op.h                 # OpKind = {kResize=1, kGaussianBlur=2}；OpContext{context, stream, cpu_level}
    resize.h             # 保留
    gaussian_blur.h      # 【新】GaussianBlur(..., int kernel_size, double sigma, ImageLayout layout)
  interop/
    opencv.h             # 保留：CopyFromCvMat / CopyToCvMat（对外数据适配器）
                         # Backend 接口已移出公共头（见 §6.2）
  mycv.h                 # 伞形头：core + image/op/resize/gaussian_blur

src/
  core/                  # 基本不变（tensor_iterator 保留：Context::Copy 依赖）
  framework/             # 【新】框架工具层（§6.4）
    op_registry.{h,cpp}  # 【移】自 src/ops/，语义不变
    image_utils.h        # 【移】自 src/ops/（ImageInfo/ParseImage/MakeImageShape/ImageByteOffset）
    dtype_dispatch.h     # 【新】DispatchImageDType(u8/f32/f64) + MYCV_CPU_IVDEP
    compiler.h           # 【新】编译器暗示宏
    profiling.h          # 【新】MYCV_PROFILE_SCOPE / MYCV_PROFILE_OP（默认零开销）
    stats.h              # 【新】SnapshotMemory（按 MemoryKind 聚合）
  backends/              # 服务层：只提供能力，不含任何算子实现
    service.h            # 【移】Backend 接口（原 include/mycv/backends/backend.h）
    backend_factory.cpp  # 不变（Context::Create 仍走这里）
    backend_copy.cpp     # 不变
    cpu/                 # 内存管理器、流不变
    cuda/
      cuda_common.h      # 不变（CudaStatus）
      cuda_device_utils.h# 【新】DeviceTensor/DeviceImage/ThreadIndex/ThreadStride/launch 辅助
                         #      （自已删除的 cuda_kernels.cu 抽出）
    opencv/
      opencv_tensor_access.{h,cpp}  # 【新】CvDepth/CvCall/FlatMat/ImageMat/RejectNchw
                                    #      （自已删除的 opencv_kernels.cpp 抽出）
    npu/                 # 占位
  ops/
    registry.cpp         # 【新】两行注册清单（§6.3）
    CMakeLists.txt       # 【新】每算子一行 add_subdirectory
    common/
      op_common.h        # 【移】自 src/ops/；只剩 ValidateInvocation / AllocateOutput
    resize/              # 自治文件夹（§6.1）
    gaussian_blur/       # 自治文件夹（§6.1）
    fourier/             # 自治文件夹（§6.1）
    wavelet/             # 自治文件夹（§6.1）

tests/
  test_framework.h       # 不变（零依赖测试框架）
  core/                  # 【新】由 unit/ 重组：shape/tensor/views/memory/backend/copy，6 个 TU
  backends/              # 【新】由 consistency/ 重组：CUDA/OpenCV 后端级测试（拷贝、生命周期、跨后端对拍）
  CMakeLists.txt         # 薄壳：只 add_subdirectory(core) / add_subdirectory(backends)

samples/                 # 5 个样例；图像链路只用 resize + gaussian_blur
                         # 【删】顶层 benchmarks/（基准移入各算子 benchmark/）
```

---

## 6. 关键设计

### 6.1 算子文件夹契约（已落地）

每个算子文件夹是一个**完备解决方案**：公共入口、全部版本实现（含各后端）、构建脚本、测试、基准、说明文档。五个算子的真实内容：

```
src/ops/resize/
  resize_internal.h        # 内部契约：ResizeParams、CPU 内核原型、RegisterResizeKernels()
  resize_cpu.h             # CPU 内核声明（v1 src/ops/cpu_resize.h 改名迁入）
  resize.cpp               # 入口：校验 → FindKernel → 分配输出 → 调内核；并注册全部版本
  resize_cpu_naive.cpp     # 参考实现（long double 累加；u8 输出 half-away-from-zero）
  resize_cpu_omp.cpp       # MYCV_WITH_OPENMP
  resize_cpu_tbb.cpp       # MYCV_WITH_TBB
  resize_cuda.cu           # MYCV_WITH_CUDA
  resize_opencv.cpp        # MYCV_WITH_OPENCV
  CMakeLists.txt           # 逐文件宏门控 + 挂本算子 test_resize / resize_benchmark
  tests/test_resize.cpp    # 手算值、NCHW/nearest、单元素、CPU 各级 memcmp 位精确
  benchmark/resize_benchmark.cpp
  README.md                # 版本矩阵、容差约定、profiling 数据

src/ops/gaussian_blur/
  gaussian_blur_internal.h      # GaussianBlurParams、BuildGaussianKernel、内核原型、注册声明
  gaussian_blur_cpu.h           # 共享 CPU 像素骨架（索引解码/REPLICATE 边界/两趟累加）
  gaussian_blur.cpp             # 入口 + 六版本注册
  gaussian_blur_cpu_naive.cpp   # 参考实现（double 单累加器，i=0..k-1 升序）
  gaussian_blur_cpu_omp.cpp     # MYCV_WITH_OPENMP
  gaussian_blur_cpu_tbb.cpp     # MYCV_WITH_TBB
  gaussian_blur_cpu_simd.cpp    # MYCV_ENABLE_SIMD（MYCV_WITH_SIMD_GENERIC）
  gaussian_blur_cpu_avx2.cpp    # MYCV_COMPILER_HAS_AVX2；独立 OBJECT 库带 -mavx2
  gaussian_blur_cuda.cu         # MYCV_WITH_CUDA（两趟 kernel，f64 中间缓冲）
  gaussian_blur_opencv.cpp      # MYCV_WITH_OPENCV（cv::GaussianBlur + BORDER_REPLICATE）
  CMakeLists.txt / tests/test_gaussian_blur.cpp / benchmark/ / README.md

src/ops/laplacian_pyramid/    # 组合算子：基于 resize + gaussian_blur，无自有像素级版本
  laplacian_pyramid_internal.h  # LaplacianPyramidParams（含 next_gaussian 输出槽）、阶段内核原型
  laplacian_pyramid.cpp         # 入口：逐级分配/调用阶段内核、组装 levels 向量；版本注册
  laplacian_pyramid_host.cpp    # CPU 与 OpenCV：组合子算子 + host 端逐元素相减
  laplacian_pyramid_cuda.cu     # CUDA：组合子算子 + device 端相减 kernel
  CMakeLists.txt / tests/test_laplacian_pyramid.cpp / benchmark/ / README.md

src/ops/fourier/              # 自包含算子：2-D FFT
  fourier_internal.h          # FourierParams（正向 imag 输出槽 / 逆向 other 输入）、内核原型
  fourier_cpu.h               # 共享骨架：twiddle 表、radix-2 / O(n^2) DFT、逐平面驱动
  fourier.cpp                 # 入口：校验 → FindKernel → 分配 real/imag → 调内核；版本注册
  fourier_cpu_naive.cpp       # 参考实现（SequentialRunner）
  fourier_cpu_omp.cpp         # MYCV_WITH_OPENMP（OpenMPRunner）
  fourier_cpu_tbb.cpp         # MYCV_WITH_TBB（TbbRunner）
  fourier_cuda.cu             # MYCV_WITH_CUDA：每行/列一个 block，就地 radix-2 蝶形
  CMakeLists.txt / tests/test_fourier.cpp / benchmark/ / README.md

src/ops/wavelet/              # 自包含算子：单层正交 2-D DWT（Haar / Db4）
  wavelet_internal.h          # WaveletParams（extra_in/extra_out 三槽）、滤波器系数、内核原型
  wavelet_cpu.h               # 共享骨架：行/列分析与合成、逐平面驱动
  wavelet.cpp                 # 入口：校验（偶数尺寸）→ 分配四带 → 调内核；版本注册
  wavelet_cpu_naive.cpp       # 参考实现（SequentialRunner）
  wavelet_cpu_omp.cpp         # MYCV_WITH_OPENMP（OpenMPRunner）
  wavelet_cpu_tbb.cpp         # MYCV_WITH_TBB（TbbRunner）
  wavelet_cuda.cu             # MYCV_WITH_CUDA：逐样本线程的分析/合成内核
  CMakeLists.txt / tests/test_wavelet.cpp / benchmark/ / README.md
```

契约要点：

- **文件唯一归属**：除 §11 列出的登记点外，本文件夹内文件不被任何其他算子触碰。改 `resize` 的 OpenCV 实现只动 `resize_opencv.cpp`，与别的算子零冲突。
- **版本矩阵**：README.md 维护一张表——每个 `(后端, CpuKernelLevel)` 组合是否存在、由哪个文件实现、对拍方式。这是"某算子的某版本实现在哪"的单一答案。
- **参数结构体归位**：`ResizeParams` / `GaussianBlurParams` 定义在各自的 `*_internal.h`（v1 集中在已删除的 `src/ops/op_params.h`）。
- **公共入口模式不变**：固定骨架（校验 → 查注册表 → 分配输出 → 调内核），公共头（`include/mycv/ops/*.h`）只声明 API。
- **版本不齐不算错**：resize 从未有 SIMD 实现（v1 的 SIMD 只存在于已删除的元素级算子），因此请求 `kSimdGeneric`/`kSimdAvx2` 时注册表查不到内核、返回 `kNotImplemented`，`SupportsKernelLevel` 返回 false。需要 SIMD 时的落地样板见 gaussian_blur。
- **AVX2 的构建约束**：`set_source_files_properties(... COMPILE_OPTIONS -mavx2)` 是目录作用域的，无法作用于父目录创建的 `mycv` target。因此 AVX2 TU 编成独立 OBJECT 库（`-mavx2` + `MYCV_ARCH_AVX2`），再用 `target_sources(mycv PRIVATE $<TARGET_OBJECTS:...>)` 嵌入——不用 `target_link_libraries`，避免被连带进 `install(EXPORT)`。注意注册代码（如 `gaussian_blur.cpp`）属于 `mycv` target，`MYCV_ARCH_AVX2` 必须同时 `target_compile_definitions(mycv PRIVATE MYCV_ARCH_AVX2=1)`，否则 `#ifdef` 会静默排除 AVX2 内核注册；`-mavx2` 仍只加在 OBJECT 库上，保持静态库可移植。


### 6.2 后端服务层：能力提供者，不是算子容器

**后端收窄后的职责**：内存管理器（`MemoryManager`）、流（`Stream`）、`CopyBuffer`、后端 API 封装、数据转换函数。**不再包含任何具体算子的实现。**

**对算子暴露的服务头**（内部，最终调用者不可见）：

| 服务头 | 内容（均来自 v1 代码的抽取/移动） | 使用者 |
| --- | --- | --- |
| `src/backends/cuda/cuda_device_utils.h`【新增】 | `DeviceTensor`/`DeviceImage` 打包、`ThreadIndex`/`ThreadStride`、launch 辅助——原散在 cuda_kernels.cu（已删除） | 各算子 `*_cuda.cu` |
| `src/backends/opencv/opencv_tensor_access.{h,cpp}`【新增】 | `CvDepth`/`CvCall`/`FlatMat`/`ImageMat`/`RejectNchw`——原散在 opencv_kernels.cpp（已删除） | 各算子 `*_opencv.cpp` |
| `src/framework/image_utils.h`（自 `src/ops/` 移入） | `ImageInfo`/`ParseImage`/`MakeImageShape`/`ImageByteOffset`/`ImageLinearIndex` | 图像算子的所有版本 + 后端转换函数 |

**公共面收缩**【API 变更】：

- `Backend` 接口（原 include/mycv/backends/backend.h）已整体移入 `src/backends/service.h`；公共头删除该文件。`Context` 公共头只前向声明 `Backend`，`Context::Create(BackendType, device_index)` 仍是唯一公开构造路径。
- `Context::backend()` 访问器从公共 API 移除（测试、算子作为内部消费者改用内部头访问）。
- `BackendType`/`DeviceType`/`Device`（[device.h](../include/mycv/core/device.h#L12-L23)）保持公共——`Context::Create` 的参数需要它，但它只是枚举，不暴露能力。
- [`interop/opencv.h`](../include/mycv/interop/opencv.h) 的 `CopyFromCvMat`/`CopyToCvMat` 保留公共：它们是对外数据适配器（把外部 `cv::Mat` 变成 mycv `Tensor`），不是后端能力服务。

**为什么依赖方向这样定**：v1 的痛点是 cuda/opencv 内核文件（属于 backends）反过来包含算子内部头（属于 ops）。v2 把"算子的后端实现"移进算子文件夹后，这些文件天然属于 ops 侧，需要什么后端设施就包含对应**服务头**——依赖箭头统一为 ops→backends，backends 永远不知道算子的存在。这正是依赖倒置：算子（高层策略）依赖后端服务（低层能力），且依赖的是稳定的窄接口头，而非后端实现细节。

### 6.3 注册与分发：注册表语义不变，注册位置改变

**不变的部分**：三元组分发键 `(OpKind, BackendType, CpuKernelLevel) → OpKernelFn`、`RegisterKernel`/`FindKernel`/`HasKernel` 语义、三分类错误码——整套分发协议原样保留（[`qa/op_dispatch.md`](./qa/op_dispatch.md) 仍然准确）。注册表实现从 `src/ops/op_registry.{h,cpp}` 移到 `src/framework/`。

**改变的部分**：注册的**调用点**从 v1 集中式的 `op_registry.cpp EnsureRegistered()`（已删除）改为 `src/ops/registry.cpp` 的唯一清单：

```cpp
// src/ops/registry.cpp —— 当前全部内容
void RegisterAllKernels() {
  RegisterResizeKernels();
  RegisterGaussianBlurKernels();
}
```

每个算子的 `Register<Op>Kernels()` 定义在本算子文件夹内（如 `resize_internal.h` + `resize.cpp`），把该算子**全部版本**（CPU 各级 + CUDA + OpenCV）一次性注册。给已有算子加一个后端实现 = 算子文件夹加 1 个文件 + 其 `Register<Op>Kernels()` 里加 1 行（宏门控）——**不碰任何共享文件**。

**为什么不用静态自注册（`static Registrar` 魔法）**：静态库中未被引用符号的翻译单元不会被链接器拉入，自注册对象的构造函数可能根本不执行，内核会**静默缺失**——要靠 `--whole-archive` 等链接器魔法补救，与"显式优先于隐式"原则冲突。显式一行清单代价是加算子多登记一行，换来行为完全可预期。`std::call_once` 惰性注册机制原样保留。

### 6.4 框架工具层（已落地）

**归置规则**（对 v1 "工具该不该内联"问题的系统性回答）：

- **逐像素热路径辅助**（每像素/每行调用）：`inline` 定义在 framework 头文件中，随使用方编译内联——如 `ImageByteOffset`（v1 中 resize 双线性每像素调 4 次，见 [`qa/cpubackends.md`](./qa/cpubackends.md) 对热路径的讨论）。
- **一次性冷路径函数**（每算子调用至多一次，如 `ParseImage` 校验解析）：声明在 framework 头，定义落 `framework/*.cpp`，修改实现只重编一个 TU。当前 `image_utils.h` 仍整体 header-only（inline）；日后新增冷路径函数时按此规则落 .cpp。
- 禁止"为了统一而统一"：不把热函数强行改成跨 TU 调用，那是可测量的性能回退。

**剖析与统计设施**（`src/framework/`）：

```cpp
// src/framework/profiling.h —— 编译期开关 MYCV_ENABLE_PROFILING（默认 OFF，零开销）
MYCV_PROFILE_SCOPE("resize.bilinear");            // RAII 作用域计时
MYCV_PROFILE_OP(OpKind::kResize, backend, level); // 自动以三元组为标签计时

// src/framework/stats.h —— 基于 MemoryManager::Stats() 既有接口聚合
struct MemorySnapshot {  // 按 MemoryKind 聚合：kHost / kHostPinned / kDevice
  MemoryStats host;        // allocated_bytes / peak_bytes / allocation_count
  MemoryStats host_pinned;
  MemoryStats device;      // 即显存占用
};
MemorySnapshot SnapshotMemory(Context& context);
```

- `MemoryStats`（[memory.h](../include/mycv/core/memory.h#L21-L25)）已提供 `allocated_bytes`/`peak_bytes`/`allocation_count`，`SnapshotMemory` 只做跨 MemoryManager 聚合，不需要后端改动。
- 算子 `benchmark/` 的标准输出格式：各版本耗时表 + 前后内存快照，结果记入该算子 `README.md`，形成"优化手册"的原始素材。
- `dtype_dispatch.h` 提供 `DispatchImageDType`（u8/f32/f64 三分支，替代 v1 面向 8 种算术类型的 `DispatchArithmeticDType`——后者随元素级算子一起删除）。

### 6.5 不变的内核纪律

- CPU 各版本之间**位精确**（`memcmp` 对拍 naive 基准）；跨后端 `1e-4` 相对容差（u8 规则见 [`qa/testing.md`](./qa/testing.md) §4）。
- 位精确要求 SIMD/并行版本保持与 naive 一致的累加顺序——对高斯这类归约型算子尤其关键（见 §7.2）。
- `CopyBuffer` 不经主机偷偷中转；非 CPU 后端只接受 `kNaive`；`Context::Copy` 同步返回契约——全部沿用（[`qa/context_copy.md`](./qa/context_copy.md)）。

---

## 7. 范围收缩：resize 保留，gaussian_blur 新增

### 7.1 resize：原样迁移（已完成）

实现体基本原样搬家，未重写逻辑：CPU 各级（naive/OpenMP/TBB）来自 v1 `src/ops/resize/*` 与 `src/ops/cpu_resize.h`（改名 `resize_cpu.h` 迁入算子文件夹）；CUDA 版本从已删除的 `cuda_kernels.cu` 的 `ResizeKernel` 段抽出为 `resize_cuda.cu`；OpenCV 版本从已删除的 `opencv_kernels.cpp` 抽出为 `resize_opencv.cpp`。公共 API（[resize.h](../include/mycv/ops/resize.h)）未变。

resize 没有 SIMD 版本：v1 的 SIMD 内核只属于已删除的元素级算子，因此算子文件夹内不含 `resize_cpu_simd/avx2.cpp`。请求这两级时是 `kNotImplemented`（见 §6.1）。

### 7.2 gaussian_blur：新算子（已实现）

**API**：

```cpp
// include/mycv/ops/gaussian_blur.h
// 可分离高斯平滑。layout 支持 HW/HWC/CHW/NHWC（同 resize 规则）；
// dtype 支持 u8/f32/f64（IsSupportedImageDType）。
// kernel_size 必须为 >=3 的奇数；sigma 必须 > 0 且有限。
// 边界策略：REPLICATE（边缘复制），对齐 cv::BORDER_REPLICATE 便于跨后端对拍。
Result<Tensor> GaussianBlur(const OpContext& context, const Tensor& input,
                            int kernel_size, double sigma,
                            ImageLayout layout = ImageLayout::kNHWC);
```

**实现矩阵与位精确策略**：

| 版本 | 文件 | 关键点 |
| --- | --- | --- |
| naive | `gaussian_blur_cpu_naive.cpp` | 可分离两趟（水平→垂直），核按下标从小到大累加，固定求和顺序 |
| OpenMP | `gaussian_blur_cpu_omp.cpp` | 按行/按块并行——并行粒度不改变单像素累加顺序，位精确成立 |
| SIMD 通用 | `gaussian_blur_cpu_simd.cpp` | 水平方向向量累加需保持与 naive 相同的顺序；验收以 memcmp 为准 |
| AVX2 | `gaussian_blur_cpu_avx2.cpp` | 同上；独立 TU 单独 `-mavx2` |
| CUDA | `gaussian_blur_cuda.cu` | 两趟 kernel；只接受 `kNaive` 级别 |
| OpenCV | `gaussian_blur_opencv.cpp` | `cv::GaussianBlur`；跨后端容差 `1e-4` 对拍，不要求与自研位精确 |

**校验规则**（入口层，返回 `kInvalidArgument`）：kernel_size 奇数且 ≥3；sigma > 0 且有限；图像 dtype/layout 同 resize 的校验链（复用 `ValidateInvocation` 与 `framework/image_utils.h`）。

**u8 输出语义**：中间累加用 f32/f64，输出时舍入回 u8（舍入规则与 naive 保持一致并在 README.md 写明），保证 SIMD 版本位精确。

### 7.3 删除清单（已执行）

- 公共头：`add.h` `mul.h` `cast.h` `normalize.h` `crop.h` `transpose.h` `warp_affine.h`
- 实现目录：`src/ops/{add,mul,cast,normalize,crop,transpose,warp_affine}/` 全部
- 共享内核骨架：`cpu_binary.h` `cpu_cast.h` `cpu_normalize.h` `cpu_transpose.h` `cpu_dispatch.h`（8 类算术 dtype 分发随之删除）、`op_params.h`（仅 `ResizeParams` 迁移，新增 `GaussianBlurParams` 入算子内部头）
- 顶层基准：`benchmarks/operator_benchmark.cpp`、`benchmarks/image_pipeline_benchmark.cpp`（由各算子 `benchmark/` 取代）
- 测试：`tests/unit/test_ops_elementwise.cpp`（全部算子已删）、`tests/consistency/test_cpu_kernels.cpp`（拆入两个算子各自的 `tests/`）；`test_copy_cast.cpp` 中 Cast 用例随算子删除，Copy 部分改名 `tests/core/test_copy.cpp` 保留
- samples：`02_ops_pipeline` / `03_opencv_backend` / `04_cuda_pipeline` 改写为只用 resize + gaussian_blur
- **明确保留**：`tensor_iterator.h`（`Context::Copy` 依赖，[backend_copy.cpp](../src/backends/backend_copy.cpp) 在用）、`value_cast.h`（resize/OpenCV 转换在用）、`image.h` 的 `InterpolationMode`/`ImageLayout`
- **连带清理**：`src/ops/common/op_common.h` 只剩 `ValidateInvocation` / `AllocateOutput`（`ValidateBinaryInputs`、`IsSupportedArithmeticDType`、`StridedByteOffset` 已无调用方，一并删除）

**OpKind 重编号声明**：`kResize=1, kGaussianBlur=2`。v1 的"append-only"规则服务于算子集合稳定时的兼容性；本次是声明过的大版本破坏性变更，保留 `kResize=5` 等空洞没有收益。此决定记录在案，v2 交付后恢复 append-only 纪律。

---

## 8. 迁移修改点（file-by-file，已执行）

### 8.1 删除

| 路径 | 原因 |
| --- | --- |
| `include/mycv/ops/{add,mul,cast,normalize,crop,transpose,warp_affine}.h` | 范围收缩（§7.3） |
| `src/ops/{add,mul,cast,normalize,crop,transpose,warp_affine}/` | 同上 |
| `src/ops/{cpu_binary,cpu_cast,cpu_normalize,cpu_transpose,cpu_dispatch}.h` | 随元素级算子删除 |
| `src/ops/op_params.h` | 拆入各算子 `*_internal.h` |
| `src/backends/cuda/cuda_kernels.cu` | 实现迁入算子文件夹后删除 |
| `src/backends/opencv/opencv_kernels.cpp` | 同上 |
| `benchmarks/`（整个顶层目录） | 基准移入算子文件夹 |
| `tests/unit/test_ops_elementwise.cpp`、`tests/consistency/test_cpu_kernels.cpp` | 拆分或随算子删除 |
| `PLAN.md`、`docs/architecture/`（旧 README + P0~P5）、`docs/qa/how_to_extend.md` | 过时文档（§10） |

### 8.2 移动（语义不变，路径改变）

| 原路径 | 新路径 |
| --- | --- |
| `src/ops/op_registry.{h,cpp}` | `src/framework/op_registry.{h,cpp}` |
| `src/ops/image_utils.h` | `src/framework/image_utils.h` |
| `src/ops/op_common.h` | `src/ops/common/op_common.h`（删 `ValidateBinaryInputs`/`IsSupportedArithmeticDType`/`StridedByteOffset`） |
| `src/ops/cpu_resize.h` | `src/ops/resize/resize_cpu.h` |
| `include/mycv/backends/backend.h`（`Backend` 接口） | `src/backends/service.h`（公共头删除） |

### 8.3 新建

| 路径 | 内容 |
| --- | --- |
| `src/framework/{profiling.h,stats.h,compiler.h,dtype_dispatch.h}` | §6.4 |
| `src/backends/cuda/cuda_device_utils.h` | DeviceTensor/launch 辅助抽取 |
| `src/backends/opencv/opencv_tensor_access.{h,cpp}` | CvCall/ImageMat 等抽取 |
| `src/ops/{registry.cpp,CMakeLists.txt}` | 一行式算子注册清单 + 算子子目录装载 |
| `src/ops/resize/*`、`src/ops/gaussian_blur/*` | 两个自治算子文件夹（§6.1/§7.2） |
| `include/mycv/ops/gaussian_blur.h` | 新算子公共 API |
| `tests/{core,backends}/` | 测试重组 |

### 8.4 修改

| 路径 | 改动 |
| --- | --- |
| `include/mycv/ops/op.h` | `OpKind` 收缩为 2 项并重编号；`OpKindName` 对应收缩 |
| `include/mycv/mycv.h` | 伞形头：删 7 个算子头；加 `gaussian_blur.h` |
| `include/mycv/core/context.h` | `Backend` 改前向声明；`backend()` 访问器移除 |
| 根 `CMakeLists.txt` | 源列表收缩为 core+backends+framework；`add_subdirectory(src/ops)`；删 benchmarks 子目录；选项开关全部不变 |
| `tests/CMakeLists.txt` | 薄壳化：只 `add_subdirectory(core)` / `add_subdirectory(backends)` |
| `samples/` | 改写为 resize + gaussian_blur 演示 |

---

## 9. 迁移步骤与验收门（已全部完成）

按以下顺序执行，每步结束三棵构建树必须全绿（default / full / asan，命令见 [`qa/testing.md`](./qa/testing.md) §1）。状态为实际执行结果：

| # | 步骤 | 状态 |
| --- | --- | --- |
| 1 | **框架层抽出**：建 `src/framework/`，移动 `op_registry`/`image_utils`/`op_common`，新增 profiling/stats/dtype_dispatch——纯移动+新增，行为零变化 | ✅ 三树 9/9、11/11、10/10 |
| 2 | **后端服务抽取**：从 `cuda_kernels.cu` 抽 `cuda_device_utils.h`，从 `opencv_kernels.cpp` 抽 `opencv_tensor_access.{h,cpp}`，`Backend` 内部化 | ✅ 三树 9/9、11/11、10/10 |
| 3 | **resize 迁移**：建 `src/ops/resize/` 自治文件夹，CUDA/OpenCV 实现从后端文件搬入；原后端文件对应段删除 | ✅ 三树全绿 |
| 4 | **gaussian_blur 实现**：新文件夹 + 六版本 + 测试 + 基准 + README | ✅ 三树 10/10、12/12、11/11 |
| 5 | **收缩与清理**：删 7 个算子及其测试/基准/头文件，更新 `op.h`/`mycv.h`/samples/注册清单 | ✅ 三树 8/8、10/10、9/9，3 个 samples 实跑通过 |
| 6 | **测试重组**：`tests/{core,backends}/` + 算子文件夹内 `tests/`、`benchmark/` | ✅ 三树 8/8、10/10、9/9 |
| 7 | **收尾**：grep 无残留 include；三树全绿；README 版本矩阵填写 | ✅ 零残留；dummy 算子演练通过并回滚 |

**验收门（全部满足才可判迁移完成）**

- ✅ `build-default`（无可选后端 8/8）、`build`（CUDA+OpenCV+TBB 全开 10/10）、`build-asan`（9/9）三棵树 `ctest` 全绿。
- ✅ resize 位精确对拍覆盖全部可用 CPU 级别（`tests/test_resize.cpp` 的 `ResizeCpuLevelsAreBitExact`）；跨后端容差对拍通过。
- ✅ gaussian_blur 六版本矩阵齐备（naive/OpenMP/SIMD/AVX2/TBB/CUDA/OpenCV），CPU 各级 memcmp 位精确，跨后端 1e-4（u8 ±1）通过。
- ✅ 新增算子演练：按 §11 清单试加空算子，仅新增 3 个文件 + 4 个登记点（5 行），分发返回 `kNotImplemented`，验证后已回滚。

---

## 10. 文档清理记录（本次已完成）

| 动作 | 对象 | 原因 |
| --- | --- | --- |
| 删除 | `PLAN.md`（根目录） | v1 的 P0~P6 阶段规划，已被本设计取代 |
| 删除 | `docs/architecture/README.md` + `P0.md`~`P5.md` | 描述 v1 目录结构的分阶段架构文档，被本文档取代；其中仍然有效的设计原则已收进本文档 §1 |
| 删除 | `docs/qa/how_to_extend.md` | 其扩展指南锚定 v1 目录结构（"加算子要改 op_registry.cpp/cuda_kernels.cu/…"），与 v2 矛盾；由本文档 §11 取代 |
| 保留 | `docs/qa/` 其余 8 篇 | 描述 core 层行为（视图/内存生命周期/错误模型/Copy/CUDA 可见性/测试纪律），v2 中这些机制不变；文中引用的路径已刷新为 v2 布局 |

---

## 11. 扩展指南（v2 布局）

**新增算子 X**（全部为新增或一行式登记，无共享实现文件改动）：

1. `include/mycv/ops/x.h` —— 公共 API 声明（新文件）。
2. `include/mycv/ops/op.h` —— `OpKind` 追加一枚举值（append-only，此后不得重编号）。
3. `src/ops/x/` —— 自治文件夹：`x_internal.h`、`x.cpp`、各版本实现、`CMakeLists.txt`、`tests/`、`benchmark/`、`README.md`。
4. `src/ops/registry.cpp` —— 加一行 `RegisterXKernels();`。
5. `src/ops/CMakeLists.txt` —— 加一行 `add_subdirectory(x)`。
6. `include/mycv/mycv.h` —— 加一行 include。

**给已有算子加后端实现**：算子文件夹加 1 个实现文件 + 算子内 `RegisterXKernels()` 加 1 行宏门控注册 + 算子 `CMakeLists.txt` 加 1 行。core/backends/其他算子零改动。

**新增后端 B**：`src/backends/b/`（服务头 + 内存管理器 + 流）+ [backend_factory.cpp](../src/backends/backend_factory.cpp) 加 1 个 `case` + CMake 选项。各算子**自愿**增量添加 `*_b.cpp`——不加也不破坏现有构建（查注册表返回 `kNotImplemented`，语义见 [`qa/op_dispatch.md`](./qa/op_dispatch.md) §3）。

**合并冲突预期**：v1 加算子的冲突面是 4 个热点文件（§2.1）；v2 收敛为 4 个一行式登记点（op.h / registry.cpp / ops CMakeLists / mycv.h）——冲突概率趋近于零，即使冲突也是单行级解决。
