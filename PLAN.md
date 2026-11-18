# mycv 算子库实施计划

> 版本：v0.1（P0–P5 已落地，P6 待实施）
> 定位：面向科研平台的跨平台算子库，先保证"能理解、能验证、能扩展"，再逐步走向生产。

---

## 1. 目标与非目标

### 1.1 目标

1. **一套 core 抽象**：`Tensor` / `Buffer` / `MemoryManager` / `Stream` / `Context` / `Backend`，屏蔽 CPU、OpenCV、CUDA、NPU 差异。
2. **后端可替换**：OpenCV 只是 `BackendType` 中的一个后端，不是依赖本体；CUDA 同理。
3. **统一内存管理**：对上只暴露 `MemoryManager`，向下由各后端实现（对齐分配、`cv::Mat`、`cudaMalloc`、设备内存池）。
4. **CPU 多版本算子**：同一算子提供 `naive / OpenMP / SIMD` 多实现，由 `CpuKernelLevel` 枚举显式选择。
5. **静态编译、显式选择**：后端在编译期开关，运行期由调用方指定；不做平台探测、不做自动降级。
6. **工程完备**：CMake 构建、单元测试、一致性测试、samples、后续 benchmark，目录结构按可扩展方式组织。

### 1.2 明确不做（MVP 边界）

| 不做 | 原因 |
| --- | --- |
| 自动平台/后端识别 | 与科研核心诉求无关，行为不可预期 |
| 运行时自动降级（如 CUDA 失败回退 CPU） | 同上，且会让实验结论不可复现 |
| 算子融合、图编译、自动微分 | 属于更高层框架，不放进算子库 |
| 复刻 OpenCV 全量 API | OpenCV 只是后端之一 |
| Python 绑定（现阶段） | 留到 P6 之后以 C ABI 形式补 |
| 分布式 / 多机 | 超出库的职责 |

---

## 2. 核心设计决策

| # | 决策 | 选择 | 理由 |
| --- | --- | --- | --- |
| D1 | 后端如何选择 | `Context::Create(BackendType, device_index)` 显式指定，失败返回 `kUnsupported` | 无隐式回退，实验可复现 |
| D2 | Backend 与 Device 的关系 | 两者分离：`BackendType{kNativeCpu,kOpenCV,kCUDA,kNPU}`，`DeviceType{kCPU,kGPU,kNPU}`；`DefaultDeviceType()` 只做静态映射 | OpenCV 是 CPU 上的后端，CUDA 是 GPU 后端，混为一谈会让调度层写死 |
| D3 | Tensor 的内存模型 | 描述符（shape/dtype/strides/offset）+ `shared_ptr<Buffer>` | 支持零拷贝 View/Slice/Permute；非连续 Reshape 显式深拷贝 |
| D4 | 内存释放 | `Buffer` 持有一个由后端注入的 deleter；分配统计放在后端内部的 shared state | Tensor 可以比 Context 活得久，且便于后续接内存池 |
| D5 | 错误处理 | `Status` + `Result<T>`，core 不抛异常 | 部分 NPU/嵌入式工具链以 `-fno-exceptions` 编译 |
| D6 | 多版本算子选择 | `OpContext.cpu_level` 显式指定；非 CPU 后端只接受 `kNaive` | 避免"以为在跑 AVX2，其实跑的是 naive" |
| D7 | 编译策略 | 静态库 `libmycv.a`；x86 AVX2 内核使用独立编译单元并单独加 `-mavx2`；不默认 `-march=native` | 跨平台分发时不会因指令集导致非法指令 |
| D8 | 对齐 | 默认 64 字节，接口可传 | 兼顾 AVX-512（64B）与 cache line |
| D9 | 测试框架 | 零依赖 mini harness，注册式 | 科研环境常离线；后续可无痛换 gtest |

### 2.1 关键接口（已落地）

```cpp
// 显式后端选择，无自动探测
auto ctx = mycv::Context::Create(mycv::BackendType::kNativeCpu).value();

// 统一分配：任何后端都走 MemoryManager
auto t = ctx->AllocateTensor(mycv::Shape{2, 3}, mycv::DataType::kFloat32).value();

// 统一调度描述符
mycv::OpContext op_ctx;
op_ctx.context = ctx.get();
op_ctx.stream = &ctx->stream();
op_ctx.cpu_level = mycv::CpuKernelLevel::kOpenMP;
```

### 2.2 P1 已落地接口

```cpp
// 元数据视图：不拷贝，共享 Buffer
Result<Tensor> Tensor::View(const Shape& shape);                 // 元素数不变
Result<Tensor> Tensor::Slice(int dim, int64_t begin, int64_t end);
Result<Tensor> Tensor::Permute(std::initializer_list<int> order);
Result<Tensor> Tensor::Clone() const;                            // 真正深拷贝

// dtype 转换（f32<->u8、f32<->f16 软件实现）
Result<Tensor> Tensor::Cast(DataType dst_dtype) const;

// 跨后端/跨设备统一拷贝入口（按 host/device 内存类型选择路径）
Status Context::Copy(Tensor* dst, const Tensor& src, Stream* stream = nullptr);
```

---

## 3. 目录结构与职责

```
mycv/
├── CMakeLists.txt                 # 单一入口，后端开关都是 option()
├── PLAN.md
├── include/mycv/                  # 唯一对外可见区
│   ├── mycv.h                     # 伞形头
│   ├── version.h
│   ├── core/
│   │   ├── error.h                # Status / Result / 宏
│   │   ├── dtype.h
│   │   ├── shape.h                # 固定容量，无堆分配
│   │   ├── device.h               # BackendType / DeviceType / Device
│   │   ├── kernel_level.h         # CpuKernelLevel
│   │   ├── memory.h               # Buffer / MemoryManager / MemoryStats
│   │   ├── stream.h
│   │   ├── tensor.h
│   │   └── context.h
│   ├── backends/backend.h         # Backend 接口 + 显式工厂
│   └── ops/op.h                   # OpContext / OpKind（稳定 id，只追加）
├── src/
│   ├── core/                      # 目前：error/dtype/shape/tensor/context
│   ├── backends/
│   │   ├── backend_factory.cpp    # 唯一的 switch 分发点
│   │   ├── cpu/                   # 已实现：backend / memory / stream
│   │   ├── opencv/                # 已实现：backend / memory / convert / kernels
│   │   ├── cuda/                  # 已实现：backend / memory / stream / copy / kernels
│   │   └── npu/                   # 已实现占位，返回 kUnsupported
│   └── ops/                       # 每个算子一个目录
│       └── <op>/                  # 例：add/
│           ├── add.cpp                # 公共校验与注册分发
│           ├── add_cpu_naive.cpp
│           ├── add_cpu_omp.cpp
│           ├── add_cpu_tbb.cpp
│           ├── add_cpu_simd.cpp
│           └── add_cpu_avx2.cpp       # 单独加 -mavx2
├── tests/
│   ├── test_framework.h           # 零依赖 harness
│   ├── unit/                      # 类型/内存/后端/每个算子
│   ├── consistency/               # 多实现、多后端对拍（P3/P4/P5）
│   └── CMakeLists.txt
├── samples/
│   ├── 00_hello_tensor.cpp
│   ├── 01_memory_copy.cpp
│   ├── 02_ops_pipeline.cpp
│   ├── 03_opencv_backend.cpp
│   └── 04_cuda_pipeline.cpp
├── benchmarks/                    # P3 起，不进入 ctest 门禁
└── cmake/                         # P6 补齐 find_package 导出 config
```

**扩展规则**（写进 code review checklist）：

1. 新增后端 = 新增 `src/backends/<name>/` + 在 `backend_factory.cpp` 加一个 `case`，不改 core。
2. 新增算子 = 新增 `include/mycv/ops/<op>.h` + `src/ops/<op>/` + `OpKind` 追加 id，不改已有算子。
3. 新增 CPU 优化版本 = 新增一个 `*_<level>.cpp` + 在 dispatch 表加一行，不改 naive 实现。
4. core 层不允许 `#include` 任何第三方头（OpenCV/CUDA/厂商 SDK 只能出现在对应 backend 目录）。

---

## 4. 算子矩阵（MVP 范围）

图例：`✅` 已实现并有对拍测试 / `–` 未提供该版本 / `P6` 未来规划。请求未提供的 CPU level 返回 `kUnsupported`；OpenCV/CUDA 传入非 `kNaive` 返回 `kInvalidArgument`，均不静默换实现。

| 算子 | CPU naive | OpenMP | TBB | Generic SIMD | AVX2 | OpenCV | CUDA | NPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Add` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | P6 |
| `Mul` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | P6 |
| `Cast` | ✅ | – | – | ✅ | ✅ | ✅ | ✅ | P6 |
| `Normalize` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | P6 |
| `Resize` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | P6 |
| `Crop` | ✅ | – | – | – | – | ✅ | ✅ | P6 |
| `Transpose` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | P6 |
| `WarpAffine` | ✅ | – | – | – | – | ✅ | ✅ | P6 |

当前后端限制：

- OpenCV 仅支持 u8/f32/f64 与 HW/HWC/NHWC；f16/NCHW 明确报错，不做隐式布局转换。
- CUDA 支持非连续输入和 NHWC/NCHW 图像算子；每次公开算子调用在返回前同步，跨 GPU peer-to-peer 暂未实现。
- CUDA 编译必须显式指定 `CMAKE_CUDA_ARCHITECTURES`，本机验证值为 `89`。
- 当前仅验证 x86_64 AVX2；NEON/aarch64 尚未实现，不声明可用。

**每算子统一内核签名**（P2 落地）：

```cpp
using CpuKernelFn = Status (*)(const OpContext& ctx,
                               const Tensor& a, const Tensor& b,
                               Tensor* out);
```

`Crop` / `Transpose` 的 naive 版本同时充当"view 正确性"的 ground truth：先验证 view 的 stride 计算，再验证拷贝。

---

## 5. 分阶段任务

### P0 — 骨架与可验证构建（已完成）

**任务**

- [x] core 类型：`Status/Result`、`Shape`、`DataType`、`Device/BackendType`、`CpuKernelLevel`
- [x] `Buffer` / `MemoryManager` / `Stream` / `Backend` / `Context`
- [x] CPU 后端：64B 对齐分配、拷贝、统计
- [x] NPU 后端占位（返回明确 `kUnsupported`）
- [x] CMake：静态库、OpenMP 开关、SIMD 能力探测、install 规则
- [x] 4 个单元测试 + 2 个 sample

**验收（已实际执行）**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure     # 4/4 passed
./build/samples/00_hello_tensor
./build/samples/01_memory_copy
```

> 本机注意：IDE 注入的 `LD_LIBRARY_PATH` 会让 cmake/二进制加载到旧版 libstdc++，执行前用 `env -u LD_LIBRARY_PATH` 清理。

---

### P1 — core 类型完备（已完成）

**任务**

1. `Tensor::View / Slice / Permute / Reshape / Clone`，严格维护 stride + offset。
2. `Tensor::Cast` 与 `Context::Copy`（host/host、H2D、D2H、D2D 显式路径）。
3. `TensorIterator`：按 stride 遍历任意布局，供 elementwise 与归一化复用。
4. 内存：CPU/CUDA free-list 池复用 `MemoryManager` 接口；统计活跃 `allocated_bytes`、`peak_bytes` 和 `allocation_count`，缓存保留量不计入活跃分配。
5. 生命周期测试：Tensor 比 Context 长寿、deleter 不悬垂（沿用 P0 的 shared state 方案）。
6. CMake 增加 `MYCV_ENABLE_SANITIZERS`（ASAN/UBSAN）。

**验收**

- `ctest` 全绿，新增 view/slice/cast/跨后端拷贝用例。
- ASAN + UBSAN 下全部测试通过（`-DMYCV_ENABLE_SANITIZERS=ON`）。
- 非连续输入（slice 后的 tensor）能正确参与 `Context::Copy`。
- `Tensor` 在 `Context` 析构后仍可安全释放。

**退出条件**：core 层 API 冻结（之后只允许追加），P2 可以并行开工。

---

### P2 — CPU naive 参考实现 + 分发框架（已完成）

**任务**

1. `src/ops/op_registry.{h,cpp}`：`(OpKind, BackendType, CpuKernelLevel) -> KernelFn` 表；后端未实现算子返回 `kNotImplemented`，CPU level 未注册返回 `kUnsupported`，**不做回退**。
2. 统一参数校验 helper：shape 匹配、dtype 支持表、连续性要求。
3. 8 个算子的 naive 实现（单文件/算子，可读性优先，作为 ground truth）。
4. `samples/02_ops_pipeline.cpp`：`Resize -> Normalize -> Transpose -> Cast` 的预处理链。
5. 测试：每个算子 3 类用例（手算小例、随机对比、边界：单元素/非连续/dtype 不支持）。

**验收**

- 8 个算子在 `kNaive` 下全部通过测试。
- 未实现组合返回 `kNotImplemented`，且错误信息包含 op 名与 level。
- 样例输出可人工核对。
- `-Wall -Wextra -Wpedantic` 零警告。

---

### P3 — CPU 多版本优化（x86_64 已完成）

**任务**

1. `Add/Mul/Normalize/Resize/Transpose` 补 `kOpenMP` 与 `kTBB`，线程数由调用方控制，库内不写死。
2. `Add/Mul/Cast/Normalize` 补 `kSimdGeneric`：编译器向量化友好写法，保持可移植。
3. `Add/Mul/Cast/Normalize` 补 `kSimdAvx2`：单独 TU + `set_source_files_properties(... COMPILE_OPTIONS "-mavx2")`。
4. `CpuBackend::SupportsKernelLevel` 由编译期宏与已注册内核共同决定，避免虚报能力。
5. 一致性测试：naive vs OpenMP/TBB/SIMD/AVX2 对拍；`benchmarks/` 输出 CSV（耗时、GiB/s、加速比）。

**验收**

- 元素级算子（Add/Mul/Cast/Normalize）多版本 **bit-exact**。
- 插值类（Resize/WarpAffine）误差 ≤ 1e-5（相对），u8 允许 ±1。
- 请求未编译的 level 返回 `kUnsupported`，绝不静默换实现。
- `benchmarks/operator_benchmark` 实际运行并输出 CSV（耗时、GiB/s、加速比），性能只作观测记录，不设硬门禁。

---

### P4 — OpenCV 后端（已完成）

**任务**

1. `option(MYCV_WITH_OPENCV OFF)`；开启时 `find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)`。
2. `OpenCvMemoryManager`：`Buffer` 持有一个 `cv::Mat`（通过 deleter 释放），保证 Tensor/Mat 同生命周期。
3. dtype 映射：u8/f32/f64 支持；f16 返回 `kUnsupported`。
4. layout 约束：只承诺 `HW`/`HWC`/`NHWC`；`NCHW` 明确拒绝（需要时由调用方显式 Transpose），不偷偷转换。
5. 算子映射：`cv::add / multiply / resize / warpAffine / transpose / convertTo`，`Normalize` 用 split+convertTo 组合。
6. `Context::Copy` 支持 `native_cpu <-> opencv`（同 host，直接 memcpy）。
7. `OpContext.cpu_level != kNaive` 时返回 `kInvalidArgument`（OpenCV 后端不提供多版本语义）。

**验收**

- 未安装 OpenCV 时，默认构建与所有测试不受影响。
- 开启后 `ctest` 全绿，native vs OpenCV 按第 6 节跨后端容差对拍通过；整数 Cast 保持截断并精确一致。
- 同一 `Context` 内多次分配/释放后 `MemoryStats` 归零。

---

### P5 — CUDA 后端（已完成）

**任务**

1. `option(MYCV_WITH_CUDA OFF)`；开启时 `enable_language(CUDA)` + `find_package(CUDAToolkit)`，且必须显式传入 `CMAKE_CUDA_ARCHITECTURES`。
2. `CudaMemoryManager`：`cudaMalloc/cudaFree` + caching pool；`cudaMemcpyAsync`；统计与 CPU 实现对齐。
3. `CudaStream`：non-blocking stream + `Synchronize`；`synchronous() == false`。
4. 跨设备拷贝：`Context::Copy` 覆盖 H2D / D2H / D2D；不同 GPU 间拷贝明确返回 `kUnsupported`。
5. 内核：Add/Mul/Cast/Normalize、Resize、Crop、Transpose、WarpAffine；`CudaStatus` 统一错误包装。
6. 测试按设备可用性跳过（无 GPU 时 `ctest` 仍全绿，输出 SKIP 原因）。
7. 非 `kNaive` 的 `cpu_level` 返回 `kInvalidArgument`，不忽略参数。

**验收**

- GPU 可用机器上：8 个算子与 CPU naive 按跨后端容差对拍通过。
- H2D→kernel→D2H 往返及完整预处理流水线正确。
- 默认不开启 CUDA 的独立构建与 9 个测试全部通过；无 GPU 时测试目标输出 SKIP 原因。

---

### P6 — NPU 后端与交付化

**任务**

1. 确定目标芯片（昇腾 ACL / RKNN / 其他）后实现 `CreateNpuBackend`，只改 `src/backends/npu/`。
2. 明确 NPU 的能力边界：单算子调用（可行则做）与整图后端（另设 `GraphBackend`，不进 `Backend` 接口）。
3. 导出 `mycvConfig.cmake` + `mycvTargets.cmake`，支持 `find_package(mycv)`。
4. CI 矩阵：{gcc, clang} × {x86_64, aarch64} × {OpenMP on/off} × {tests on}。
5. 可选：`mycv_c_api.h` 稳定 C ABI，供 Python / 其他语言绑定。

**验收**

- NPU 后端可创建 Context 并完成至少 1 个算子（或明确文档化不支持项）。
- 三个平台的静态库可被外部工程 `find_package` 直接链接。
- CI 矩阵全绿。

---

## 6. 测试策略

| 层次 | 内容 | 位置 | 门禁 |
| --- | --- | --- | --- |
| 单元测试 | 类型/内存/后端/每个算子边界 | `tests/unit/` | ctest 必过 |
| 一致性测试 | naive vs OMP vs SIMD；native vs OpenCV/CUDA | `tests/consistency/` | ctest 必过 |
| 黄金数据 | 固定输入 + 固定期望输出（防回归） | `tests/consistency/data/` | ctest 必过 |
| 内存检查 | ASAN/UBSAN/LSAN | `-DMYCV_ENABLE_SANITIZERS=ON` | P1 起纳入 CI |
| 性能 | 吞吐/带宽/加速比，输出 CSV | `benchmarks/` | 仅记录，不阻断 |

**容差约定**

- 整数类型：bit-exact。
- 元素级浮点：bit-exact（同 level 内）/ 1e-6 相对（跨实现）。
- 插值类：1e-5 相对；u8 输出允许 ±1。
- 跨后端：1e-4 相对。

**命名约定**：`test_<op>_<aspect>.cpp`，例如 `test_add_non_contiguous.cpp`。

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| OpenCV `cv::Mat` 与 Tensor 所有权混淆 | 悬垂指针 / 双重释放 | `Buffer` 独占持有 `cv::Mat`，禁止裸指针逃逸；ASAN 覆盖 |
| `cv::Mat` 不支持任意 stride/NCHW | 静默错误结果 | 支持矩阵写死在头文件注释；不满足返回 `kUnsupported` |
| CUDA 内存池生命周期早于 Context | 释放崩溃 | 与 CPU 相同：池状态用 `shared_ptr` 与 Buffer 共享 |
| 多版本内核代码重复 | 维护成本 | 抽公共循环骨架 `kernel_common.h`，各 level 只写内层循环 |
| SIMD 浮点结果与 naive 不一致 | 对拍失败 | 元素级保持 bit-exact；插值类改用容差 |
| NPU 只支持整图 | 接口被污染 | `Backend` 只承诺单算子/内存；图能力另设接口 |
| 静态库链接 OpenMP/CUDA 顺序问题 | 下游集成失败 | 统一以 `mycv::mycv` imported target 交付 |
| C++ ABI 不稳定 | 跨语言集成困难 | P6 追加 C ABI，不暴露 STL 类型 |

---

## 8. 里程碑与完成定义（DoD）

| 阶段 | 状态 | DoD |
| --- | --- | --- |
| P0 骨架 | ✅ 完成 | 构建、核心测试与基础 samples 通过 |
| P1 core 完备 | ✅ 完成 | view/cast/copy 测试全绿 + ASAN/UBSAN 通过 |
| P2 naive 算子 | ✅ 完成 | 8 算子测试通过 + 错误路径明确 + 预处理链样例 |
| P3 多版本 | ✅ 完成 | naive/OpenMP/TBB/SIMD/AVX2 对拍通过 + 未支持 level 明确报错 + CSV 基准可运行 |
| P4 OpenCV | ✅ 完成 | 开关关闭零影响；开启后 8 算子一致性、互拷和内存生命周期测试全绿 |
| P5 CUDA | ✅ 完成 | RTX 4060 Ti / sm_89 上 8 算子 GPU 对拍、拷贝和流水线通过；默认构建不受影响 |
| P6 NPU/交付 | 待开始 | 确定厂商 SDK；至少 1 算子跑通 + `find_package` 可用 + CI 矩阵全绿 |

---

## 9. 下一步（P6）

1. **先确定 NPU 厂商与 SDK**：再实现 `src/backends/npu/`，不预设昇腾、RKNN 或其他厂商接口。
2. **补交付化能力**：增加 `mycvConfig.cmake`、外部工程链接验证和 CI 编译矩阵。
3. **保持执行语义明确**：当前 CUDA 公开算子在返回前同步；未来若增加异步流水，需另行定义输出生命周期与同步契约，不做隐式行为切换。

---

## 10. 当前实际验证结果（2026-11-18）

完整配置使用 CUDA 13.5、RTX 4060 Ti（compute capability 8.9）、OpenMP、TBB 和 OpenCV：

```bash
env -u LD_LIBRARY_PATH cmake -S . -B build \
  -DMYCV_WITH_OPENCV=ON \
  -DMYCV_WITH_CUDA=ON \
  -DMYCV_ENABLE_TBB=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda
env -u LD_LIBRARY_PATH cmake --build build -j 12
env -u LD_LIBRARY_PATH ctest --test-dir build --output-on-failure
```

实测结果：

```text
完整后端构建：11/11 tests passed
默认无可选后端构建：9/9 tests passed
ASAN/UBSAN（CPU + OpenCV）：10/10 tests passed
CUDA 设备测试：7/7 tests passed，覆盖全部 8 个 CUDA 算子
Samples 00–04：全部运行成功
CPU benchmark：Add AVX2 23.1970x，Mul AVX2 22.8248x（相对 naive，本次本机观测）
```

### 10.1 图像处理链路基准（`benchmarks/image_pipeline_benchmark.cpp`）

链路为 `Resize(384x512) -> Normalize -> WarpAffine(448x448, 12°/0.9) -> Crop(384x384) -> Transpose(3,384,384 f32)`，输入 768x1024x3 u8，仅在开头上传一次；3 次预热后计时 20 次。

- **缓存复用**：每轮中间结果在返回时释放，CPU 分配器改为按精确字节数缓存空闲块（与 CUDA 池语义一致），基准用“连续两轮 5 个 stage 缓冲区地址完全相同”验证，而不是假设。
- **不静默降级**：无法跑完整链路的后端/level 打印 `# skip <backend>/<level>: <原因>`；当前 `native_cpu/openmp`、`native_cpu/tbb` 因 `warp_affine` 只有 naive 实现而被跳过（Crop 同理）。
- **跨后端正确性**：逐 stage 比较，并报告首个偏离的算子；判据为“平均绝对误差 / 该 stage 值域 ≤ 0.1%”，最大值作为诊断输出。

本机实测（Release，clang++ −O3）：

```text
backend     level            ms/frame      img/s   speedup  peak MiB   cache   match
native_cpu  naive             161.402      6.196     1.000    12.422     yes     yes
opencv      naive               4.877    205.045    33.095    12.422     yes     yes
cuda        naive               1.206    829.503   133.884    12.422     yes     yes

逐级耗时（native naive）：resize 110.350 / normalize 2.419 / warp_affine 43.528 / crop 2.153 / transpose 2.944 ms
OpenCV 相对 native：mean_rel 6.5e-5，max_abs 17.87（仅旋转边界处的刀锋像素）
```

注：`native_cpu` 的 naive 是使用 `long double` 的参考实现，其耗时对编译器敏感——同一源码用 g++ −O3 构建时该链路约 54 ms（resize 约 31 ms），clang++ −O3 约 161 ms。因此跨构建树的绝对耗时不可直接比较，加速比只在同一构建内有效。同一机器上 OpenCV/CUDA 的计时抖动可达 ±30%，趋势判断应以多次运行的中位数为准。

