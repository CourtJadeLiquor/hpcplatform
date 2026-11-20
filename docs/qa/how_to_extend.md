# 如何扩展：加算子、加后端、加 CPU 优化版本

> 相关代码：[`include/mycv/ops/op.h`](../../include/mycv/ops/op.h)、[`src/ops/op_registry.{h,cpp}`](../../src/ops/op_registry.cpp)、[`src/ops/op_common.h`](../../src/ops/op_common.h)、[`include/mycv/backends/backend.h`](../../include/mycv/backends/backend.h)、[`src/backends/backend_factory.cpp`](../../src/backends/backend_factory.cpp)、[`include/mycv/core/kernel_level.h`](../../include/mycv/core/kernel_level.h)
> 三条扩展轴的共同契约：**核心层（core/）不动**。算子和后端都只往注册表里"加键"，不修改分发逻辑本身。设计动机见 [`../architecture/P0.md`](../architecture/P0.md) 的"扩展契约"。

---

## 1. 新增一个算子

算子的本质是往三元组表 `(OpKind, BackendType, CpuKernelLevel) → OpKernelFn` 里塞一行（[`op_registry.cpp`](../../src/ops/op_registry.cpp#L28-L33)）。下面以"加一个新算子 Foo"为序，列出必须动的文件和顺序。

### 1.1 步骤清单

| # | 动作 | 文件 | 关键点 |
| --- | --- | --- | --- |
| 1 | 追加 `OpKind` 枚举值 | [`include/mycv/ops/op.h`](../../include/mycv/ops/op.h#L20-L29) | **只能追加，不能重排、不能改已有数值**——`OpKind` 的数值是稳定 id，未来可能被序列化/日志/跨版本对照。 |
| 2 | 追加 `OpKindName` 的 `case` | [op.h#L31-L43](../../include/mycv/ops/op.h#L31-L43) | 漏了会让错误信息打印 `unknown`。 |
| 3 | 写公共头 | `include/mycv/ops/foo.h` | 声明 `Result<Tensor> Foo(const OpContext&, ...)`。照抄 [`add.h`](../../include/mycv/ops/add.h) 的形状：只依赖 `core/error.h`、`core/tensor.h`、`ops/op.h`。 |
| 4 | 写实现目录 | `src/ops/foo/foo.cpp` + `src/ops/foo/foo_cpu_naive.cpp` | `foo.cpp` 放公共函数 + `RegisterFooKernels()`；`foo_cpu_naive.cpp` 放可移植参考内核。 |
| 5 | 在注册表声明并调用 `RegisterFooKernels()` | [`src/ops/op_registry.cpp`](../../src/ops/op_registry.cpp#L11-L54) | 前向声明加在 #L11-L24 区；调用加进 `EnsureRegistered()` 的 `std::call_once` lambda（#L37-L54）。 |
| 6 | 把源文件加进 `mycv` 静态库 | 根 [`CMakeLists.txt`](../../CMakeLists.txt#L38-L66) | 在 `add_library(mycv STATIC ...)` 的源文件列表里加 `src/ops/foo/foo.cpp` 和 `src/ops/foo/foo_cpu_naive.cpp`。 |
| 7 | 加入伞形头 | [`include/mycv/mycv.h`](../../include/mycv/mycv.h#L18-L28) | 加 `#include "mycv/ops/foo.h"`，让 `#include <mycv/mycv.h>` 的用户拿到它。 |
| 8 | 写测试 | `tests/unit/test_foo.cpp` + 在对拍里注册 | 见第 4 节。 |

### 1.2 `foo.cpp` 的固定骨架

每个算子公共函数都长一个样，以 [`Add`](../../src/ops/add/add.cpp#L46-L58) 为模板：

```
1. ValidateInvocation(context, input)        ← 复用，别重写
2. (算子专属校验)                             ← 返回 kInvalidArgument / kUnsupported
3. FindKernel(OpKind::kFoo, input.backend(), context.cpu_level)
4. AllocateOutput(context, out_shape, out_dtype)
5. kernel(context, input, &other, &params, &output)
6. return output
```

- 第 1 步 [`ValidateInvocation`](../../src/ops/op_common.h#L13-L31) 已经把"ctx 非空 / input 已定义 / input 与 ctx 同后端同设备 / 非 CPU 后端只允许 kNaive"全做了，**不要在算子里重复校验这些**。
- 第 4 步 [`AllocateOutput`](../../src/ops/op_common.h#L33-L36) 就是 `context->AllocateTensor(...)`，输出分配在 OpContext 的后端上。
- 错误码选择（`kInvalidArgument` vs `kUnsupported` vs `kNotImplemented`）见 [`op_dispatch.md`](./op_dispatch.md) §3 和 [`error_model.md`](./error_model.md)。

### 1.3 带参数的算子怎么办

无参算子（Add/Mul）调用内核时 `params` 传 `nullptr`；有参算子（Normalize/Resize/Crop/WarpAffine）在栈上构造一个参数结构体，把地址传进去：

```cpp
const ops_internal::NormalizeParams params{&mean, &std, epsilon};
kernel.value()(context, input, nullptr, &params, &output.value());
```

见 [`normalize.cpp#L77-L79`](../../src/ops/normalize/normalize.cpp#L77-L79)。约定：

- 参数结构体定义在 [`src/ops/op_params.h`](../../src/ops/op_params.h)（`NormalizeParams`、`ResizeParams`、`CropParams` 都在这），是 `ops_internal` 可见的内部类型，**不暴露到公共头**。
- 内核签名统一是 `Status(*)(const OpContext&, const Tensor& input, const Tensor* other, const void* params, Tensor* output)`（[`op_registry.h#L15-L17`](../../src/ops/op_registry.h#L15-L17)）。`params` 只在这次同步调用期间有效，内核别存它的地址。
- 公共函数先把参数校验完（shape/范围/有限性），再构造 `params`，这样内核拿到的一定是合法参数。

### 1.4 `foo_cpu_naive.cpp`：参考内核必须可移植

`kNaive` 是**全库的真值基准**，所有优化版本都要和它对拍位精确（见第 3 节）。所以 naive 内核：

- 只用标准 C++，不用任何平台 intrinsics、不依赖 `-mavx2` 之类编译标志。
- 能复用公共骨架就复用：Add/Mul 走 [`cpu_binary.h`](../../src/ops/cpu_binary.h) 的 `BinaryNaive`，见 [`add_cpu_naive.cpp`](../../src/ops/add/add_cpu_naive.cpp)。新算子如果是同类（逐元素二元），优先加进骨架而不是各写一份。

---

## 2. 新增一个后端

后端的本质是实现 [`Backend`](../../include/mycv/backends/backend.h#L18-L44) 接口，再往 `CreateBackend` 的 switch 里加一个 `case`。核心层、算子层都不需要改。

### 2.1 步骤清单

| # | 动作 | 文件 | 关键点 |
| --- | --- | --- | --- |
| 1 | 追加 `BackendType` 枚举值 | [`include/mycv/core/device.h`](../../include/mycv/core/device.h#L12-L17) | 同样**只能追加**。它同时是三元组分发键的一部分。 |
| 2 | 新建后端目录 | `src/backends/<name>/` | 放 `xxx_backend.{h,cpp}`、内存管理器、流、内核等。 |
| 3 | 实现 `Backend` 接口 | `src/backends/<name>/xxx_backend.cpp` | 见下。 |
| 4 | 写工厂函数 | `xxx_backend.cpp` | `Result<std::shared_ptr<Backend>> Create<Name>Backend(int device_index)`。 |
| 5 | 声明工厂 | [`src/backends/backends_internal.h`](../../src/backends/backends_internal.h#L12-L15) | 加一行 `Create<Name>Backend`。 |
| 6 | 接进总工厂 | [`src/backends/backend_factory.cpp`](../../src/backends/backend_factory.cpp#L8-L29) | 加一个 `case`，用 `#ifdef MYCV_WITH_<NAME>` 包住：编译进来就调用，没编译就返回 `kUnsupported("<name> backend is not built")`（照抄 OpenCV/CUDA 的写法）。 |
| 7 | CMake 选项与源文件 | 根 [`CMakeLists.txt`](../../CMakeLists.txt#L147-L172) | 加 `option(MYCV_WITH_<NAME> OFF)`；`find_package` 厂商 SDK；`target_sources` 加后端源文件；`target_compile_definitions(mycv PUBLIC MYCV_WITH_<NAME>=1)`。 |
| 8 | 注册内核 | 新建 `Register<Name>Kernels()`，在 [`op_registry.cpp`](../../src/ops/op_registry.cpp#L19-L24) 前向声明并在 `EnsureRegistered` 的 `#ifdef` 块里调用 | 非 CPU 后端**只能注册 `CpuKernelLevel::kNaive`**——其他 level 在查表前就会被拒（[`op_registry.cpp#L66-L71`](../../src/ops/op_registry.cpp#L66-L71)）。 |
| 9 | 写对拍测试 | `tests/consistency/test_<name>_backend.cpp` | 仿照 [`test_cuda_backend.cpp`](../../tests/consistency/test_cuda_backend.cpp)，在 [`tests/CMakeLists.txt`](../../tests/CMakeLists.txt#L20-L32) 里用 `if(MYCV_WITH_<NAME>)` 包住。 |

### 2.2 `Backend` 接口哪些必须实现，哪些可以不碰

[`backend.h#L18-L44`](../../include/mycv/backends/backend.h#L18-L44)：

| 方法 | 是否必须 | 说明 |
| --- | --- | --- |
| `type()` / `device()` / `memory()` / `stream()` / `Synchronize()` | 必须纯虚 | 后端身份、设备、内存管理器、默认流、同步。 |
| `SupportsKernelLevel(level)` | 必须纯虚 | 非 CPU 后端：`return level == CpuKernelLevel::kNaive;`。 |
| `Describe()` | 必须纯虚 | 一行可读描述，用于日志。 |
| `CopyBuffer(...)` | **有默认实现**（基类） | 默认实现会对主机可达内存做拷贝。若你的后端设备内存主机碰不到，**必须 override 并返回 `kUnsupported`**——绝不能偷偷经主机中转，否则会破坏 `Context::Copy` 的语义契约（见 [`backend.h#L28-L33`](../../include/mycv/backends/backend.h#L28-L33) 和 [`context_copy.md`](./context_copy.md)）。 |
| `SupportsDType(dtype)` | 有默认实现 | 默认"只要 `DTypeSize != 0` 就支持"。如果你的后端不支持 f16/NCHW 等，override 拒绝，**不要在算子里默默转换**（见 OpenCV 后端的做法，[`../architecture/P4.md`](../architecture/P4.md)）。 |

### 2.3 最小骨架长什么样

看 NPU 的占位实现——它目前只有工厂，返回 `kUnsupported`，但上层已经可以针对它写代码了：

```cpp
// src/backends/npu/npu_backend.cpp
Result<std::shared_ptr<Backend>> CreateNpuBackend(int device_index) {
  (void)device_index;
  return Status::Unsupported("npu backend is a compile-time placeholder ...");
}
```

见 [`npu_backend.cpp`](../../src/backends/npu/npu_backend.cpp)。注释里明确写了："换成真实 ACL/RKNN 实现时，改动不得超出 `src/backends/npu/`"——这就是后端可插拔的硬约束。

### 2.4 接进来之后别忘了

- **拷贝路径**：如果新后端能做设备内/跨设备拷贝，override `CopyBuffer`；不能就老实返回 `kUnsupported`，让调用方显式分两步（设备→主机→设备）。
- **外部内存可见性**：如果后端允许 `Wrap` 外部内存，务必在文档里写清同步责任——CUDA 就踩过这个坑，见 [`cuda_stream_visibility.md`](./cuda_stream_visibility.md)。

---

## 3. 新增一个 CPU 内核级别（优化版本）

给已有算子加一个更快的 CPU 实现，比如加一个 `kSimdAvx512` 或新的线程库版本。这是 P3 的工作，也是最容易"优化完结果不对"的地方。

### 3.1 步骤清单

| # | 动作 | 文件 | 关键点 |
| --- | --- | --- | --- |
| 1 | 追加 `CpuKernelLevel` 枚举值 | [`include/mycv/core/kernel_level.h`](../../include/mycv/core/kernel_level.h#L11-L18) | **只能追加**。 |
| 2 | 追加 `CpuKernelLevelName` 的 `case` | [kernel_level.h#L20-L30](../../include/mycv/core/kernel_level.h#L20-L30) | 漏了 `Describe()` 和错误信息会打 `unknown`。 |
| 3 | 每个算子加一个实现文件 | `src/ops/<op>/<op>_cpu_<level>.cpp` | 实现一个 `Status <Op>Cpu<Level>(...)`，签名同 `OpKernelFn`。 |
| 4 | 在每个算子的 `Register<Op>Kernels()` 里注册 | 如 [`add.cpp#L23-L42`](../../src/ops/add/add.cpp#L23-L42) | 用 `#ifdef MYCV_WITH_<LEVEL>`（或 ISA 宏）包住一行 `RegisterKernel(...)`。 |
| 5 | CMake：编译宏 + 源文件 + ISA 标志 | 根 [`CMakeLists.txt`](../../CMakeLists.txt) | 见下。 |
| 6 | 让能力被探测到 | （无需改代码） | 见 §3.3 的机制。 |
| 7 | 加进对拍 | [`tests/consistency/test_cpu_kernels.cpp`](../../tests/consistency/test_cpu_kernels.cpp#L22-L30) | 把新 level 加进 `AvailableLevels` 的枚举列表，位精确对拍自动覆盖。 |

### 3.2 CMake 怎么做才不污染整个库

关键原则：**绝不使用 `-march=native`**，否则编出来的静态库换台机器就非法指令崩溃。正确做法是**每个 ISA 内核一个独立翻译单元，单独给编译标志**：

- 线程库类（OpenMP/TBB）：整个库加编译宏 `MYCV_WITH_OPENMP=1`，并在 `if(MYCV_ENABLE_OPENMP)` 块里 `target_sources` 加对应文件（[`CMakeLists.txt#L92-L116`](../../CMakeLists.txt#L92-L116)）。
- ISA 类（AVX2）：只给那几个 TU 加标志——
  ```cmake
  set_source_files_properties(${MYCV_AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "-mavx2")
  ```
  见 [`CMakeLists.txt#L128-L145`](../../CMakeLists.txt#L128-L145)。这样库里只有这些文件能用 AVX2 指令，其余文件保持可移植。

### 3.3 "支持哪些 level"是怎么被知道的

这是个容易忽略的设计点：`CpuBackend::SupportsKernelLevel` **不是看编译宏，而是看注册表里有没有链接进来的内核**——

```cpp
return ops_internal::HasKernel(OpKind::kAdd, BackendType::kNativeCpu, level);
```

见 [`cpu_backend.cpp#L10-L14`](../../src/backends/cpu/cpu_backend.cpp#L10-L14)。它拿 `kAdd` 当探针：只要 Add 为某个 level 注册了内核，就认为整个库支持这个 level。

> 推论：**新增 level 时，必须给 Add 也注册这个 level 的内核**（哪怕 Add 在该 level 下只是复用 naive），否则 `SupportsKernelLevel` 返回 false，你给其他算子注册的内核永远不会被选中，对拍也不会跑到。这是个隐蔽的连锁要求。

### 3.4 位精确是硬要求，不是建议

CPU 各版本之间要求**逐位相同**，对拍直接用 `std::memcmp` 和 naive 基准比：

```cpp
MYCV_EXPECT_EQ(std::memcmp(added.value().data(), reference_add.data(),
                           reference_add.ByteSize()), 0);
```

见 [`test_cpu_kernels.cpp#L52-L53`](../../tests/consistency/test_cpu_kernels.cpp#L52-L53)。含义：

- 整数：必须位精确。
- 浮点：也要求位精确（不是"差不多"）。这意味着 SIMD/并行版本的归约顺序必须和 naive 完全一致——想换更快的归约顺序？先在 [`../architecture/P3.md`](../architecture/P3.md) 里走放宽位精确的决策流程，不能先斩后奏。
- 跨后端对拍才用容差（CPU↔CUDA 用 1e-4），那是另一条规则，见 [`testing.md`](./testing.md) §4。

---

## 4. 写测试与合并前自检

### 4.1 新算子/后端要补哪些测试

- **单元测试**（`tests/unit/`）：正常路径 + 非法参数（shape 不匹配、dtype 不支持、空 ctx）各一个，断言返回的错误码正确。
- **CPU 对拍**（[`tests/consistency/test_cpu_kernels.cpp`](../../tests/consistency/test_cpu_kernels.cpp)）：把新算子加进位精确循环；如果新 level 对某个 dtype/算子不支持，断言返回 `kUnsupported`（见 cast 的写法，[`test_cpu_kernels.cpp#L67-L77`](../../tests/consistency/test_cpu_kernels.cpp#L67-L77)）。
- **后端对拍**（`tests/consistency/test_<name>_backend.cpp`）：和 CPU naive 结果比，用容差。
- 在 [`tests/CMakeLists.txt`](../../tests/CMakeLists.txt) 里登记测试源文件，否则不会被编译运行。

### 4.2 合并前自检清单

对照 [`../architecture/P0.md`](../architecture/P0.md) 的扩展契约过一遍：

- [ ] 没有改动 `core/`（除非 P0 决策明确允许）。
- [ ] 没有改动分发逻辑 `op_registry.cpp` 的查表/错误码分支，只往注册表里加了条目。
- [ ] 没有静默降级：找不到内核返回 `kUnsupported`/`kNotImplemented`，不偷偷 fallback 到另一个实现。
- [ ] 没有静默转换：后端不支持的 dtype/layout 直接拒绝，不暗中改数据。
- [ ] `CopyBuffer` 不会偷偷经主机中转。
- [ ] 位精确（CPU 内）/ 容差（跨后端）在测试里有体现。
- [ ] 三棵构建树（default / full / asan）都能编过且测试通过，见 [`testing.md`](./testing.md)。
