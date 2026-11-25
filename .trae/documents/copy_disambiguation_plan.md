# 消除 Copy 命名歧义：删除死代码 MemoryManager::Copy 实现计划

## 仓库调研结论

用户指出两个不同层面却同名的 `Copy`：

- [`include/hpcplatform/core/memory.h:78`](../../include/hpcplatform/core/memory.h) —— `MemoryManager::Copy`（字节/Buffer 层）
- [`include/hpcplatform/core/transfer.h:41`](../../include/hpcplatform/core/transfer.h) —— 自由函数 `Copy`（张量/语义层）

全仓检索后发现，字节层实际上存在**两个**近乎同签名的原语，其中一个是死代码：

| 维度 | 自由函数 `Copy`（张量层） | `MemoryManager::Copy`（字节层，公共头） | `Backend::CopyBuffer`（字节层，内部） |
| --- | --- | --- | --- |
| 声明位置 | `include/.../transfer.h`（公共） | `include/.../memory.h`（公共头，但外部拿不到实例） | `src/core/backend.h`（**不进公共树**） |
| 操作对象 | `const Tensor&` → `Tensor*` | `Buffer*` / `const Buffer*` + offset + bytes | `Buffer*` / `const Buffer*` + offset + bytes |
| 搬运内容 | **逻辑元素**，按 stride 读/写 | 裸字节区间 | 裸字节区间 |
| 深 / 浅拷贝 | 深拷贝元素字节；`dst` 须预分配，**不拷贝元数据**（shape/dtype/stride 由调用方负责） | 深拷贝字节 | 深拷贝字节 |
| 是否要求同设备 | 同设备 D2D 支持；跨设备 P2P → `kUnsupported` | 要求两 buffer 都属本 manager（同设备）；CPU 实现遇 device → `kUnsupported` | 各后端自判：CUDA 要求至少一侧 device，跨 device index → `kUnsupported` |
| 是否要求连续内存 | **不要求**（host 非连续走 `TensorIterator` 逐元素）；但 device 非连续 → `kUnsupported`（须先 `Clone()`） | 不感知 stride，只按字节区间，调用方保证区间合法 | 不感知 stride，只按字节区间 |
| dtype / shape 感知 | 是（校验 shape 与 dtype 匹配） | 否 | 否 |
| 同步契约 | 同步，返回即可读（CUDA 内部 `cudaMemcpyAsync` + `cudaStreamSynchronize`） | CPU 同步；CUDA 委托 `CudaCopyBuffer`，同步 | 同步 |
| 执行者 | 由张量 owning backend **推断**，调用方不传 Context | 调用方持有的 manager | 调用方持有的 backend |
| 实际调用者 | 大量：tests / samples / benchmarks / `Upload` / `Download` / `Tensor::Clone` | **零**（全仓无任何调用点，包括测试） | `CopyTensorData`（[copy.cpp:119](../../src/core/copy.cpp)、[copy.cpp:140](../../src/core/copy.cpp)）、`opencv_convert.cpp` 的 H2D/D2H staging（[L72](../../src/backends/opencv/opencv_convert.cpp)、[L122](../../src/backends/opencv/opencv_convert.cpp)） |
| 状态 | 活跃，唯一对外搬运原语 | **死代码 / 冗余接口** | 活跃，真实字节层工作马 |

关键事实：

1. `MemoryManager::Copy` 在整个仓库（含 `tests/`、`src/backends/*`）**没有任何调用者**。`Context::memory()` 是 `private`（[context.h:75-76](../../include/hpcplatform/core/context.h)），`backend_ref()` 返回的 `Backend` 在公共头里只有前向声明，外部消费者根本拿不到 `MemoryManager&` 也调不到它的方法——所以它既无内部调用者，也无外部可达性。
2. 真正干活的字节层是 `Backend::CopyBuffer`：`CopyTensorData` 和 OpenCV 转换都直接调 `executor->CopyBuffer(...)`，**从不经过** `MemoryManager::Copy`。`Backend::CopyBuffer` 的默认实现自己做 `memmove`，`CudaBackend::CopyBuffer` 直接调 `CudaCopyBuffer`，都**不委托**给 `memory().Copy()`。
3. 因此 `MemoryManager::Copy` 与 `Backend::CopyBuffer` 是**两套并行的字节拷贝机制**，前者完全冗余。它出现在公共头 `memory.h` 里，才让人误以为与张量层 `Copy` 是"两个同层同语义的 Copy"。
4. 文档 [`docs/qa/context_copy.md`](../../docs/qa/context_copy.md) 的四层模型（L0 `Backend::CopyBuffer` / L1 `CopyTensorData` / L2 `Copy`+`Upload`/`Download` / L3 `From`/`Into`）**已经把字节层唯一原语定为 `CopyBuffer`**，从未提及 `MemoryManager::Copy`，说明文档侧的真实设计意图就是如此——`MemoryManager::Copy` 是历史残留。

用户关心的异同点一句话总结：**张量层 `Copy` 搬的是"逻辑元素"（stride/shape/dtype 感知、深拷贝、不要求连续、推断执行者），字节层 `MemoryManager::Copy` 搬的是"裸字节"（无 dtype/shape/stride 概念、纯区间深拷贝、同设备、要求调用方保证连续区间）；但后者根本没人用，真正的字节层是 `Backend::CopyBuffer`。**

## 待修改文件与模块

- [`include/hpcplatform/core/memory.h`](../../include/hpcplatform/core/memory.h)：删除 `MemoryManager::Copy` 虚函数声明；更新类注释（"allocation / copy interface" → "allocation interface"）。
- [`src/backends/cpu/cpu_memory_manager.h`](../../src/backends/cpu/cpu_memory_manager.h) + [.cpp](../../src/backends/cpu/cpu_memory_manager.cpp)：删除 `CpuMemoryManager::Copy` override 及其实现。
- [`src/backends/cuda/cuda_memory_manager.h`](../../src/backends/cuda/cuda_memory_manager.h) + [.cpp](../../src/backends/cuda/cuda_memory_manager.cpp)：删除 `CudaMemoryManager::Copy` override 及其实现（其实现仅一行转调 `CudaCopyBuffer`，删除后不影响 `CudaBackend::CopyBuffer`）。
- [`src/backends/opencv/opencv_memory_manager.h`](../../src/backends/opencv/opencv_memory_manager.h) + [.cpp](../../src/backends/opencv/opencv_memory_manager.cpp)：删除 `OpenCvMemoryManager::Copy` override 及其实现。
- 文档：`docs/qa/context_copy.md`、`docs/architecture.md` 核对是否有对 `MemoryManager::Copy` 的引用（初步检索无），无需改动；如发现遗漏则同步。

## 实施步骤（依赖顺序）

1. **删除接口声明**：在 `memory.h` 移除 `virtual Status Copy(...) = 0;` 及其注释，并把类注释里的 "Unified allocation / copy interface" 改为 "Unified allocation interface"。
2. **删除三个后端 override**：依次删除 `cpu` / `cuda` / `opencv` 的 `MemoryManager::Copy` 声明（`.h`）与定义（`.cpp`）。`cuda_memory_manager.cpp` 中 `CudaCopyBuffer` 仍由 `CudaBackend::CopyBuffer`（`cuda_copy.cpp`）使用，**保留不动**。
3. **全文复核**：重新 `Grep` `MemoryManager.*Copy` / `\.Copy\(` / `->Copy\(`，确认无残留调用、无残留 override、无编译悬空引用；确认 `Backend::CopyBuffer` 路径完整。
4. **构建 + 回归测试**：`env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cmake --build build -j8`，再 `env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ctest --test-dir build -j4`。预期：除已知的 `test_load_operators`（构建时带 CUDA 导致 3 个子用例假设失败，与本次无关）外，其余全绿。
5. **文档一致性核对**：确认 `docs/qa/context_copy.md` 的四层模型与代码现状一致（字节层唯一原语 = `CopyBuffer`）。

## 依赖与注意事项

- **不破坏公共 ABI/API 的理由**：`MemoryManager::Copy` 是 `virtual`，删除会改变类布局，但（a）外部代码无法获取 `MemoryManager&`（`Context::memory()` 私有），（b）无内部调用者，（c）`MemoryManager` 无外部子类化路径（后端都编进库内，`Backend` 本身在 `src/` 不对外）。因此无源码或二进制破坏面。
- **版本号**：删除一个公共头里的虚函数属于 API 收窄。当前版本 `0.2.1`，按 semver 严格来说应升 minor（`0.3.0`）。但该函数既不可达又无调用者，实际影响为零；是否升版请用户定夺（默认不升，保持改动聚焦）。
- **不引入新抽象**：不新建 `CopyBytes` 之类的预留接口——死代码不应换个名字继续留着。未来若真需要从 `MemoryManager` 暴露字节拷贝，再加新方法并写真实调用者。
- 注释保持英文（代码），本文档为中文。

## 验证

- 编译通过（ninja，`build/`）。
- `ctest`：16/17 用例通过（`test_load_operators` 为预先存在的无关失败）。
- `grep` 确认全仓无 `MemoryManager::Copy` 残留、无 `.Copy(` / `->Copy(` 成员调用残留。
- 现有 `tests/core/test_copy.cpp`、`tests/core/test_memory.cpp`、`tests/backends/test_cuda_backend.cpp` 覆盖张量层 `Copy` 与字节层 `CopyBuffer` 的行为，删除死代码后行为不变。

## 风险与对策

- **风险**：误删 `CudaCopyBuffer`（`cuda_copy.cpp` / `backend_internal.h`）而非 `CudaMemoryManager::Copy`。**对策**：仅删 `MemoryManager` 派生类的 `Copy` override；`CudaCopyBuffer` 是自由函数、被 `CudaBackend::CopyBuffer` 使用，保留。删前用 `Grep CopyBuffer` 确认调用链。
- **风险**：某个未被检索到的外部/下游依赖该虚函数。**对策**：已确认 `Context::memory()` 私有 + `Backend` 前向声明，公共 API 不可达；库内全仓 grep 零调用。若用户下游有自定义后端，需同步告知。
- **备选方案（不推荐）**：若用户希望保留该接口以备未来扩展，可改名为 `CopyBytes` 而非删除。但保留死代码违反"不过度设计、不留死代码"原则，故推荐删除。
