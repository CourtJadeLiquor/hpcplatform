# 项目结构整理：模块边界与"什么不该在 core"

> 相关代码：`include/hpcplatform/` 公共头树、`src/framework/` 内部支撑层
> 结论先行：公共 API 按职责分 4 模块；core 只放类型系统 + 执行原语；顶层只放聚合伞形头和版本；计时/剖析/统计是应用关心的事，不是库 API。

---

## 1. 模块划分原则

hpcplatform 公共 API 最终稳定在 4 个模块，划分标准是**依赖方向 + 扩展落点**：

| 模块 | 职责 | 扩展落点 | 对消费者暴露什么 |
| --- | --- | --- | --- |
| `core/` | 类型系统 + 执行原语 | 后端实现（src/backends/） | dtype / shape / device / memory / stream / error / kernel_level / context / tensor / transfer |
| `ops/` | 算子库 | 新算子文件夹（src/ops/） | op.h（OpContext + OpKind）+ 各算法头 + operator.h（配置驱动层） |
| `interop/` | 第三方类型适配 | TensorConvert<T> 特化 | conversion.h（trait 抽象）+ opencv.h（cv::Mat 特化） |
| 顶层 | 聚合伞形头 + 版本 | 永不扩展 | hpcplatform.h + version.h |

**依赖方向**：core 无依赖 → ops 依赖 core → interop 依赖 core（trait 在 interop/ 不是 core/ 的原因见 [trait_vs_interface.md](./trait_vs_interface.md)）。反向依赖不允许。

### 为什么 trait 在 interop/ 不是 core/

早期版本 trait 在 `core/conversion.h`，后来移到 `interop/conversion.h`。原因：

- trait 的唯一作用是"给外部类型加 Tensor 转换"——它天生属于 interop 的扩展层
- core 应该只关心"Tensor 能不能被分配、能不能和其他 Tensor 互转、能不能传给算子"——不关心"Tensor 和 cv::Mat 怎么互转"
- primary template 返回 Unsupported，不 include 任何外部头——放进 core 也不会污染 core，但**边界语义不对**：core 是"我们自己的类型系统"，interop 是"和外部世界对接"

### 为什么 operator.h 从顶层移到 ops/

早期 `operator.h` 在 `include/hpcplatform/` 顶层，和 `context.h`/`tensor.h` 平级。但它的职责是"配置驱动算子对象"——LoadOperators 把 YAML 配置实例化成 `ResizeOperator` / `GaussianBlurOperator`，而这些具体算子头（resize.h / gaussian_blur.h）都在 ops/ 里。operator.h 依赖 ops/op.h（OpKind、OpContext），放在顶层等于把 ops 层的一部分暴露到了和 core 同级的位置，边界混乱。

移到 `ops/operator.h` 后，ops/ 包含三件事：op.h（执行上下文 + OpKind）、各算法头、operator.h（配置驱动）。边界清晰：ops/ 就是算子层，配置驱动层归这里，不归顶层。

### 为什么 timing.h 从 core 删除

早期 `core/timing.h` 提供 Stopwatch / ScopedTimer / ToMilliseconds。后来发现：

- **全库唯一消费者是 sample 05**。benchmark 文件全直接用 `<chrono>`，根本不用库提供的计时器
- 计时工具不是库 API——库提供的是"执行算子、分配 Tensor、转换数据"，不是"测量算子花了多久"
- sample 05 里改成 sample-local 的 `ScopedTimer`（10 行代码）加 `<chrono>`，零成本、零依赖
- 更关键的理由：**库不应该替应用做计时**。应用的计时需求千奇百怪（采样、累加、分阶段、GPU 侧事件计时），库提供一个通用 Stopwatch 反而给人"应该用这个"的错觉，而实际永远不够用

结论：库不提供计时工具，应用自己用 `<chrono>` 或外部库。

### 为什么 profiling.h / stats.h 从 src/framework 移到 tests/perf

这两个文件的唯一消费者是 `tests/perf/test_operator_perf.cpp`——性能评估用例，不是库源码。库内部不需要剖析宏，不需要内存快照。放在 src/framework 里让它们看起来像库的公共/内部 API，但实际上只有测试用。移到 tests/perf 后，它们成为**测试工具**，不是库工具——边界正确。

---

## 2. 内部结构（src/）

公共 API 的模块划分决定了 src/ 的组织：

```
src/core/          # Backend 契约 + 核心实现
src/backends/      # cpu / cuda / opencv / npu —— 不含算子
src/framework/     # 库内共享支撑层（registry / descriptor / image_utils / dtype_dispatch / compiler）
src/ops/
  common/          # ValidateInvocation / AllocateOutput
  loader/          # 配置驱动层（从原 src/operator/ 移入，ops/ 下的子层）
  <op>/            # 每算子自治文件夹
```

### src/framework/ 为什么保留

framework/ 曾被混入 profiling.h 和 stats.h，显得像 grab-bag。清理后保留的文件都是**库内部双向依赖的共享层**：

- `op_registry.{h,cpp}`：被 cpu_backend.cpp（注册内核）和各算子 operator.cpp（查找内核）双向使用
- `op_descriptor.{h,cpp}`：被 load_operators_real.cpp（实例化）和各算子 operator.cpp（注册描述符）双向使用
- `image_utils.h`：被 gaussian_blur、resize、laplacian_pyramid、wavelet、fourier 五个算子共用
- `dtype_dispatch.h`：gaussian_blur_simd、fourier、wavelet、laplacian_pyramid 共用
- `compiler.h`：gaussian_blur_cpu_simd 等 SIMD 路径共用

这些是跨算子共享的内部工具，不属于任何单一算子层，位置在 framework/ 是合理的。

### src/ops/loader/ 为什么从 src/operator/ 改名并移入

早期 `src/operator/`（单数）和 `src/ops/`（复数）平级。名字不统一是小问题，层级不对是大问题：loader 的职责是"把 YAML 配置实例化成已绑定的 Operator 对象"——它依赖 op_registry、op_descriptor、各算子头，**它是 ops 层的子层**，不应该和 ops/ 平级。移入 `src/ops/loader/` 后，层级关系正确：ops/ 是算子层，loader/ 是它的配置驱动子层。

---

## 3. 顶层头文件：只有两个

`include/hpcplatform/` 最终顶层只有：

| 文件 | 职责 |
| --- | --- |
| `hpcplatform.h` | 聚合伞形头：一次性 include 所有公共 API |
| `version.h` | 版本宏：MAJOR / MINOR / PATCH / VERSION_STRING |

**没有其他顶层头。** 这确保了顶层只承担两个纯粹的职责——聚合和版本——而不混入任何具体模块的实现细节。

---

## 4. 架构约束（已写入 architecture.md §2.2）

| 规则 | 内容 |
| --- | --- |
| R1 | 公共头不得 include src/ 下任何文件 |
| R2 | `src/core/` 不得 include framework / backends / ops |
| R3 | `src/framework/` 只依赖 core 和稳定的 ops/op.h、ops/image.h |
| R4 | `src/backends/` 只依赖 core / framework，不得 include src/ops/ |
| R5 | 算子文件夹入口/CPU 实现只依赖公共头、core、framework、ops/common、本文件夹内部 |
| R6 | 依赖单向无环 |

这 6 条是模块边界的底线。任何新增文件如果违反了某条，就是模块边界需要调整的信号。

---

## 5. 一句话总结

> 公共 API 按"依赖方向 + 扩展落点"分 4 模块：core（类型+原语）/ ops（算子）/ interop（第三方适配）/ 顶层（聚合+版本）。core 只放必须的，计时/剖析/统计是应用关心的事不是库 API；顶层只放两个文件；内部 src/ 按同样原则组织，依赖单向无环。
