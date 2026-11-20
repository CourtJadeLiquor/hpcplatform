# mycv 架构文档总览

> 适用版本：v0.1（P0–P5 已落地，P6 待实施）
> 文档视角：**项目架构**——回答"系统由什么组成、为什么这样组成、各部分如何协作"。
> 配套阅读：[`docs/qa/`](../qa/) 从学习者与贡献者视角解释各模块的易混淆点。

---

## 1. 如何阅读本套文档

mycv 是一个面向科研平台的跨平台 C++17 算子库。它的核心诉求不是"功能最多"，而是**行为可预期、实验可复现、扩展可判定**。整套架构文档按里程碑（P0–P5）组织，每个里程碑既描述"交付了什么"，也讲清"为什么这么设计"。

建议阅读顺序：

1. **先读本文件**：建立全局分层与术语共识。
2. **[`P0.md`](./P0.md)**：整体架构与设计哲学——这是整套系统的"宪法"。
3. 按你的关注点深入：
   - 想理解 Tensor / 内存 / 生命周期 → [`P1.md`](./P1.md)
   - 想理解算子如何被选择与执行 → [`P2.md`](./P2.md)
   - 想理解 CPU 多版本优化与 benchmark → [`P3.md`](./P3.md)
   - 想理解某个具体后端 → [`P4.md`](./P4.md)（OpenCV）/ [`P5.md`](./P5.md)（CUDA）

---

## 2. 文档地图

| 文档 | 阶段 | 状态 | 核心内容 |
| --- | --- | --- | --- |
| [`P0.md`](./P0.md) | 总纲 | ✅ | 定位、目标/非目标、设计哲学、分层架构、核心抽象一览、扩展契约 |
| [`P1.md`](./P1.md) | Core 完备 | ✅ | Tensor/Buffer/MemoryManager/Stream/Context 精确契约、视图与 stride、跨后端拷贝、生命周期倒置、Wrap 零拷贝 |
| [`P2.md`](./P2.md) | 算子分发 | ✅ | `(OpKind, BackendType, CpuKernelLevel)` 三元组分发、校验层、错误码语义、新增算子 |
| [`P3.md`](./P3.md) | CPU 多版本 | ✅ | `CpuKernelLevel` 体系、bit-exact 约定、AVX2 独立编译单元、benchmark 方法 |
| [`P4.md`](./P4.md) | OpenCV 后端 | ✅ | `cv::Mat` 所有权、dtype/layout 约束、不静默转换、互拷 |
| [`P5.md`](./P5.md) | CUDA 后端 | ✅ | non-blocking stream、caching pool、跨设备拷贝、同步契约、外部内存可见性 |

> P6（NPU 后端与交付化）尚未实施，故未单列架构文档；其规划见仓库根目录 [`PLAN.md`](../../PLAN.md) 第 9 节。

---

## 3. 一页纸架构速览

```
┌─────────────────────────────────────────────────────────────┐
│  应用 / 调用方                                                │
│  Context::Create(BackendType)  ──  显式指定，无自动探测        │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│  算子层 Ops（include/mycv/ops/*.h）                            │
│  Add / Mul / Cast / Normalize / Resize / Crop / Transpose /   │
│  WarpAffine   ——  统一入口：OpContext{context, stream, level} │
└───────────────┬─────────────────────────────────────────────┘
                │  (OpKind, BackendType, CpuKernelLevel)
┌───────────────▼─────────────────────────────────────────────┐
│  分发层 Registry（src/ops/op_registry.cpp）                   │
│  三元组 → OpKernelFn；不匹配则 kUnsupported / kNotImplemented │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│  核心抽象 Core（include/mycv/core/）                          │
│  Tensor · Buffer · MemoryManager · Stream · Context ·         │
│  Shape · Device · DataType · Status/Result                    │
└───────┬───────────────────────┬─────────────────────────────┘
        │ 虚接口                 │ 持有
┌───────▼──────────┐   ┌────────▼─────────────────────────────┐
│  后端 Backends    │   │  Context                              │
│  native_cpu       │   │  绑定一个 Backend + 一个 Device        │
│  opencv           │   │  暴露 memory() / stream() / Copy()    │
│  cuda             │   │  / Wrap()                             │
│  npu (占位)        │   └──────────────────────────────────────┘
└───────────────────┘
```

**三条主线**：

- **显式选择**：后端、设备、CPU 内核级别全部由调用方命名，库不做平台探测、不做自动降级。
- **描述符与存储分离**：`Tensor` 只描述（shape/dtype/strides/offset），`Buffer` 才持有内存；后端把"怎么分配/释放"藏进 deleter。
- **错误可判定**：core 不抛异常，用 `Status`/`Result`；每种失败有确定的 `StatusCode`，不静默换实现。

---

## 4. 术语表

| 术语 | 含义 |
| --- | --- |
| Backend（后端） | 实现提供者：`native_cpu` / `opencv` / `cuda` / `npu`。由 `BackendType` 标识。 |
| Device（设备） | 执行硬件：`cpu` / `gpu` / `npu`，带 `index`。由 `Device` 结构体标识，与后端分离。 |
| Buffer | 不透明的设备无关内存句柄，持有 `data/bytes/device/kind` 与一个 deleter。 |
| MemoryManager | 后端实现的统一分配/拷贝/统计接口。 |
| Tensor | 描述符（shape/dtype/strides/offset）+ `shared_ptr<Buffer>`。 |
| View | 零拷贝元数据视图，共享底层 Buffer。 |
| Wrap | 零拷贝接管外部内存（`cv::Mat` / `cudaMalloc` / NPU …）。 |
| OpContext | 单次算子调用的调度描述符：`context` / `stream` / `cpu_level`。 |
| CpuKernelLevel | CPU 实现版本：`naive` / `openmp` / `simd_generic` / `simd_avx2` / `simd_neon` / `tbb`。 |
| 不静默降级 | 请求的后端或内核级别不可用时返回错误码，绝不偷偷换成另一个实现。 |

---

## 5. 文档约定

- **代码即事实**：所有接口签名、枚举值、文件路径均锚定到当前仓库源码；若与代码冲突，以代码为准并修订本文档。
- **PDF/手册友好**：使用标准 Markdown 标题、表格与代码块；不依赖站点特定语法，便于 `pandoc` 等工具直接导出 PDF。
- **每个设计决策都回答"为什么"**：避免只罗列"是什么"，重点解释取舍与边界。
