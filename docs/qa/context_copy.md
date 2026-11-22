# Context::Copy：拷贝的是元素不是字节，怎么选路、何时同步

> 相关代码：`src/core/copy.cpp`、`include/mycv/core/context.h`、`src/core/backend.h`
> 易混点：Copy 到底拷了什么、跨后端怎么决定谁来拷、拷完能不能立刻用。

---

## 1. Copy 拷贝的是【逻辑元素】，不是裸字节

`Context::Copy(dst, src)` 的契约是：把 `src` 的**逻辑元素**搬到 `dst`（[context.h](../../include/mycv/core/context.h#L53-L55) 注释："Copies logical elements, including non-contiguous views"）。

这意味着：

- `src` 可以是**非连续视图**（`Slice` 后的张量）——Copy 会按 `src` 的 stride 逐元素读，按 `dst` 的 stride 逐元素写，而不是粗暴 `memcpy` 整块。
- 要求 `dst` 与 `src` 的 shape / dtype / 元素数匹配；不匹配返回错误。
- 连续 host↔host 才会走快速 `memcpy`；其他情况逐元素搬运。

> 所以"把一个 slice 视图拷到新张量"是安全的，结果就是视图看到的那些元素，不会把 stride 间隙里的垃圾也带过去。

---

## 2. 谁来执行拷贝：由【目标上下文】选路

`dst` 属于哪个 Context，就由那个 Context 的后端决定拷贝路径：

| src \ dst | host (native/opencv) | device (cuda) |
| --- | --- | --- |
| host | 直接 memcpy / 逐元素 | H2D：目标后端的 `CopyBuffer`（CUDA 用 `cudaMemcpyAsync`） |
| device | D2H：源后端的 `CopyBuffer` | D2D：同设备内拷贝 |

后端的 `Backend::CopyBuffer`（[backend.h](../../src/core/backend.h)）有一条硬规则：

> 实现碰不到某类内存时，**必须返回 `kUnsupported`，而不是在调用方背后经 host staging**。

含义：如果一个新后端不会做 D2H，它应该老实返回 `kUnsupported`，让调用方知道，而不是偷偷先拷到 host 再拷到 device（那样会引入隐藏拷贝、破坏性能预期）。新增后端只需正确实现 `CopyBuffer`，就能参与 `Context::Copy` 的选路。

**跨不同 GPU（peer-to-peer）**：当前明确返回 `kUnsupported`，不经 host 中转。需要的话调用方自己显式分两段拷。

---

## 3. 同步契约：Copy 返回即完成

`Context::Copy` 在返回前**同步完成**：

- CUDA 路径会 `cudaStreamSynchronize`，所以返回后 `dst` 的数据已经在目标内存里、立即可读；
- CPU 路径本身同步。

这是当前的公开契约。好处是生命周期简单：Copy 返回后你可以立即读 `dst`、可以立即让 `src` 析构。代价是 host 侧不能重叠 Copy 与计算（无异步流水）。若未来加异步 Copy，会是**新的显式入口**，不会偷偷改掉现有 `Copy` 的语义（[架构设计](../architecture.md) §1"显式优先于隐式"）。

---

## 4. 常见误用

| 误用 | 后果 | 正确做法 |
| --- | --- | --- |
| 跨设备 Copy 却不传目标 Context（在源 Context 上拷） | 路径选错 / 报错 | 在 `dst` 所在的 Context 上调用 `Copy(&dst, src)` |
| 期望 Copy 自动处理 dtype 转换 | 不转换，dtype 不符报错 | 先 `Cast` 再 Copy，或用算子输出直接是目标 dtype |
| 期望 Copy 自动处理 layout（NCHW↔NHWC） | 不转换 | 显式 `Transpose` / `Permute` |
| Copy 后立刻在另一条流上用 `dst` 但没同步 | （当前 Copy 已同步，无碍；但若是自己写的异步路径则可能读到旧值） | 遵循"返回即完成"的公开契约 |

---

## 5. 与 `Clone` 的关系

`Tensor::Clone()` 内部就是"在 owning backend 上分配新张量 + `CopyTensorData`"（[tensor.cpp](../../src/core/tensor.cpp#L141-L149)）。区别：

- `Clone()`：自动找归属后端（Context 死了会建临时后端兜底），总是深拷贝到同后端。
- `Context::Copy`：你指定目标 Context，可跨后端。

> 记忆点：**同后端深拷贝用 `Clone`，跨后端/跨设备搬运用 `Context::Copy`。**

---

## 6. 一句话总结

> Copy 搬的是逻辑元素（非连续也正确），路由由目标 Context 的后端决定，碰不到的内存返回 `kUnsupported` 不偷转；返回前同步，立即可读。dtype/layout 不自动转，要转自己显式调算子。
