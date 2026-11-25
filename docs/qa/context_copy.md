# Upload / Download / Copy：谁负责搬运、怎么选路、何时同步

> 相关代码：`src/core/copy.cpp`、`include/hpcplatform/core/transfer.h`、`src/core/backend.h`
> 易混点：三个自由函数怎么分工、跨后端怎么决定谁来拷、拷完能不能立刻用，以及 `Tensor::From/Into` 如何把这两步隐藏掉。

---

## 0. 三个入口的分工

三者都是 **`hpcplatform` 命名空间下的自由函数**，不再挂在 `Context` 上：执行者从张量本身推断，调用方不必（也不可能）选错 Context。

| 入口 | 方向 | 约束 | 用途 |
| --- | --- | --- | --- |
| `Upload(host_src, &device_dst)` | host → device | 源必须是 host 内存，目标必须是 device 内存 | 上卡，目标预分配 |
| `Download(device_src, &host_dst)` | device → host | 源必须是 device 内存，目标必须是 host 内存 | 回主机，目标预分配 |
| `Copy(&dst, src)` | 任意同类/跨后端 | shape/dtype 匹配，目标由目标后端驱动 | host↔host 跨后端、同设备搬运 |

方向（内存种类）在 `Upload`/`Download` 入口处就地校验；`Copy` 用**目标张量的所属后端**作为执行者（`ResolveOwningBackend`），因此下载时目标 host 张量由哪个 Context 分配不影响结果。三者都要求调用方**先分配目标**，返回前同步完成。

> 普通用户通常不需要直接调用这三个函数：`Tensor::From<T>(external, context)` 与 `Tensor::Into<T>(tensor)` 会在转换内部完成 H2D/D2H（见 §3），显式函数保留给需要控制预分配与复用缓冲区的场景。

---

## 1. Copy 拷贝的是【逻辑元素】，不是裸字节

`Copy(dst, src)` 的契约是：把 `src` 的**逻辑元素**搬到 `dst`（[transfer.h](../../include/hpcplatform/core/transfer.h#L19-L23)）。

这意味着：

- `src` 可以是**非连续视图**（`Slice` 后的张量）——Copy 会按 `src` 的 stride 逐元素读，按 `dst` 的 stride 逐元素写，而不是粗暴 `memcpy` 整块。
- 要求 `dst` 与 `src` 的 shape / dtype / 元素数匹配；不匹配返回错误。
- 连续 host↔host 才会走快速 `memcpy`；其他情况逐元素搬运。

> 所以"把一个 slice 视图拷到新张量"是安全的，结果就是视图看到的那些元素，不会把 stride 间隙里的垃圾也带过去。

---

## 2. 谁来执行拷贝：由【张量的所属后端】选路

`Copy` 用 `dst` 的 owning backend 驱动；设备侧存在时由设备后端执行：

| src \ dst | host (native/opencv) | device (cuda) |
| --- | --- | --- |
| host | 直接 memcpy / 逐元素 | H2D：目标后端的 `CopyBuffer`（CUDA 用 `cudaMemcpyAsync`） |
| device | D2H：源后端的 `CopyBuffer` | D2D：同设备内拷贝 |

后端的 `Backend::CopyBuffer`（[backend.h](../../src/core/backend.h)）有一条硬规则：

> 实现碰不到某类内存时，**必须返回 `kUnsupported`，而不是在调用方背后经 host staging**。

含义：如果一个新后端不会做 D2H，它应该老实返回 `kUnsupported`，让调用方知道，而不是偷偷先拷到 host 再拷到 device（那样会引入隐藏拷贝、破坏性能预期）。新增后端只需正确实现 `CopyBuffer`，就能同时支撑 `Copy`、`Upload`、`Download` 与 `Tensor::From/Into` 的选路。

**跨不同 GPU（peer-to-peer）**：当前明确返回 `kUnsupported`，不经 host 中转。需要的话调用方自己显式分两段拷。

---

## 3. 透明转换：`Tensor::From` / `Tensor::Into`

- `Tensor::From<T>(external, context)`：目标域由 `context` 决定。CPU Context 直接 host 拷贝；device Context 分配设备张量后，用设备后端的 `CopyBuffer` 完成 H2D（同步返回）。
- `Tensor::Into<T>(tensor)`：源域由张量决定。host 张量直接读出；device 张量自动 D2H；非连续设备张量先 `Clone` 一次再拷。
- 两者都复用 §2 的 `CopyBuffer` 选路，不会绕过后端能力校验。

因此样例里的算法代码只出现 `Tensor::From(image, *cuda)`、`Tensor::Into<cv::Mat>(result)`，看不到 `Upload` / `Download`。

---

## 4. 同步契约：调用返回即完成

`Copy` / `Upload` / `Download` 在返回前**同步完成**：

- CUDA 路径会 `cudaStreamSynchronize`，所以返回后 `dst` 的数据已经在目标内存里、立即可读；
- CPU 路径本身同步。

这是当前的公开契约。好处是生命周期简单：返回后你可以立即读 `dst`、可以立即让 `src` 析构。代价是 host 侧不能重叠 Copy 与计算（无异步流水）。若未来加异步 Copy，会是**新的显式入口**，不会偷偷改掉现有语义（[架构设计](../architecture.md) §1"显式优先于隐式"）。

---

## 5. 常见误用

| 误用 | 后果 | 正确做法 |
| --- | --- | --- |
| 期望 Copy 自动处理 dtype 转换 | 不转换，dtype 不符报错 | 先 `Cast` 再 Copy，或用算子输出直接是目标 dtype |
| 期望 Copy 自动处理 layout（NCHW↔NHWC） | 不转换 | 显式 `Transpose` / `Permute` |
| 对非连续设备张量反复 `Into` | 每次都可能 `Clone` | 先 `Clone`/`Reshape` 成连续张量，再多次读取 |
| Copy 后立刻在另一条流上用 `dst` 但没同步 | （当前 Copy 已同步，无碍；但若是自己写的异步路径则可能读到旧值） | 遵循"返回即完成"的公开契约 |

---

## 6. 与 `Clone` 的关系

`Tensor::Clone()` 内部就是"在 owning backend 上分配新张量 + `CopyTensorData`"（[tensor.cpp](../../src/core/tensor.cpp)）。区别：

- `Clone()`：自动找归属后端（Context 死了会建临时后端兜底），总是深拷贝到同后端。
- `Copy(&dst, src)`：目标张量的后端驱动，可跨后端。
- `Upload` / `Download`：host↔device 边界的显式入口，目标预分配且方向校验。
- `Tensor::From/Into`：把上述搬运封装进类型转换，算法代码不需要关心。

> 记忆点：**同后端深拷贝用 `Clone`，host↔device 用 `Upload`/`Download`，跨后端搬运用 `Copy`，第三方类型转换用 `From`/`Into`。**

---

## 7. 一句话总结

> Copy 搬的是逻辑元素（非连续也正确），路由由张量所属后端决定，碰不到的内存返回 `kUnsupported` 不偷转；返回前同步，立即可读。`From/Into` 把 H2D/D2H 藏在转换里，用户只描述算法。
