# 内存生命周期：Tensor 为什么能比 Context 长寿

> 相关代码：`include/mycv/core/memory.h`、`include/mycv/core/tensor.h`、`src/core/tensor.cpp`、各后端 `*_memory_manager.*`
> 配套：[`cpubackends.md`](./cpubackends.md) 讲 CPU 侧的三层结构，本文讲通用模型与 `Wrap`。

---

## 1. 那个让人心里一紧的场景

```cpp
Result<Tensor> MakeFoo() {
  auto ctx = Context::Create(BackendType::kCUDA).value();
  return ctx->AllocateTensor(Shape{2, 3}, DataType::kFloat32);
  // ctx 在这里析构 —— 返回的 Tensor 还安全吗？
}

auto t = MakeFoo().value();   // 还能用吗？会悬垂吗？
```

**答案：安全。** 这是 mycv 刻意保证的。但很多人第一次看到会怀疑，所以值得讲清机制。

---

## 2. 三个角色，三种寿命

理解这件事的关键是分清谁持有谁：

```
Context ──shared_ptr──> Backend ──持有──> MemoryManager
                         │
                         └── 内部持有 ──> shared_ptr<AllocatorState>   ← 真正的池/统计状态
                                              ▲
Tensor ──shared_ptr──> Buffer ──deleter(闭包)──┘ 捕获 shared_ptr<AllocatorState>
Tensor ──weak_ptr────> Backend   (owner_)
```

| 角色 | 寿命 | 作用 |
| --- | --- | --- |
| `Context` / `Backend` | 调用方控制 | 提供分配策略与默认流 |
| `AllocatorState`（后端内部） | 被 `Backend` 和所有活着的 `Buffer` 共同持有 | 持有空闲块池与统计计数器 |
| `Buffer` | 被 `Tensor` 持有（`shared_ptr`） | 持有数据指针 + deleter |
| `Tensor` | 调用方控制 | 描述符 |

**核心 trick**：`Buffer` 的 deleter 捕获的不是 `Backend*`（裸指针），而是 `shared_ptr<AllocatorState>`。所以：

- `Context` 析构 → `Backend` 析构 → `Backend` 对 `AllocatorState` 的引用没了；
- 但只要还有 `Tensor` 活着，它的 `Buffer` 就还持有 `AllocatorState` 的一份引用 → 池与统计状态继续存活；
- 最后一个 `Tensor` 释放 → `Buffer` 析构 → deleter 把内存（或整池）正确还给 OS/驱动 → `AllocatorState` 才真正析构。

一句话：**分配策略（Backend）可以死，回收责任（随 Buffer 延续的 AllocatorState）不能死。**

---

## 3. `owner_` 为什么用 weak_ptr，不用 shared_ptr

`Tensor::owner_` 是 `std::weak_ptr<Backend>`（[tensor.h](../../include/mycv/core/tensor.h#L83-L84) 注释："Expires with the Context"）。

- 如果用 `shared_ptr<Backend>`，`Tensor` 会把 `Context`/`Backend` **续命**——Context 明明该析构了却因为还有 Tensor 而活着，池无法释放，违背调用方预期。
- 用 `weak_ptr`：`Tensor` 不延长 Context 寿命，但 `Clone` / 跨后端拷贝时可以 `lock()` 试探——如果 Backend 还在就用它分配；如果已过期（Context 死了），`ResolveOwningBackend` 会**创建一个临时后端**兜底完成深拷贝（[tensor.cpp](../../src/core/tensor.cpp#L141-L149) 的 `Clone` 路径）。

> 这就是 "tensor 比 context 长寿" 测试（`CudaTensorCanOutliveItsContext`）验证的东西：Context 先死，Tensor 后死，全程无 use-after-free。ASAN 构建全绿覆盖此路径。

---

## 4. `Wrap`：借用 vs 收养

`Context::Wrap` 把外部内存包成 Tensor，**不分配不拷贝**。释放语义完全由 deleter 决定（[context.h](../../include/mycv/core/context.h#L44-L51)）：

| deleter | 语义 | 什么时候释放 | 适用场景 |
| --- | --- | --- | --- |
| 空（`nullptr`） | **借用 borrow** | 永不释放（调用方负责） | `cv::Mat` 还在外部管、你只想临时读一下 |
| 非空 | **收养 adopt** | 最后一个 `Tensor`/`Buffer` 引用死亡时调 deleter | 你想把 `cudaMalloc` 的指针交给 Tensor 管生命周期 |

`Buffer::~Buffer` 的逻辑很简单（[memory.h](../../include/mycv/core/memory.h#L41-L43)）：deleter 非空且 `data_ != nullptr` 就调用。

**两个必须记住的后果**：

1. **借用时，内存活多久由你决定，Tensor 不管。** 如果你 Wrap 了一个栈上数组，然后让 Tensor 逃出作用域——就是 use-after-free。借用只适合"外部 owner 明确比 Tensor 长寿"的场景。
2. **Wrap 不经过 `MemoryManager::Allocate`，所以 `MemoryStats` 不变。** 外部内存不算管理器的分配，统计里看不到它。这是刻意的：管理器只统计自己分配的内存。收养的 deleter 也直接释放（如 `cudaFree`），不回池。

---

## 5. 外部内存的"数据可见性"是另一个问题（易混淆）

寿命和可见性是两件事：

- **寿命**：deleter 保证指针在 Tensor 活着时不被释放（收养），或你保证（借用）。
- **可见性**：如果外部内存在另一条流上产生（如默认流上的 `cudaMemcpy`），数据对 mycv 的 non-blocking 流未必立即可见。

寿命对了，可见性不对，内核照样读到垃圾。详见 [`cuda_stream_visibility.md`](./cuda_stream_visibility.md)。

---

## 6. 一句话总结

> `Buffer` 持数据、`AllocatorState` 持池、两者通过 `shared_ptr` 解耦，所以 `Context` 先死不影响已分配 Tensor 的释放；`owner_` 用 `weak_ptr` 不续命 Context，`Clone` 在 Context 死后用临时后端兜底。`Wrap` 的 deleter 为空是借用、非空是收养，且不进统计。
