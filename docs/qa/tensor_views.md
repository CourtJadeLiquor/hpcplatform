# Tensor 视图：为什么零拷贝，stride/offset 怎么算

> 相关代码：`include/mycv/core/tensor.h`、`src/core/tensor.cpp`、`include/mycv/core/tensor_iterator.h`

---

## 1. 先记住一个心智模型

`Tensor` 不是数据，它是一张**"数据在哪里、怎么解读"的便签**：

```
Tensor {
  buffer_  : shared_ptr<Buffer>   // 真正的内存（可能被很多 Tensor 共享）
  shape_   : Shape                 // 每个维度多长
  strides_ : int64_t[8]            // 每个维度走一步要跨多少个【元素】
  offset_  : size_t                // 从 buffer 起点数多少个【字节】开始（见 §2）
  dtype_, device_, backend_
}
```

`View` / `Slice` / `Permute` 之所以零拷贝，是因为它们**只改这张便签**（shape/strides/offset），`buffer_` 还是同一个 `shared_ptr`。数据一个字节都没动。

---

## 2. 最容易搞错的一点：stride 的单位是【元素】，不是字节

`Tensor::Stride(dim)` 返回的是**元素数**，不是字节数（[tensor.h](../../include/mycv/core/tensor.h#L85-L86) 注释明确写了 "Stride in elements (not bytes)"）。

这里有一组**单位必须分清**的约定（实测自 `src/core/tensor.cpp`）：

- `strides_` / `Stride(dim)`：单位是**元素**。`Slice` 里算偏移就是 `begin * Stride(dim)`（元素）。
- `offset_`：单位是**字节**。`Slice` 最终写的是 `offset_ += begin * Stride(dim) * ElementSize()`（[tensor.cpp](../../src/core/tensor.cpp#L107-L108)）。
- `Tensor::data()`：返回 `buffer_->data() + offset_`，即**已定位到视图起点的字节指针**（[tensor.h](../../include/mycv/core/tensor.h#L53-L55)）。
- `Tensor::Data<T>()`：`reinterpret_cast<T*>(data())`，即视图起点的**元素指针**。

所以写内核的正确姿势：用 `tensor.Data<T>()` 拿元素起点指针，用 `tensor.Stride(dim)`（元素）做索引，`ptr[i * stride]` 直接寻址。**不要**拿 `data()`（字节）再自己乘 `sizeof(T)`——`offset_` 已经是字节，混算会错位。

> 经验法则：在内核里永远用 `tensor.Data<T>()` + `tensor.Stride(dim)`（元素）做寻址；不要混用 `data()`（字节）和 stride（元素），否则会错位。

---

## 3. 三种视图各自做了什么

| 操作 | 改了什么 | 前提 | 拷贝？ |
| --- | --- | --- | --- |
| `View(shape)` | 改 `shape_` + 重算连续 `strides_` | 输入**必须连续**（否则 `kInvalidArgument`）且 `Numel` 不变 | 否 |
| `Slice(dim, begin, end)` | 改 `offset_`（加上 `begin * stride`）+ 该维长度 | 不越界 | 否 |
| `Permute(order)` | 重排 `shape_` 与 `strides_`（按 order 交换） | order 是合法排列 | 否 |
| `Reshape(shape)` | 连续→等价 View；非连续→**Clone** | Numel 不变 | 视情况 |
| `Clone()` | 分配新 Buffer，逐元素深拷贝 | — | 是 |

关键点：

- **`Reshape` 永远给你一个有效张量**：能零拷贝就零拷贝，不能就显式深拷贝。它不会"半成功"，也不会失败。代价是非连续时多一次拷贝，但行为可预期。
- **拷贝构造 `Tensor b = a` 不是深拷贝**：只是 `buffer_` 引用计数 +1，`a` 和 `b` 共享内存。要独立副本必须 `a.Clone()`。这是新手最常踩的坑——改了 `b` 发现 `a` 也变了。

---

## 4. 视图的代价：整块内存一起活

`Slice` 出一个 4×4 的小块，底层可能是一个 1024×1024 的大 Buffer。因为视图共享 `shared_ptr<Buffer>`，**只要小块还活着，整块大内存就不能释放**。

这不是 bug，是取舍（[`../architecture/P0.md`](../architecture/P0.md) §9）：用"轻量视图"换"简单所有权"。如果你的场景会切出大量小视图且长期持有，导致原大张量无法回收——这是已知局限，应在调用方主动 `Clone` 出独立小张量。

---

## 5. 非连续张量还能用吗？能，但要区分两类操作

视图常常产生**非连续**张量（如 `Slice` 后 stride 不等于连续值）。它们并非二等公民：

- **`Context::Copy`**：拷贝的是**逻辑元素**，按 stride 逐元素搬运，非连续输入正确（见 [`context_copy.md`](./context_copy.md)）。
- **元素级内核**：通过 `TensorIterator` 按 stride 遍历任意布局，`Add`/`Mul`/`Cast`/`Normalize` 都接受非连续输入。
- **要求连续的算子**：在校验层显式声明并拒绝（如某些后端的图像算子），不会静默给出错误结果。

判断连续性用 `Tensor::IsContiguous()`。

---

## 6. 一句话总结

> 视图 = 改便签不改数据；`Clone` 才换数据。stride 是元素、`Data<T>()` 是元素指针，别和字节的 `data()` 混算。`Reshape` 非连续会深拷贝，拷贝构造不会。
