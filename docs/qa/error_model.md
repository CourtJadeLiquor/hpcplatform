# 错误模型：为什么不用异常，Status/Result 怎么用

> 相关代码：`include/mycv/core/error.h`
> 易混点：`Result::value()` 不抛异常、六种错误码怎么选、`MYCV_RETURN_IF_ERROR` 的用法。

---

## 1. 为什么不用异常

core 层明确**不抛异常**，错误用 `Status` / `Result<T>` 表达（[error.h](../../include/mycv/core/error.h#L21-L23) 注释写明动机）：

1. 部分 NPU / 嵌入式工具链以 `-fno-exceptions` 编译，异常不可用；
2. 异常让 C ABI 边界与跨语言绑定变复杂（为后续 Python 绑定留出空间）；
3. 异常会打断"错误码即契约"的可判定性——库希望每种失败都有确定的 `StatusCode`，调用方据此决策。

代价是调用方必须**显式检查**返回值，不能靠 try/catch 偷懒。这是刻意的。

---

## 2. Status 与 Result

### `Status`

```cpp
class Status {
  bool ok() const;
  StatusCode code() const;
  const std::string& message() const;
};
```

工厂方法：`Status::Ok()` / `InvalidArgument(...)` / `OutOfMemory(...)` / `Unsupported(...)` / `NotImplemented(...)` / `BadState(...)` / `Internal(...)`。

### `Result<T>`

要么是值，要么是 `Status`：

```cpp
template <typename T>
class Result {
  bool ok() const;
  const Status& status() const;
  T& value();              // ⚠️ 不抛异常
  T ValueOr(T fallback) const;
};
```

**最重要的一个坑**：`Result::value()` **不会抛异常**。如果 `!ok()`，调用 `value()` 拿到的是一个**默认构造的 `T`**（不是错误、不是异常），用了就是静默错误。所以：

```cpp
auto r = ctx->AllocateTensor(...);
if (!r.ok()) { /* 处理 r.status() */ return; }   // 必须先查
Tensor t = r.value();                             // 再取值
```

> 新手常写 `Tensor t = ctx->AllocateTensor(...).value();` 一步到位——失败时 `t` 是个空壳（`buffer_ == nullptr`），后续 `t.Data<float>()` 返回 `nullptr`，解引用就崩。**永远先查 `ok()`。**

---

## 3. `MYCV_RETURN_IF_ERROR`

最常用的错误传播宏（[error.h](../../include/mycv/core/error.h#L87-L91)）：

```cpp
#define MYCV_RETURN_IF_ERROR(expr)              \
  do {                                          \
    ::mycv::Status mycv_status = (expr);         \
    if (!mycv_status.ok()) return mycv_status;   \
  } while (0)
```

用法（算子公共函数里随处可见）：

```cpp
MYCV_RETURN_IF_ERROR(ValidateInvocation(context, input));
auto kernel = FindKernel(...);
if (!kernel.ok()) return kernel.status();   // Result 没有宏，手动查
```

> `MYCV_RETURN_IF_ERROR` 只接返回 `Status` 的表达式；返回 `Result<T>` 的要手动 `if (!x.ok()) return x.status();`，因为还需要把 `T` 取出来。

---

## 4. 六种错误码怎么选

`StatusCode`（[error.h](../../include/mycv/core/error.h#L9-L17)）：

| 码 | 什么时候用 | 典型场景 |
| --- | --- | --- |
| `kOk` | 成功 | — |
| `kInvalidArgument` | **调用方传参非法**，重试同样参数还是错 | shape 含非正维度、后端要非 naive level、View 非连续输入 |
| `kOutOfMemory` | 分配失败 | `posix_memalign` / `cudaMalloc` 返回错误 |
| `kUnsupported` | **当前构建/后端不支持这个能力**，但换构建或后端可能行 | 没编 AVX2 却请求 `kSimdAvx2` |
| `kNotImplemented` | **功能根本没实现** | NPU 后端的任意算子、跨 GPU P2P 拷贝 |
| `kBadState` | 对象状态不对，不是参数问题 | 对未定义的 Tensor 调 `View`（`buffer_ == nullptr`） |
| `kInternal` | 不该发生的内部错误（兜底） | 理论上不可达的分支 |

最容易混的是 `kUnsupported` vs `kNotImplemented` vs `kInvalidArgument`，口诀见 [`op_dispatch.md`](./op_dispatch.md) §3：**参数错 → Invalid；没编进来 → Unsupported；没这功能 → NotImplemented。**

---

## 5. 错误信息要写什么

错误信息应包含**足够定位的上下文**：op 名、后端名、level 名、具体哪个参数。例如分发表的未命中信息（[op_registry.cpp](../../src/framework/op_registry.cpp#L78-L85)）：

- `cpu kernel 'add' is not built for level simd_avx2`
- `op 'add' is not implemented for backend npu`
- `opencv backend accepts only CpuKernelLevel::kNaive; requested openmp`

这样调用方拿到 `Status::message()` 就能直接定位，不用猜。

---

## 6. 新增代码的错误处理纪律

1. 可能失败的函数返回 `Status` 或 `Result<T>`，**不要抛异常**，也不要用 `assert` 替代（`assert` 在 Release 被裁掉）。
2. 跨 API 边界的错误必须转成 `Status`：第三方错误（`cudaError_t`、`cv::Status`）用后端的包装函数转成 `Status`（CUDA 是 `CudaStatus`）。
3. 调用方拿到 `Result<T>` 先查 `ok()`，再碰 `value()`。
4. 错误码选择按 §4 的语义，不要图省事一律 `kInternal`。

---

## 7. 一句话总结

> core 不抛异常，所有失败走 `Status`/`Result`；`Result::value()` 失败时不抛而是给默认值，所以**必须先查 `ok()`**；错误码要选准——参数错、没编进、没实现、状态错、内部错各有归属，信息里带上 op/后端/level 才好定位。
