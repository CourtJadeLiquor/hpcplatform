# 算子分发：我的算子到底跑的是哪个实现

> 相关代码：`src/framework/op_registry.{h,cpp}`、`src/ops/common/op_common.h`、`src/ops/register_ops.cpp`
> 这是全库最容易"以为跑了 A 其实跑了 B"的地方，也是 hpcplatform 最在意的地方。

---

## 1. 分发的键：三元组

一次算子调用能精确命中哪个内核，由三个维度共同决定（[op_registry.cpp](../../src/framework/op_registry.cpp#L28)）：

```
(OpKind, BackendType, CpuKernelLevel)  →  一个函数指针 OpKernelFn
```

- `OpKind`：哪个算子（`kAdd` / `kMul` / …），来自算子公共函数内部写死；
- `BackendType`：哪个后端，来自**输入张量**的 `input.backend()`（不是你传的 OpContext！）；
- `CpuKernelLevel`：哪个 CPU 版本，来自 `OpContext.cpu_level`。

> 注意后端取自输入张量。所以 `Add(cuda_ctx_op, gpu_tensor, ...)` 会查 `(kAdd, kCUDA, kNaive)`——即使你心里想跑 CPU，只要输入在 GPU 上，就走 CUDA 内核。要跑 CPU，先把张量 `Copy` 到 CPU 上下文。这是"数据在哪就在哪算"，不做隐式跨设备调度。

---

## 2. 一次调用的顺序（以 Add 为例）

[add.cpp](../../src/ops/add/add.cpp#L46-L58) 的固定骨架：

```
1. ValidateInvocation(ctx, input)        ← 先校验：input 的 backend/device 必须匹配 ctx
2. ValidateBinaryInputs(input, &other)   ← shape / dtype / 广播
3. FindKernel(kAdd, input.backend(), ctx.cpu_level)
4. AllocateOutput(...)                   ← 在 ctx 的后端上分配输出
5. kernel(ctx, input, &other, nullptr, &output)
```

第 1 步很关键：**输入张量必须属于这个 OpContext 的后端/设备**，否则直接 `kInvalidArgument`，连查表都不查。这也是为什么 `Wrap` 出来的张量能直接喂算子——它的 `backend()`/`device()` 被设成了 Context 的（[`memory_lifecycle.md`](./memory_lifecycle.md) §4）。

第 4 步：输出分配在 **OpContext 的后端**上。通常输入和 ctx 同后端，所以输出也在那；但如果故意构造（输入在 A、ctx 是 B），第 1 步就会拦下。

---

## 3. 查不到时返回什么：三个错误码的精确边界

`FindKernel`（[op_registry.cpp](../../src/framework/op_registry.cpp#L64-L86)）对未命中分三种情况，**绝不能混**：

```
请求 (op, backend, level)
  │
  ├─ backend 不是 kNativeCpu 且 level != kNaive ?
  │     └─ 是 → kInvalidArgument  "该后端只接受 kNaive"
  │
  └─ 查表命中 ?
        ├─ 命中 → 返回内核函数
        └─ 未命中
              ├─ backend == kNativeCpu ?
              │     └─ 是 → kUnsupported  "这个 CPU level 没编译进来"
              └─ 否 → kNotImplemented  "这个后端没实现这个算子"
```

| 错误码 | 什么时候 | 你该怎么理解 / 怎么办 |
| --- | --- | --- |
| `kInvalidArgument` | 非 CPU 后端传了非 `kNaive` | 你传参错了。OpenCV/CUDA 没有多版本，传 `kOpenMP` 是非法参数，改回 `kNaive`。 |
| `kUnsupported` | CPU 后端请求了**没编译**的 level | 当前二进制没编这个版本。要么换构建开关重编，要么换一个已编的 level。**库不会替你降到 naive。** |
| `kNotImplemented` | 后端压根**没实现**该算子 | 这个后端就是不支持（如 NPU 的一切）。换后端，或等实现。 |

口诀：**参数错 → Invalid；没编进来 → Unsupported；没这功能 → NotImplemented。**

> 为什么要分这么细？因为调用方写降级逻辑时需要知道：是"我参数写错了"（改代码）、"换个编译选项"（改构建）、还是"这条路真的不通"（换方案）。如果库只回一个模糊错误，上层就只能猜。另外——**降级是调用方的事，库本身不做任何静默降级**。

---

## 4. 为什么不做"自动降级到 naive"

这是 hpcplatform 的核心立场（[架构设计](../architecture.md) §1"显式优先于隐式"），也是最常被挑战的设计：

- 如果请求 AVX2 但没编，自动跑 naive——功能上"能用"，但你**以为在测 AVX2 的性能，实际测的是 naive**，实验结论作废。
- 科研场景下，"静默换实现"比"报错"危险得多：报错你立刻知道，静默你可能发表了错误数字。
- 所以库选择**失败得明确**：返回 `kUnsupported`，让你有意识地决定"重编"还是"真的要跑 naive"（真要跑就显式传 `kNaive`）。

---

## 5. 注册是延迟的、按编译宏守卫的

内核表在第一次 `FindKernel` 时通过 `std::call_once` 填充（[op_registry.cpp](../../src/framework/op_registry.cpp#L35-L54)）。每个算子的 `Register*Kernels()` 里，每个版本都用 `#ifdef` 包住：

```cpp
#ifdef hpcplatform_ARCH_AVX2
RegisterKernel(kAdd, kNativeCpu, kSimdAvx2, &AddCpuAvx2);
#endif
```

含义：**编译期没开的宏，这个版本根本不会出现在运行期的表里**。所以"请求未编译 level → 表里没有 → `kUnsupported`"是编译期就决定的，运行期无法绕过。这保证了二进制的能力边界是封闭的、可审计的。

---

## 6. 快速自检：我怎么知道实际跑了哪个

- **编译期**：看构建时开了哪些宏（`build/CMakeCache.txt` 里的 `hpcplatform_WITH_OPENMP` / `hpcplatform_ENABLE_TBB` / `hpcplatform_ARCH_AVX2` 等）。
- **运行期**：`Backend::SupportsKernelLevel(level)` 告诉你这个后端/构建能不能跑某 level（由编译宏 + 已注册内核共同决定，不虚报）。
- **行为验证**：`tests/consistency/test_cpu_kernels.cpp` 拿 naive 当 oracle 逐位对拍其他版本——如果某版本结果与 naive 不一致（元素级应为 bit-exact），测试会红。

---

## 7. 一句话总结

> 命中靠三元组 `(op, 输入张量的 backend, ctx.cpu_level)`；查不到分三种错——非 CPU 要非 naive 是 `kInvalidArgument`、CPU 没编进是 `kUnsupported`、后端没实现是 `kNotImplemented`；库永远不静默换实现，要降级你自己显式传 `kNaive`。
