# 为什么 hpcplatform 禁用友元类

> 相关代码：`include/hpcplatform/core/context.h`、`include/hpcplatform/core/tensor.h`
> 结论先行：对外头文件里零 friend。需要内部访问时，靠公开 accessor 或把该能力移入类自身的 static 方法。

---

## 1. 问题：hpcplatform 的早期设计里有 3 个跨架构 friend

```cpp
// context.h（早期版本）
friend class Tensor;
friend class OpContext;
const std::shared_ptr<Backend>& backend_ptr() const { return backend_; }

// tensor.h（早期版本）
friend class Context;
```

Context 把 `Tensor` 和 `OpContext` 声明为 friend，让它们直接读 `backend_ptr()` 和 `memory()`。后来又引入了 `internal::TensorFactory` 把 `WrapExternal` 暴露给 TensorConvert 特化，又加了一个 friend。

**为什么要全部删掉？**

---

## 2. 友元的问题是什么

### 2.1 友元破坏模块边界，但编译器不警告

friend 声明是 C++ 里最软的边界机制：

- 它出现在公开头文件里，等于在 API 契约上开了个后门——"任何能 include 这个头的代码都能声明自己是 friend"？不，friend 是定义方主动开的。但问题是：**模块 A 在自己的头里写 `friend class B`，等于承认"B 有权直接读我的私有成员"**，而 B 可能在另一个模块、另一个目录、另一个 namespace 里。
- 这让依赖方向变模糊：原本 A→B 是单向依赖，但 friend 让 B 能反向访问 A 的内部实现，实际变成双向耦合。
- 更糟的是，编译器不会提醒你这是个问题。friend 是"合法的私有访问"——你写 `X` 时，`Y` 直接读 `X.private_field_`，编译器 OK，lint 工具也不报警，直到有人重构 `X` 忘了更新 friend 链，或者发现 `Y` 里藏着不该有的对 `X` 内部状态的依赖。

### 2.2 高速迭代阶段，friend 是最不该用的工具

hpcplatform 处于 API 快速调整期。这个阶段的设计原则是：

- **接口不稳定时，别引入 friend。** 接口还在变，friend 会把"暂时需要的内部访问"固化成硬依赖，每次调整类结构都得同步改 friend 声明，很容易漏。
- **friend 是稳定架构的最后手段。** 只有当接口已经稳定、证明了"公开 accessor 会泄漏实现细节"、且"把该能力移入类自身做不到"时，才考虑 friend。典型场景：std::ostream 和 operator<<，需要访问 ostream 的私有 buffer，但 ostream 是稳定接口、operator<< 只能是自由函数（不能是成员）。

hpcplatform 完全不符合这个场景——它在高速迭代。

### 2.3 friend 让重构成本指数上升

假设今天 Context 有 3 个 friend：Tensor、OpContext、Xxx。明天 Context 想改 `backend_` 的存储方式——比如从 `shared_ptr<Backend>` 换成 `unique_ptr` 加裸指针，或者把 Backend 拆成两个小接口。需要检查：

1. 3 个 friend 类的所有文件
2. 这些类的所有下游使用点
3. 每个地方对 `backend_` 的假设是什么

如果没有 friend，改 Context 内部实现只需：
1. 确保公开 accessor（`backend_ref()` / `stream()` / `GetMemoryStats()`）语义不变
2. 改内部
3. 重新编译，编译器会把所有访问 private `backend_` 的代码找出来（因为它们不能直接访问）

**friend 让编译器从"严格门卫"退化成"睁一只眼闭一只眼"，代价是重构时的隐式成本。**

---

## 3. hpcplatform 的解法：公开 accessor + 把能力移进类自身

### 3.1 `backend_ptr()` → `backend_ref()`：公开但安全

原来 Context 的 `backend_ptr()` 是 private，friend 才能调。改成 public 叫 `backend_ref()`，**不是**泄漏内部——因为：

```cpp
// context.h 里 Backend 只是前向声明
class Backend;

class Context {
 public:
  const std::shared_ptr<Backend>& backend_ref() const;  // 公开，但调用方不知道 Backend 有什么方法
 private:
  std::shared_ptr<Backend> backend_;
};
```

调用方（库外消费者）的代码里只有 `class Backend;` 前向声明，所以即使拿到了 `shared_ptr<Backend>`，也**无法调用任何 Backend 方法**——编译器根本不知道 Backend 有哪些方法。只有 `src/core/backend.h` 里有完整定义，库内部代码才能调 `CopyBuffer`、`Allocate`、`SupportsDType` 等内部方法。

这比 friend 干净得多：
- 依赖方向清晰：调用方拿一个 opaque handle，不碰内部
- 编译器强制：没有完整定义就是调不了
- 公开 accessor 的语义稳定，内部怎么存都不影响调用方

### 3.2 WrapExternal：把工厂从 `internal::TensorFactory` 移回 `Tensor` 自身

早期版本为了"Tensor 不想自己暴露 WrapExternal"，搞了一个 `internal::TensorFactory`，然后加 friend 让它能调 `Tensor::WrapExternal`（private）。这完全是多此一举：

- `Tensor::WrapExternal` 要访问的私有成员（`buffer_`、`owner_`），**它本来就是 Tensor 的静态成员方法**，天然能访问自己类的私有成员，不需要 friend
- `internal::TensorFactory` 只是个转发壳子：`static Result<Tensor> WrapExternal(...) { return Tensor::WrapExternal(...); }`——纯样板代码
- 放在 `namespace internal` 里只是靠命名约定警告"别碰"，编译器不强制

最终把 `Tensor::WrapExternal` 上移到 public，删除了 `TensorFactory` 和 `tensor_internal.h`。跨运行时适配器（TensorConvert 特化）直接调 `Tensor::WrapExternal`，零 friend、零样板、零 internal namespace。

---

## 4. 什么时候可以用 friend（hpcplatform 现在还没遇到）

只有同时满足以下三个条件时才考虑：

1. **接口已稳定**：类的公开方法集不会再大改
2. **公开 accessor 会泄漏实现**：比如需要访问一个"只用于内部优化、不属于公共语义"的成员
3. **该能力确实不能移入类自身**：比如 ostream 的 `operator<<` 必须是自由函数

三个条件缺一不可。hpcplatform 0.2.x 阶段三者都不满足，所以禁用 friend。

---

## 5. 一句话总结

> 对外头文件里零 friend。需要内部访问时，靠公开 accessor（`backend_ref()` 是 opaque handle，前向声明天然隔离）或把能力移入类自身的 static 方法（`Tensor::WrapExternal` 不需要 friend 就能访问自己的私有成员）。高速迭代阶段 friend 是最不该用的工具。
