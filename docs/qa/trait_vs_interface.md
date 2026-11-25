# 静态分发 vs 运行时分发：为什么 TensorConvert 是 trait

> 相关代码：`include/hpcplatform/interop/conversion.h`、`include/hpcplatform/interop/opencv.h`、`include/hpcplatform/core/tensor.h`（`From`/`Into` 声明）
> 结论先行：调用方在编译期就知道要转成什么类型，用 trait（静态 polymorphism）比虚接口（运行时 polymorphism）更干净、更轻、更安全。

---

## 1. 问题：hpcplatform 要给 cv::Mat / Eigen::Matrix / 未来的 NPU 句柄加转换

Tensor 要能和各种外部数据结构互转。两种直觉方案：

**方案 A：虚接口**
```cpp
class IConvert {
 public:
  virtual Result<Tensor> From(const void* external, Context& ctx) = 0;
  virtual ~IConvert() = default;
};

// 然后有一个注册表：
RegisterConverter<cv::Mat>(new CvMatConverter());
```

**方案 B：trait（静态 polymorphism）**
```cpp
template <typename T>
struct TensorConvert {
  static Result<Tensor> From(const T& external, Context& context);  // primary template 返回 Unsupported
  static Result<T> Into(const Tensor& tensor);
};

// 特化：
template <> struct TensorConvert<cv::Mat> { /* 实现 */ };
```

hpcplatform 用了 B。为什么？

---

## 2. 虚接口方案的三个硬问题

### 2.1 外部类型无法继承 IConvert

cv::Mat、Eigen::Matrix、第三方 NPU DeviceHandle 都是只读的——你改不了它们的头文件，加不了基类。虚接口方案要解决这个，就只能退化成"类型擦除 + 注册查找"：

```cpp
// 伪代码，真实实现更脏
class AnyHandle {
  void* ptr_;
  const std::type_info& ti_;
};

class IConvert {
 public:
  virtual bool Matches(const std::type_info& ti) = 0;
  virtual Result<Tensor> From(const AnyHandle& external, Context& ctx) = 0;
};
```

然后要维护一个 `std::vector<std::unique_ptr<IConvert>>` 注册表，每次 `Tensor::From(mat, ctx)` 都要：
1. 把 cv::Mat 装进 AnyHandle（void* + type_info）
2. 遍历注册表，挨个 `Matches`
3. 找到后从 void* reinterpret_cast 回 cv::Mat*
4. 调 From

**每一步都有类型擦除和重解释**，编译器帮不上任何忙。任何一步 cast 错了就是 UB，而这一切都藏在 `Tensor::From<cv::Mat>` 的表面之下。

### 2.2 运行时开销 + 注册表生命周期

- 每次转换都要遍历注册表 + type_info 比较 + virtual dispatch。对高频算子路径，这是纯浪费。
- 注册表的生命周期、线程安全、"谁负责注册/注销"全是额外复杂度。
- 如果有两个转换同时注册了 `const cv::Mat&` 和 `cv::Mat`（按值），哪个先调？注册表顺序就是行为，没有编译期约束。

### 2.3 Tensor 要持有或知道注册表

虚接口方案里，Tensor 或者 Context 必须能拿到注册表——要么 Tensor 里存一个指针，要么 Context 里存，要么全局单例。不管哪种，核心类型和注册表机制耦合了。

---

## 3. trait 方案为什么对

### 3.1 零类型擦除，编译器全程帮忙

```cpp
// Tensor::From<cv::Mat>(mat, ctx) 展开后等价于：
TensorConvert<std::decay_t<cv::Mat>>::From(mat, ctx);

// 编译器直接定位到 cv::Mat 特化，零开销
// 如果没有特化，primary template 返回 Unsupported（运行时可诊断）
```

- 没有 void*，没有 reinterpret_cast，没有 type_info 比较
- 没有注册表，没有生命周期问题，没有注册顺序问题
- 编译器在实例化时就知道调哪个，一切确定

### 3.2 零运行时开销

模板特化就是静态成员函数调用，编译期内联后和直接写 `CvMatFromTensor(mat, ctx)` 没有任何区别。对高频路径零成本。

### 3.3 零耦合

Tensor 只声明 template `<typename T> static Result<Tensor> From(const T&, Context&);` 这一个声明，不 include 任何 interop 头——它根本不知道有 cv::Mat 的存在。特化在 `interop/opencv.h` 里，消费者 include 了这个头才会生效。core 不依赖 interop，interop 依赖 core，依赖方向单向干净。

### 3.4 primary template = 零开销的错误报告

```cpp
template <typename T, typename Enable = void>
struct TensorConvert {
  static Result<Tensor> From(const T& external, Context& context) {
    return Status::Unsupported("no Tensor::From conversion registered for this type");
  }
  // Into 同理
};
```

没有特化 → primary template 返回 kUnsupported → 消费者拿到一个清晰的错误状态，而不是运行时崩溃或静默不工作。这比"忘记注册 converter"的 UB 安全得多。

---

## 4. 什么时候该用虚接口（hpcplatform 没遇到）

虚接口适合：

- **异构对象运行时存入同一个容器**：比如 `vector<unique_ptr<IPlugin>>`，需要统一调度不同插件
- **类型在编译期确实未知**：比如从配置文件加载插件，插件类型只能在运行时决定

hpcplatform 的场景恰恰相反：调用方在写 `Tensor::From<cv::Mat>(mat, ctx)` 时，**类型完全已知**，编译期就确定了要调哪个特化。这正是 trait 方案的最佳适用点。

---

## 5. 静态分发也叫 CRTP 吗？

不完全是。CRTP（Curiously Recurring Template Pattern）是静态多态的一种特定形式：基类模板 `Derived` 参数化，派生类继承 `Base<Derived>`，基类里通过 `static_cast<Derived*>(this)` 转调。

hpcplatform 用的是**trait 特化**——没有基类，没有继承，只有 `template <typename T>` + 特化。它和 CRTP 同属"静态 polymorphism"家族，但更轻、更灵活，特别适合"给第三方类型加扩展点"的场景。

---

## 6. 一句话总结

> 调用方在编译期就知道要转成什么类型——用 trait 特化，编译器在实例化时直接定位到正确实现，零开销、零类型擦除、零注册表、依赖方向单向。虚接口只适合编译期确实未知类型的场景。
