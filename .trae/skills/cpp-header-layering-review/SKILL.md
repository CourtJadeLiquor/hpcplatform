---
name: cpp-header-layering-review
description: Audit C++ public header layout by module, including common-type consolidation, layer boundaries, and friend-free encapsulation. Use when reviewing or reorganizing an include/ tree or deciding where a type belongs. Not for general code review.
---

# C++ 公共头分层审查

对 C++ 库的 `include/` 树做职责与模块划分审查，产出可执行的整改清单，然后落地。

## 适用与不适用

用：用户要求审查/整理对外头文件、判断某个类型该放哪个头、拆解 god class 的对外面、移除 friend、划分模块边界。

不用：普通 bug 修复、性能优化、安全审计、内部实现重构（不涉及头文件边界时）。

## 一、先划模块，再动文件

先输出模块表，再改代码。模块划分只有一个判据：**依赖方向 + 扩展落点**。

四模块范式（通用）：

| 模块 | 职责 | 收什么 |
| --- | --- | --- |
| `core/` | 类型系统 + 执行原语 | 值类型、枚举、抽象接口、门面、跨边界自由函数 |
| `ops/` | 领域能力 | 算子/功能头 + 执行上下文 + 配置驱动层 |
| `interop/` | 第三方适配 | 编译期 trait + 具体第三方特化 |
| 顶层 | 只两件事 | 聚合伞形头 + 版本头 |

硬规则：顶层**只能**有聚合头和版本头。任何具体模块的头放在顶层都是越位。

## 二、单个类型该放哪

按优先级判定，命中即停：

1. **无行为、无资源、无依赖的分类/标识/基础值**（枚举、`struct` 值类型、定容描述符）→ 收进 `core/common.h`。**不要**为单个枚举新建 `dtype.h` / `device.h` / `kernel_level.h` 这类一头一类型文件；它们只让 include 列表变长，并制造"这属于哪一层"的歧义。
2. **抽象接口或持有资源** → 独立成头（`memory.h` 的 `Buffer`/`MemoryManager`、`stream.h` 的 `Stream`）。
3. **基础错误类型** → 独立成头且**零依赖**（`Status`/`Result<T>`），允许被 `common.h` 反向依赖。
4. **只给外部类型加扩展点** → `interop/`，不进 `core/`。核心不 include 任何第三方头。
5. **配置驱动层**（把声明式配置实例化成对象）→ 属于 `ops/` 子层，与 `ops/` 平级是错的。

## 三、禁用 friend

对外头文件里 **0 friend**。任何出现 friend 的对外头都是缺陷，按下面替换：

| 原写法 | 替换 |
| --- | --- |
| `friend class X;` + `private: backend_ptr()` | 改为 `public: backend_ref()`。只要 `Backend` 在公开头里**只有前向声明**，外部拿到 `shared_ptr<Backend>` 也调不到任何 Backend 方法，封装不破 |
| `internal::XxxFactory` 转发壳 + friend | 把能力改为目标类自身的 `static` 成员。静态成员天然能访问本类私有，不需要 friend，也不需要 `namespace internal` |
| `friend` 给测试开后门 | 该能力要么提升为有文档的公共工厂（说明使用场景和生命周期），要么测试改走公共路径 |

只用 friend 的三个同时成立的场景：接口已冻结、公开 accessor 会泄漏内部语义、能力确实无法移入类自身（如 `operator<<`）。迭代期一律不满足。

## 四、静态分发 vs 虚接口

默认静态（trait 特化），除非类型在编译期确实未知。

选 trait：转换/序列化/哈希等"给已存在类型加扩展点"的场景。理由：第三方类型不可继承；编译期分发零开销；primary template 返回 `kUnsupported` 提供零成本错误报告；核心不被第三方头污染。

选虚接口：异构对象需运行时同容器调度（插件、策略注册）。此时才接受类型擦除与注册表生命周期成本。

`struct TensorConvert<T>` 用 `struct` 而非 `class` 是元编程惯例（`std::hash`/`std::is_integral` 同），默认 public 省样板。

## 五、头文件自包含

每个公共头必须自己 include 它用到的所有东西，禁止依赖传递包含。审查时逐个头单独编译是唯一可靠验证：

```bash
# 对每个公共头生成一个最小 TU 并编译
for h in $(find include -name '*.h'); do
  echo "#include \"${h#include/}\"" > /tmp/probe.cpp
  g++ -std=c++17 -Iinclude -fsyntax-only /tmp/probe.cpp || echo "FAIL $h"
done
```

常见隐患：宏里用 `std::cout` 却没 include `<iostream>`；用 `size_t` 却没 `<cstddef>`；用 `uint8_t` 却没 `<cstdint>`。

## 六、顺手修掉被暴露的真实缺陷

头文件改动常让"靠传递包含侥幸编译"的代码暴露。此时**不要**补一个 include 糊过去，先看它是不是设计问题：

- 库的宏向 stdout/stderr 打印 → 删除打印，库只返回状态。
- `RETURN_IF_ERROR` 无条件 `return` → 会导致宏之后的代码变死代码，改为只在失败时返回。
- 修完必须确认被测代码真的执行了：改前测试"通过"可能是因为函数提前返回，改后才真正跑到断言。

宏的标准形态：

```cpp
#define hpcplatform_RETURN_IF_ERROR(expr)                        \
  do {                                                           \
    const ::hpcplatform::Status hpcplatform_status = (expr);     \
    if (!hpcplatform_status.ok()) return hpcplatform_status;     \
  } while (0)
```

## 七、执行顺序

1. 盘点：`find include -type f | sort`，同时列出 `src/` 顶层目录，看公共面与实现面是否一一对应。
2. 统计每个头的引用次数（grep），识别"一头一类型"的碎头。
3. 产出**模块表 + 迁移表**（旧路径 → 新路径 → 理由），先给用户看。
4. 批量迁移 include：用脚本按"删除多个旧 include → 在第一个旧 include 位置插入合并后的新 include"，避免手工漏改。脚本只处理精确的 include 行字面量，不做模糊匹配。
5. 删旧头、删随之失效的 `.cpp`、更新构建脚本。
6. 构建 → 修编译错误 → 跑全量测试。
7. 同步文档的目录树与类图；废弃内容直接删除，不保留"历史版本"段落。

## 八、验证与汇报

```bash
# libstdc++ 版本冲突时（本机 cmake 需要系统库）
env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cmake --build build -j8
env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ctest --test-dir build -j4 --output-on-failure
```

汇报必须包含：模块表、迁移文件清单、每个删除项的理由、friend 数量前后对比、构建与测试结果，以及预先存在的失败用例（标明与本次无关，不擅自修改）。

## 反模式清单

- 为一个枚举新建一个头，`core/` 里堆满 `dtype.h`/`device.h`/`kernel_level.h`
- 顶层放具体模块头（`operator.h` 应属于 `ops/`）
- 工具类（计时器、日志、统计）塞进 `core/`：库不应替应用做这类事
- 用 `namespace internal` + friend 当"半私有"通道
- 单数/复数目录平级（`src/operator/` 与 `src/ops/`）
- 为"未来可能扩展"预先建注册表/插件机制
- 头文件靠传递包含编译
- 库代码自己打印错误
