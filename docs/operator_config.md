# 配置驱动算子对象（`hpcplatform_WITH_YAML`）

让算法研究者通过一份配置文件实例化算子对象、调参、切换执行环境（实验机 CPU / 生产机 CUDA），而无需改代码。库只负责把配置变成**已绑定参数的算子对象**；执行流程由调用方自行编排，库不构建计算图，也不假设图像算法。

完整架构背景见 [`architecture.md`](./architecture.md) §2.1、§7.3、§10。

***

## 1. 构建

默认关闭，不引入任何第三方依赖：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

启用配置驱动层（需要 yaml-cpp，例如 `apt install libyaml-cpp-dev`）：

```bash
cmake -S . -B build -Dhpcplatform_WITH_YAML=ON
```

未启用时，公共 API 仍可链接，但 `LoadOperators*` 返回 `kUnsupported`（消息含 `hpcplatform_WITH_YAML`），默认构建零回归。

***

## 2. 快速开始

```cpp
#include "hpcplatform/hpcplatform.h"
using namespace hpcplatform;

// 从文件加载；也可用 LoadOperatorsFromString(yaml_text) 内联配置。
auto loaded = LoadOperators("config.yaml");
if (!loaded.ok()) { /* loaded.status() */ return 1; }
OperatorMap ops = std::move(loaded.value());

// 按名字取具体类型的算子对象；类型不匹配或名字不存在返回 nullptr。
auto* resize = GetOperator<ResizeOperator>(ops, "downscale");
auto* blur   = GetOperator<GaussianBlurOperator>(ops, "smooth");

auto ctx = Context::Create(BackendType::kNativeCpu).value();
auto input = ctx->AllocateTensor(Shape{1, 8, 8, 3}, DataType::kFloat32).value();

// 调用方自行编排：先缩放、再模糊。每个对象已绑定自己的参数。
auto small   = resize->Run(input);          // Result<Tensor>
auto blurred = blur->Run(small.value());    // Result<Tensor>
```

可运行示例见 [`samples/05_config_operator.cpp`](../samples/05_config_operator.cpp) 与 [`samples/configs/operators.yaml`](../samples/configs/operators.yaml)。

***

## 3. 配置契约

顶层是一个 `operators` 列表，每个元素描述一个算子对象实例：

```yaml
operators:
  - name: downscale          # 实例名，必填且唯一；map 的键，GetOperator 按此查找
    op: resize               # 算子类型，必填；必须是已注册的描述符名
    backend: native_cpu      # 后端，必填：native_cpu / opencv / cuda / npu
    cpu_level: naive         # 可选，默认 naive
                              #   CPU 后端：naive / openmp / simd_generic / simd_avx2 / simd_neon / tbb
                              #   非 CPU 后端必须为 naive（否则 kInvalidArgument）
    device_index: 0          # 可选，默认 0；非 CPU 后端用于选卡
    params:                  # 算子专属参数，全部按标量字符串解析
      output_height: 2
      output_width: 2
      interpolation: bilinear
      layout: nhwc
```

字段规则：

- `name`：必填、非空、在文件内唯一；重名 → `kInvalidArgument`。
- `op`：必填；未注册的名字 → `kInvalidArgument`。
- `backend`：必填；非法名字 → `kInvalidArgument`；该后端未编译进当前构建 → `kUnsupported`。
- `cpu_level`：可选，省略视为 `naive`；非法名字 → `kInvalidArgument`。
- `device_index`：可选，省略为 `0`；非负整数，非法 → `kInvalidArgument`。
- `params`：可选（缺省为空 map）；键和值都必须是标量。每个算子的描述符自行校验必填项与类型。
- YAML 语法错误 → `kInvalidArgument`。

***

## 4. 环境优先级

执行环境按以下优先级解析，高优先级命中后低优先级**完全被忽略**：

| 优先级 | 触发条件 | 行为 |
| --- | --- | --- |
| 1 | 环境变量 `hpcplatform_BACKEND` 非空 | 所有对象共享**同一个** `Context`；`backend` / `cpu_level` / `device_index` 全部来自环境变量，配置里的环境字段被忽略 |
| 2 | 调用 `LoadOperators(path, shared_context)` 传入非空 `shared_context` | 所有对象共享该 `Context`；配置环境字段被忽略；`cpu_level` 取 `naive` |
| 3 | （默认） | 每个条目按自己的 `backend` / `cpu_level` / `device_index` 各自 `Context::Create`，对象各自持有独立 `Context` |

环境变量：

- `hpcplatform_BACKEND`：`native_cpu` / `opencv` / `cuda` / `npu`。
- `hpcplatform_CPU_LEVEL`（可选）：省略为 `naive`。
- `hpcplatform_DEVICE_INDEX`（可选）：省略为 `0`。

**无静默回退**：优先级 1 或 2 请求的后端若无法创建（未编译、设备不可用），整个 `LoadOperators` 返回错误，不会降级到配置里的后端或 CPU。例如在未编译 CUDA 的构建里设置 `hpcplatform_BACKEND=cuda` → 失败。

传入 `nullptr` 作为 `shared_context` → `kInvalidArgument`（不是"不共享"，而是参数错误）。

***

## 5. 对象模型

- `Operator`（`include/hpcplatform/operator.h`）是元数据基类，不含 `Run`（不假设输入输出元数）：
  - `name()` / `kind()` / `backend()` / `cpu_level()` / `context()`。
  - 持有 `std::shared_ptr<Context>`，可与其他对象共享。
- 每个具体算子在自己的公共头追加派生类（如 `ResizeOperator`、`GaussianBlurOperator`），提供**强类型** `Run(...)`，内部调用已有的公共自由函数，因此 `Run` 的结果与直接调用**位精确一致**。
- `OperatorMap = std::map<std::string, std::unique_ptr<Operator>>`。
- `GetOperator<T>(const OperatorMap&, name)`：名字不存在或动态类型不是 `T` 返回 `nullptr`，由调用方决定如何处理。

***

## 6. 错误语义（fail-fast）

`LoadOperators*` 在构造**任何**对象之前，先对所有条目做完整校验：

1. `op` 名能找到描述符；
2. `ValidateParams` 通过；
3. 环境能解析（`Context::Create` 成功 / 共享 context 合法）；
4. `FindKernel(desc->Kind(), backend, cpu_level)` 可用（顺带强制：非 CPU 后端 + 非 naive 级别 → `kInvalidArgument`，与 `ValidateInvocation` 一致）。

任一条失败 → 立即返回错误，`OperatorMap` 保持空，**不产生任何半成品对象**。全部通过后才逐条 `Construct` 填入 map。

错误码三分类（与全局错误模型一致）：

| 错误码 | 触发场景 |
| --- | --- |
| `kInvalidArgument` | 未知 op、缺/错参数、重名、YAML 语法错、非法 backend/cpu_level/device_index 名字、非 CPU + 非 naive 级别组合、null shared_context |
| `kUnsupported` | 请求的 backend 未编译进当前构建（如无 CUDA 构建里请求 cuda） |
| `kNotImplemented` | backend 已编译但该 `(op, level)` 没有内核实现 |

***

## 7. 支持的算子参数

### `resize`

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `output_height` | 整数 | `> 0` |
| `output_width` | 整数 | `> 0` |
| `interpolation` | 字符串 | `nearest` / `bilinear` |
| `layout` | 字符串 | `nhwc` / `nchw` |

`Run(const Tensor& input)` 返回 `Result<Tensor>`，等价于 `hpcplatform::Resize(ctx, input, output_height, output_width, interpolation, layout)`。

### `gaussian_blur`

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `kernel_size` | 整数 | 奇数且 `>= 3` |
| `sigma` | 浮点数 | 有限且 `> 0` |
| `layout` | 字符串 | `nhwc` / `nchw` |

`Run(const Tensor& input)` 返回 `Result<Tensor>`，等价于 `hpcplatform::GaussianBlur(ctx, input, kernel_size, sigma, layout)`。

### `laplacian_pyramid`

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `levels` | 整数 | `>= 1`，且每级尺寸保持 `>= 2x2` |
| `kernel_size` | 整数 | 奇数且 `>= 3` |
| `sigma` | 浮点数 | 有限且 `> 0` |
| `layout` | 字符串 | `nhwc` / `nchw` |

`Run(const Tensor& input)` 返回 `Result<LaplacianPyramid>`（`levels` 含 `levels + 1` 个张量：前 `levels` 个为带通层，最后一个为最粗残差），等价于 `hpcplatform::BuildLaplacianPyramid(ctx, input, levels, kernel_size, sigma, layout)`。只接受 `f32` / `f64` 输入。

### `fourier_transform`

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `layout` | 字符串 | `nhwc` / `nchw` |

`Run(const Tensor& input)` 返回 `Result<ComplexImage>`（`real` / `imag` 两个同形张量），等价于 `hpcplatform::FourierTransform(ctx, input, layout)`。只接受 `f32` / `f64` 输入。

### `wavelet_transform`

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `kind` | 字符串 | `haar` / `db4` |
| `layout` | 字符串 | `nhwc` / `nchw` |

`Run(const Tensor& input)` 返回 `Result<WaveletBands>`（`ll` / `lh` / `hl` / `hh`），等价于 `hpcplatform::WaveletTransform(ctx, input, kind, layout)`。输入 H/W 必须为偶数，只接受 `f32` / `f64`。

新增算子仍按同一描述符模式接入，接入步骤见 [`architecture.md`](./architecture.md) §10。

***

## 8. 环境切换示例

同一份配置，不改代码即可换后端：

```bash
# 实验机：CPU
./05_config_operator samples/configs/operators.yaml

# 生产机：整块切到 CUDA（需 CUDA 构建）
hpcplatform_BACKEND=cuda hpcplatform_DEVICE_INDEX=0 ./05_config_operator samples/configs/operators.yaml
```

设置 `hpcplatform_BACKEND` 后，配置里每个条目的 `backend` 字段都被忽略，所有对象共享同一个 CUDA `Context`。
