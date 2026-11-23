# hpcplatform 配置驱动算子对象 - 产品需求文档

## Overview
- **Summary**：在现有 `(OpKind, BackendType, CpuKernelLevel) → OpKernelFn` 内核注册机之上，新增一层“算子参数配置 → 算子对象”的工厂能力。用户从 YAML 配置文件读出一组命名算子实例（算子类型 + 执行环境 + 参数），库查注册机解析内核并构造出持有该参数的算子对象；用户持有多个对象后**自行决定执行流程**。
- **Purpose**：让科研人员通过改配置文件调参、切换执行环境，无需改 C++；可同时持有多个不同参数 / 不同后端的算子对象做对比实验；库只在算子层面提供工具，不介入业务执行流程。
- **Target Users**：算法 / 科研人员（懂算法实验，不熟悉 C++ 算子实现）；库维护者。

## Goals
- G1：一份 YAML 配置可定义多个命名算子实例，每条含 op 类型、执行环境（backend / cpu_level / device_index）、参数。
- G2：提供 `LoadOperators` 加载配置 → 查注册机构造算子对象，返回 `name → Operator` 映射。
- G3：算子对象持有预绑定的参数与执行环境，暴露**类型化** `Run`，用户自行编排调用顺序与数据流。
- G4：可同时持有多个不同参数 / 不同后端的对象；环境来源支持“每条配置自带 / 全局共享 Context / 环境变量”三级覆盖。
- G5：不建计算图、不假设图像算法、不改变 `KernelKey` 三元组注册机、不引入 core 第三方依赖（YAML 可选门控）。
- G6：错误三分类 + fail-fast：加载时整体校验，任一算子不可用则整体失败、不返回半初始化对象；不静默回退。

## Non-Goals
- NG1：不构造计算图 / DAG / 节点编排；不提供 `RunPipeline` 式一键执行；数据流由用户掌控。
- NG2：不做运行时自动探测最优后端、不做偏好链 / 自动回退（环境变量是显式授权的覆盖，不是探测）。
- NG3：v1 不做多入多出算子对象（fourier / wavelet / inverse_* / laplacian_pyramid）；仅做单入单出的 `resize` 与 `gaussian_blur`，其余后续按同模式增量。
- NG4：不做 Python 绑定 / 命令行 runner；不做配置生成器 / GUI。
- NG5：不修改 `KernelKey` 与 `FindKernel` 语义；core 不依赖 yaml-cpp。
- NG6：未启用 `hpcplatform_WITH_YAML` 时 `LoadOperators` 返回 `kUnsupported`。

## Background & Context
- 注册机 `src/framework/op_registry.{h,cpp}`：`(OpKind, BackendType, CpuKernelLevel) → OpKernelFn`，`FindKernel` 对未命中严格三分类（架构 §7.1）。本次不改。
- 现有公共 API 是自由函数（`hpcplatform::Resize(ctx, input, h, w, ...)`），每次调用需重传全部参数；科研人员当前手写 C++ 调度（见 `samples/02_ops_pipeline.cpp`）。
- 架构原则 #1：显式优先于隐式；原则 #6：错误三分类、不静默降级。
- 正 / 逆变换共享 `OpKind`（方向由参数标志区分），但 v1 不涉及。

## 配置契约（v1）
YAML，顶层为 `operators` 列表，每条一个命名算子实例：

```yaml
operators:
  - name: resize_256          # 唯一实例名，作为返回 map 的 key
    op: resize                # 描述符名
    backend: native_cpu       # native_cpu | opencv | cuda
    cpu_level: openmp         # 可选，默认 naive；仅 native_cpu 有意义
    device_index: 0           # 可选，默认 0
    params:
      output_height: 256
      output_width: 256
      interpolation: bilinear  # nearest | bilinear
      layout: nhwc             # nhwc | nchw

  - name: blur_3
    op: gaussian_blur
    backend: native_cpu
    cpu_level: naive
    params:
      kernel_size: 3
      sigma: 1.0
      layout: nhwc
```

- v1 描述符：`resize`、`gaussian_blur`（均单入单出）。
- 参数字段名与对应公共函数形参一致；标量以字符串承载后转类型。
- `name` 在配置内唯一（重复 → `kInvalidArgument`）。

## 算子对象模型
- 公共基类 `Operator`（`include/hpcplatform/operator.h`）：只持元数据 —— `name()`、`kind()`、`backend()`、`cpu_level()`、`context()`（返回持有的 `shared_ptr<Context>`）。**不预设 `Run` 签名**（不假设单入单出）。
- 每个算子一个 concrete 子类（v1：`ResizeOperator`、`GaussianBlurOperator`），持有强类型参数 + `shared_ptr<Context>` + `CpuKernelLevel`，提供类型化 `Result<Tensor> Run(const Tensor&) const`，内部复用现有公共入口（校验 + 形状推断 + 分配 + 调内核），仅参数预绑定。
- 用户通过 `GetOperator<T>(map, name)` 按名取类型化指针（缺失或类型不符返回 `nullptr`）。

## 环境解析优先级（高 → 低）
1. **环境变量**：若设置 `hpcplatform_BACKEND`（可选 `hpcplatform_CPU_LEVEL`、`hpcplatform_DEVICE_INDEX`），则所有对象共享该环境，忽略配置里的 backend / cpu_level / device_index；该后端不可用则整体失败（不回退）。
2. **全局共享 Context**：调用 `LoadOperators(config, shared_context)` 时，所有对象复用该 Context，配置里的环境字段被忽略。
3. **每条配置自带**：默认模式，每个对象持有自己的 `Context`。

生效环境在对象上可查（`op->backend()` / `cpu_level()`），不静默。

## Functional Requirements
- **FR-1**：公共头 `include/hpcplatform/operator.h` 声明 `Operator` 基类、`using OperatorMap = std::map<std::string, std::unique_ptr<Operator>>;`、`LoadOperators(config_path)` / `LoadOperatorsFromString(yaml)` / `LoadOperators(config_path, shared_context)` 三个重载、`GetOperator<T>` 模板。
- **FR-2**：YAML 配置解析为算子实例列表；字段缺失 / 类型错 / 语法错 → `kInvalidArgument`。
- **FR-3**：描述符注册表：`op name → OpDescriptor`，描述符提供 `ValidateParams(ParamMap)`、内核可用性检查（`FindKernel`）、`Construct(context, cpu_level, ParamMap) → unique_ptr<Operator>`；沿用显式清单注册（不静态自注册）。
- **FR-4**：加载时整体校验（fail-fast）：所有实例的 op 名已知、参数合法、`(op, backend, level)` 内核在注册机中可用、`Context::Create` 成功；任一失败则整体返回错误，**不返回任何对象**。
- **FR-5**：环境解析按上述三级优先级；环境变量 / 共享 Context 模式下所有对象共享同一环境。
- **FR-6**：构造出的对象持有预绑定参数；`Run` 复用现有公共入口逻辑，结果与直接调用公共函数一致。
- **FR-7**：用户可持有多个对象（不同参数 / 不同后端）并自行决定调用顺序。
- **FR-8**：未启用 `hpcplatform_WITH_YAML` 时 `LoadOperators` 可链接，返回 `kUnsupported("LoadOperators requires hpcplatform_WITH_YAML")`。

## Non-Functional Requirements
- **NFR-1**：默认构建（无 YAML / 无可选后端）编译通过，既有测试全过。
- **NFR-2**：纯增量，不改现有公共 API 签名与 ABI。
- **NFR-3**：core 零第三方依赖不变；yaml-cpp 仅 `hpcplatform_WITH_YAML` 时引入。
- **NFR-4**：代码注释全英文；Markdown 文档中文。
- **NFR-5**：遵守 SOLID 与编码规范；不可变变量带 const 且置前（用户规则）。
- **NFR-6**：不静默降级 / 回退；环境变量指定的后端不可用即报错。

## Constraints
- **Technical**：C++17；`-fno-exceptions`；显式注册模式；`-Wall -Wextra -Wpedantic`。
- **Dependencies**：yaml-cpp 可选（`find_package(yaml-cpp REQUIRED)` 仅 `hpcplatform_WITH_YAML`）。
- **Layering**：遵守 R1–R6；`Operator` 基类与公共 API 在 `include/hpcplatform/`；描述符接口 + 注册表在 `src/framework/`（不依赖具体算子）；YAML 解析与加载器在新增 `src/operator/`；各算子 concrete 类与描述符留在算子文件夹内。

## Assumptions
- A1：用户在 C++ 中调用 `LoadOperators` 并持有 `OperatorMap`。
- A2：输入张量由用户在对象对应 backend / device 上预先分配；不在对象内做跨后端拷贝。
- A3：同一 `OperatorMap` 内的对象可来自不同后端（各自 Context）；用户负责跨对象的数据搬运（如需要）。

## Acceptance Criteria

### AC-1：配置加载构造算子对象并可执行
- **Type**：`rule`
- **Given**：CPU 构建 + YAML 配置定义 `resize_256` 与 `blur_3`
- **When**：`auto ops = LoadOperators(path).value(); auto* r = GetOperator<ResizeOperator>(*ops, "resize_256"); r->Run(input);`
- **Then**：对象存在且类型正确；`Run` 结果与直接调用 `Resize(ctx, input, 256, 256, ...)` 位精确一致
- **Pass Condition**：等价性测试通过
- **Evidence**：`tests/operator`

### AC-2：持有多个不同参数对象
- **Type**：`rule`
- **Given**：配置定义 `resize_256` 与 `resize_512`（同 op 不同参数）
- **When**：分别取出并 `Run` 同一输入
- **Then**：两个对象各自按绑定参数产生正确的不同尺寸输出，互不干扰
- **Pass Condition**：两输出 shape 分别为 256×256 与 512×512
- **Evidence**：`tests/operator`

### AC-3：环境切换（配置自带 / 共享 Context / 环境变量）
- **Type**：`rule`
- **Given**：同一份配置，分别 (a) 每条自带 `native_cpu`；(b) 传入共享 cuda Context（CUDA 构建）；(c) 设置 `hpcplatform_BACKEND=cuda`
- **When**：加载并检查对象
- **Then**：(a) 对象 `backend()` 为 native_cpu；(b)(c) 所有对象 `backend()` 为 cuda 且共享同一 Context；CUDA 构建下 (b)(c) 可运行
- **Pass Condition**：三种模式对象 `backend()` 符合预期
- **Evidence**：`tests/operator` + CUDA 构建树验证

### AC-4：内核不可用 fail-fast 且不返回对象
- **Type**：`rule`
- **Given**：配置请求未编译的 `(op, backend, level)`（如 cuda 但未开 `hpcplatform_WITH_CUDA`）
- **When**：`LoadOperators`
- **Then**：返回 `kUnsupported` 或 `kNotImplemented`，`OperatorMap` 为空（不返回半初始化对象）
- **Pass Condition**：返回错误码且 map 为空
- **Evidence**：`tests/operator` 错误路径

### AC-5：配置错误显式报错
- **Type**：`rule`
- **Given**：未知 op 名 / 缺必填参数 / 参数类型错 / 重名 / YAML 语法错 / 环境变量指定不可用后端
- **When**：`LoadOperators`
- **Then**：返回 `kInvalidArgument` 或对应错误码，不返回对象
- **Pass Condition**：各场景返回正确错误码
- **Evidence**：`tests/operator` 错误路径

### AC-6：未启用 YAML 时可链接并返回 Unsupported
- **Type**：`rule`
- **Given**：未开 `hpcplatform_WITH_YAML` 的构建
- **When**：调用 `LoadOperators`
- **Then**：链接成功，返回 `kUnsupported`，信息含 `hpcplatform_WITH_YAML`
- **Pass Condition**：默认构建编译通过且行为可测
- **Evidence**：默认构建 + 测试

### AC-7：开闭原则 —— 新增算子接入成本
- **Type**：`rubric`
- **Dimension**：新增一个算子对象所需的共享文件改动数
- **Scale**：1-5
- **Anchors**：1 = 需改 framework / loader 多处；3 = 算子文件夹内加 concrete 类 + 描述符 + 登记清单一行；5 = 零共享改动、仅算子自治文件夹
- **Pass Threshold**：>= 4
- **Evidence**：代码审查 + 扩展指南文档

### AC-8：分层与依赖合规
- **Type**：`rule`
- **Given**：完整改动
- **When**：检查 include 与构建依赖
- **Then**：core 不依赖 yaml-cpp / ops；framework 不依赖具体算子；公共头不 include `src/**`；默认构建不链接 yaml-cpp
- **Pass Condition**：符合 R1–R6
- **Evidence**：构建依赖审查

### AC-9：现有回归不破坏
- **Type**：`rule`
- **Given**：改动后全量构建（默认 + 全开 + ASan）
- **When**：运行既有测试
- **Then**：既有测试 100% 通过
- **Pass Condition**：ctest 全过
- **Evidence**：ctest 输出

## Open Questions
- [ ] 环境变量命名是否定为 `hpcplatform_BACKEND` / `hpcplatform_CPU_LEVEL` / `hpcplatform_DEVICE_INDEX`？（暂定，待确认）
- [ ] v1 之后多入多出算子的 `Run` 签名与取用方式是否沿用“类型化 + `Get<T>`”？（v1 不涉及，留作后续 spec）
- [ ] 是否需要 `LoadOperators` 支持部分成功（跳过不可用算子）而非整体失败？当前定 fail-fast 整体失败，待确认。
