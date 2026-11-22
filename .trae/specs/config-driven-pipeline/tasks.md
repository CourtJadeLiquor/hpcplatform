# mycv 配置驱动算子对象 - 实现计划

## Task 1：反向 名→枚举 映射工具（core 公共头增量）
- **Status**：`pending`
- **Priority**：high
- **Depends On**：None
- **Description**：
  - `include/mycv/core/device.h`：增加 `Result<BackendType> BackendTypeFromName(const std::string&)`。
  - `include/mycv/core/kernel_level.h`：增加 `Result<CpuKernelLevel> CpuKernelLevelFromName(const std::string&)`。
  - `include/mycv/ops/op.h`：增加 `Result<OpKind> OpKindFromName(const std::string&)`。
  - `include/mycv/ops/image.h`：增加 `InterpolationModeName` / `InterpolationModeFromName`、`ImageLayoutFromName`。
  - 均为纯值映射 inline 函数（append-only，不改枚举值），未知名返回 `kInvalidArgument`。
- **Acceptance Criteria Addressed**：AC-3、AC-5、基础支撑 FR-2/FR-7
- **Test Requirements**：
  - `rule` TR-1.1：每个 name→enum 与既有 enum→name 往返一致；未知名返回非 OK。证据：`tests/operator` 或 `tests/core` 用例。

## Task 2：framework 描述符接口与注册表
- **Status**：`pending`
- **Priority**：high
- **Depends On**：None
- **Description**：
  - 新增 `src/framework/op_descriptor.h` + `src/framework/op_descriptor.cpp`：
    - `using ParamMap = std::map<std::string, std::string>;`。
    - 抽象基类 `OpDescriptor`：`virtual const char* Name() const = 0;`、`virtual OpKind Kind() const = 0;`、`virtual Status ValidateParams(const ParamMap&) const = 0;`、`virtual Result<std::unique_ptr<Operator>> Construct(const std::shared_ptr<Context>&, CpuKernelLevel, const ParamMap&) const = 0;`。
    - 注意：`Operator` 是公共类型（Task 3 声明），描述符接口依赖公共头 `mycv/operator.h` —— framework 依赖公共稳定类型符合 R3。
    - 注册表：`void RegisterOpDescriptor(std::unique_ptr<OpDescriptor>)`、`const OpDescriptor* FindOpDescriptor(const std::string&)`，沿用 `op_registry` 的 `std::call_once` 显式注册模式。
  - 内核可用性检查由 loader 调 `FindKernel(desc->Kind(), backend, level)` 完成（复用注册机），描述符不重复实现。
- **Acceptance Criteria Addressed**：AC-7、AC-8
- **Test Requirements**：
  - `rule` TR-2.1：注册后 `FindOpDescriptor` 命中；未注册返回 nullptr。证据：单元测试。

## Task 3：公共 API 头 + 构建骨架（YAML 门控 + stub）
- **Status**：`pending`
- **Priority**：high
- **Depends On**：Task 2
- **Description**：
  - 新增公共头 `include/mycv/operator.h`：
    - 声明 `class Operator`（元数据基类：`name/kind/backend/cpu_level/context`，持有 `shared_ptr<Context> context_` 与 `CpuKernelLevel cpu_level_`）。
    - `using OperatorMap = std::map<std::string, std::unique_ptr<Operator>>;`。
    - 声明 `Result<OperatorMap> LoadOperators(const std::string& config_path);`、`LoadOperatorsFromString(const std::string& yaml);`、`LoadOperators(const std::string& config_path, const std::shared_ptr<Context>& shared_context);`。
    - 模板 `template<class T> T* GetOperator(const OperatorMap&, const std::string& name);`（header-only）。
  - `include/mycv/mycv.h` 增加一行 include。
  - 顶层 `CMakeLists.txt`：增加 `option(MYCV_WITH_YAML ... OFF)`；`if(MYCV_WITH_YAML) find_package(yaml-cpp REQUIRED)` + 挂载 `src/operator` + 链接 + 定义 `MYCV_WITH_YAML=1`；未启用时编译 stub 实现使 `LoadOperators` 返回 `kUnsupported`。
  - 新增描述符登记清单骨架 `src/operator/register_descriptors.cpp`（显式列表，v1 先空）。
- **Acceptance Criteria Addressed**：AC-6、AC-8、FR-1/FR-8
- **Test Requirements**：
  - `rule` TR-3.1：默认构建调用 `LoadOperators` 返回 `kUnsupported` 且信息含 `MYCV_WITH_YAML`。证据：默认构建链接 + 测试。

## Task 4：resize 与 gaussian_blur 的 concrete Operator + 描述符
- **Status**：`pending`
- **Priority**：high
- **Depends On**：Task 2、Task 3
- **Description**：
  - `src/ops/resize/` 新增 `resize_operator.h/.cpp`（或在现有文件追加）：
    - `class ResizeOperator : public Operator`，持有 `ResizeParams`（强类型）；`Result<Tensor> Run(const Tensor&) const` 内部调 `mycv::Resize`。
    - `class ResizeDescriptor : public OpDescriptor`：`ValidateParams` 校验 output_height/output_width/interpolation/layout 必填与类型；`Construct` 解析参数并 `make_unique<ResizeOperator>`。
    - `void RegisterResizeDescriptor();` 并在 `register_descriptors.cpp` 加一行。
  - `src/ops/gaussian_blur/` 同理：`GaussianBlurOperator` + `GaussianBlurDescriptor` + 登记。
  - concrete 类的公共声明（`ResizeOperator` / `GaussianBlurOperator`）需对外可见——放在对应公共头 `include/mycv/ops/resize.h` / `gaussian_blur.h` 追加（纯增量），或新建 `include/mycv/ops/resize_operator.h`。选定：在现有公共头追加类声明（保持一个算子一个公共头）。
- **Acceptance Criteria Addressed**：AC-1、AC-2、AC-7
- **Test Requirements**：
  - `rule` TR-4.1：`ResizeOperator::Run` 与直接 `Resize` 位精确一致；参数缺失/类型错 `ValidateParams` 返回 `kInvalidArgument`。证据：单元测试。
  - `rule` TR-4.2：`GaussianBlurOperator` 同验证。证据：单元测试。

## Task 5：YAML 解析器
- **Status**：`pending`
- **Priority**：high
- **Depends On**：Task 3
- **Description**：
  - 新增 `src/operator/operator_config.h`：内部结构 `OperatorInstanceConfig{name, op, backend, cpu_level, device_index, ParamMap params}`、`OperatorsConfig`。
  - 新增 `src/operator/yaml_parser.cpp`：yaml-cpp 解析为 `OperatorsConfig`；字段缺失/类型错/语法错/重名 → `kInvalidArgument`。
  - 仅 `MYCV_WITH_YAML` 时编译。
- **Acceptance Criteria Addressed**：AC-5、FR-2
- **Test Requirements**：
  - `rule` TR-5.1：合法配置解析正确；缺必填/重名/语法错误返回 `kInvalidArgument`。证据：解析器单元测试。

## Task 6：LoadOperators 实现（环境解析 + fail-fast 校验 + 构造）
- **Status**：`pending`
- **Priority**：high
- **Depends On**：Task 4、Task 5
- **Description**：
  - 新增 `src/operator/load_operators.cpp`（`MYCV_WITH_YAML` 时替换 stub）：
    - 解析 YAML → `OperatorsConfig`。
    - 环境解析：若环境变量 `MYCV_BACKEND` 设 → 构造共享 Context（所有条目复用，忽略配置环境字段）；否则若调用方传入 `shared_context` → 复用；否则每条 `Context::Create(backend, device_index)`。
    - 整体 fail-fast 校验（不构造任何对象前完成）：每条 op 名可查描述符 → `ValidateParams` → `FindKernel(kind, backend, level)` 可用；`Context::Create` 成功。任一失败返回错误，`OperatorMap` 保持空。
    - 全部通过后逐条 `Construct` 填入 map，返回。
  - 非 CPU 后端 + 非 naive level → `kInvalidArgument`（与 `ValidateInvocation` 一致）。
- **Acceptance Criteria Addressed**：AC-1、AC-2、AC-3、AC-4、AC-5、AC-8
- **Test Requirements**：
  - `rule` TR-6.1：加载合法配置得到对象，`Run` 等价直接调用（AC-1）。
  - `rule` TR-6.2：两个不同参数对象输出不同 shape（AC-2）。
  - `rule` TR-6.3：未编译后端配置返回错误且 map 空（AC-4）。
  - `rule` TR-6.4：未知 op / 缺参数 / 重名返回 `kInvalidArgument` 且 map 空（AC-5）。
  - `rule` TR-6.5：环境变量 / 共享 Context 模式下对象 `backend()` 正确（AC-3，CPU 路径）。

## Task 7：测试集 tests/operator
- **Status**：`pending`
- **Priority**：high
- **Depends On**：Task 6
- **Description**：
  - 新增 `tests/operator/` + `CMakeLists.txt`，用 `tests/test_framework.h`。
  - 覆盖 AC-1 ~ AC-6（CPU 路径）；CUDA 路径在 CUDA 构建树验证（AC-3 的 (b)(c)）。
  - 配置以字符串内联（`LoadOperatorsFromString`）避免外部路径依赖；环境变量用例在子进程或 setenv 控制。
- **Acceptance Criteria Addressed**：AC-1 ~ AC-6
- **Test Requirements**：
  - `rule` TR-7.1：`tests/operator` 全部用例在默认构建 + `MYCV_WITH_YAML=ON` 构建通过。证据：ctest 输出。

## Task 8：示例 samples/05 + 示例 YAML
- **Status**：`pending`
- **Priority**：medium
- **Depends On**：Task 6
- **Description**：
  - 新增 `samples/05_config_operator.cpp`：演示 `LoadOperators` → `GetOperator` → 持有多个对象 → 用户自行调用（含一个 cpu + 一个不同参数的对比）。
  - 新增 `samples/configs/operators.yaml` 示例（中文注释）。
  - `samples/CMakeLists.txt` 挂载（`MYCV_WITH_YAML` 门控）。
- **Acceptance Criteria Addressed**：AC-3、G3
- **Test Requirements**：
  - `rule` TR-8.1：示例在 `MYCV_WITH_YAML` + CPU 构建下可编译运行并产生正确输出。证据：手动运行输出。

## Task 9：文档更新
- **Status**：`pending`
- **Priority**：medium
- **Depends On**：Task 6
- **Description**：
  - 更新 `docs/architecture.md`：分层图新增 operator 层；§7 增加“描述符注册表（高阶，构造算子对象）”小节；§10 扩展指南增加“新增算子对象接入配置”步骤。
  - 新增 `docs/operator_config.md`（中文）：配置契约、环境优先级、对象模型、错误语义、示例。
- **Acceptance Criteria Addressed**：AC-7、AC-8
- **Test Requirements**：
  - `rubric` TR-9.1：维度“文档与实现一致性”；scale 1-5；anchors 1=多处不一致，3=主体一致少量遗漏，5=完全一致且覆盖扩展步骤；threshold >= 4；证据：文档审查核对代码。
