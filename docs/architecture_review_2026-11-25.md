# HPCPlatform 架构评审报告（2026-11-25）

> 评审方法：基于 staff-engineer-mode 的 `architecture-decisions`（主）+ `production-readiness-review`（辅）路由，
> 对仓库全量源码、构建系统、测试与文档做静态评审；所有结论均带 `file:line` 级证据。
> 评审性质为设计级架构评审（非 diff 评审）。

## 总评

**整体评级：优秀。距"生产级"的差距不在架构与抽象，而在工程化外圈（CI、发布流程）——是"补门禁"而非"改设计"。**

- 五层结构（公共 API → core → framework → backends/ops → 装配）物理隔离清晰，依赖方向单向。
- SOLID 五原则全部达标；项目规范（禁 friend、const 约定、无异常错误模型）执行到位。
- NPU 以显式 `Unsupported` 占位而非伪实现，规避了最常见的 LSP 破坏模式。

## 系统图与依赖方向

```text
samples / tests / (外部调用方)
        │
include/hpcplatform/          ← 公共 API 契约（core, ops, interop, version.h）
        │
src/core/                     ← 核心实现 + 内部契约（backend.h, copy.h）
        ↑                          Backend 抽象在此，不在公共头 —— 依赖方向正确
src/framework/                ← op_registry / op_descriptor（分发框架）
        ↑
src/backends/{cpu,cuda,opencv,npu}   ← 全部依赖 core，core 不反向依赖
src/ops/{resize,gaussian_blur,wavelet,fourier,laplacian_pyramid}  ← 算子自治目录
```

**限界上下文**：Tensor/内存（core）、后端执行（backends → core，conformist）、
算子实现（ops → framework registry，customer/supplier）、
配置加载（loader → descriptor，anti-corruption layer 隔离 YAML 模型）。
各上下文职责与翻译面清晰。

## SOLID 五原则逐项结论

| 原则 | 结论 | 关键证据 |
| --- | --- | --- |
| SRP 单一职责 | 达标 | [memory.h](../include/hpcplatform/core/memory.h) 只管分配抽象；[tensor.h](../include/hpcplatform/core/tensor.h) 只做形状/类型/步长描述符，所有权分离到 `Buffer` |
| OCP 开闭 | 基本达标（一处已知代价） | 算子经 `op_registry` 的 `(OpKind, BackendType, CpuKernelLevel)` 三元组注册分发，新增算子只加自治目录；例外：新增后端需改 `src/backends/factory.cpp` 的 switch 与 `BackendType` 枚举 —— 二门决策，可接受，不建议改为自动注册 |
| LSP 里氏替换 | 达标 | 四后端实现同一 `src/core/backend.h` 纯虚契约；NPU 占位显式返回 `Unsupported`（`src/backends/npu/npu_backend.cpp`）而非伪造实现，并配 `SupportsDType`/`SupportsKernelLevel` 能力查询，无静默降级 |
| ISP 接口隔离 | 达标 | `Backend` 接口宽但内聚（内存/流/拷贝/能力同属一个语义域）；内部契约（`copy.h`、`tensor_iterator.h`）明确不进公共头 |
| DIP 依赖倒置 | 达标 | 依赖方向 backends→core 单向（`src/core/backend.h` 头部注释明文声明）；配置经 `op_descriptor.h` 解耦 YAML 与实现 |

### 项目规范符合性

- 源码 **0 处 friend**（满足"禁止友元类"规范；`docs/qa/friend_classes.md` 记录了主动移除决策）。
- 错误处理统一 `Status`/`Result<T>`，核心库无异常（兼容 `-fno-exceptions` 的嵌入式工具链）。
- 注册表用 `std::call_once` 保证线程安全（`op_registry.cpp`）；内存管理器 mutex + atomic 统计。
- 公共接口 const 约定执行良好；注释全英文。

## 跨柱权衡决策表

| 决策 | 默认采用 | 被否方案及理由 |
| --- | --- | --- |
| 错误模型 | Status/Result，无异常 | C++ 异常：HPC 热路径开销不可控，已否 |
| 后端选择 | 显式指定，错误即返回 | 静默 fallback：破坏可复现性，Readme 明文否决 |
| 算子扩展 | 目录自治 + 静态注册（call_once） | 动态库插件：引入 ABI/供应链负担，无对应需求 |
| YAML 解析 | yaml-cpp 封装层（非手写解析器）+ 输入防御 | 纯手写解析器：健壮性风险；已用 yaml-cpp 并在封装层加边界检查 |
| NPU | 编译期占位返回 Unsupported | 伪实现/stub 返回成功：违反 LSP，已正确规避 |

## 适配度函数（Fitness Functions，已内建且可测试）

| 不变量 | 检查方式 | 本地路径 |
| --- | --- | --- |
| CPU 各级别位精确一致 | 对拍 naive/omp/tbb | `src/ops/resize/tests/test_resize.cpp` 等 |
| 配置加载 fail-fast（先全量校验后构造） | `load_operators_real.cpp` + 失败路径测试 | `tests/operator/test_load_operators.cpp` |
| 依赖方向 backends→core | 目录物理隔离 + 公共头零内部泄漏 | `include/` vs `src/core/backend.h` |
| 缺失内核返回 NotImplemented 不降级 | `FindKernel` 查询 + 错误码 | `op_registry.cpp` |

## 风险登记

| 编号 | 风险 | 状态（2026-11-25） | 说明 |
| --- | --- | --- | --- |
| B1 | 无 CI，测试全靠手动触发 | **已解决** | 新增 `.github/workflows/ci.yml`：Debug/Release 矩阵、ASan+UBSan、`-Werror`、CUDA 构建冒烟四道门禁 |
| R2 | YAML 解析健壮性 | **已解决** | yaml-cpp 封装层增加输入防御（见下节），补 7 个加固测试 |
| R3 | 跨算子脚手架重复（描述符解析、naive/omp/tbb 三连结构雷同） | 接受（观察项） | 当前 5 个算子规模下可接受；若算子数超过 8 个再抽公共描述符解析 helper |
| R4 | CMake sanitizer 开关两分支均为 OFF（死逻辑） | **已解决** | Debug 默认 ON，与注释及 Readme 声明的意图一致 |
| R5 | 无发布工程（CHANGELOG、tag 流程） | 待办 | 建议引入最小发布清单；个人项目优先级可自定 |

## YAML 配置解析加固（2026-11-25 落地）

配置文本按**不可信输入**处理。`src/ops/loader/yaml_parser.cpp` 在进入 yaml-cpp 之前增加预检，
解析阶段增加结构性上限，全部以 `kInvalidArgument` 快速失败：

| 防御项 | 上限/规则 |
| --- | --- |
| 输入总大小 | ≤ 1 MiB |
| YAML 锚点/别名 | 一律拒绝（防止循环别名在解析器内无界递归） |
| flow 集合嵌套深度 | ≤ 64（防止解析器栈耗尽） |
| 算子条目数 | ≤ 1024 |
| 单算子 params 条目数 | ≤ 256 |
| name/op/backend/cpu_level 长度 | ≤ 256 字符 |
| params 键/值长度 | ≤ 1024 字符 |

引号内的 `*`、`[`、`{` 等字符被正确豁免（有对应测试 `QuotedSpecialCharactersStillParse`）。
对应新增测试：超大输入、锚点/别名拒绝、深嵌套拒绝、条目数上限、params 数上限、引号豁免。

## CI 门禁（2026-11-25 落地）

`.github/workflows/ci.yml`，push/PR 到 main 触发，并发组去重：

| Job | 内容 | 性质 |
| --- | --- | --- |
| `build-and-test` | ubuntu-24.04 × Debug/Release 矩阵，系统包依赖，CTest 全量 | 阻断 |
| `sanitizers` | ASan + UBSan（CUDA 关闭以保持 LeakSanitizer 全开） | 阻断 |
| `warnings` | Host 代码 `-Werror` | 阻断 |
| `cuda-build` | CUDA 仅编译冒烟（runner 无 GPU，运行时 CUDA 测试在本地验证） | 阻断（仅编译） |

所有门禁配置已在本地等价配置下验证通过（sanitizer 16/16、`-Werror` 全量编译干净）。

## CUDA 环境性测试用例改造（2026-11-25 落地）

`EnvVarOverridesBackend`、`EnvVarBackendAvailabilityHonored`（原 `EnvVarUnavailableBackendFails`）、
`ConfigBackendAvailabilityHonored`（原 `UnavailableBackendFails`）三个用例原先硬编码假设
`backend: cuda` 不可用，在有 CUDA 设备的机器上断言方向不成立。

现改为基于运行时探针 `CudaBackendAvailable()`（`Context::Create(BackendType::kCUDA)` 是否成功）
分支断言：CUDA 不可用时验证 fail-fast 无回退；可用时验证对象绑定到 CUDA 后端。
同一套断言在 CUDA 与非 CUDA 构建树下均成立，两个分支已分别在本地（CUDA 开）与
临时构建（CUDA 关）下验证通过。

## 结论

作为研究/学习型算子平台，本项目已超过同类项目水准；作为面向未来的生产库，
架构与抽象无需返工。剩余差距集中在发布工程（R5），属低频、可自定节奏的事项。
