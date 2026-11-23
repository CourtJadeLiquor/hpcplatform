# Review — 配置驱动算子对象

审查方式：全新上下文子代理独立只读审查，核对 spec.md / tasks.md 的全部 AC（AC-1 ~ AC-9）与 NFR（NFR-1 ~ NFR-6），并叠加用户编码规则（const 参数、顺序、英文注释、SOLID、不过度设计）。

## 结论

**可交付。** 无阻断 / 高 / 中 级缺陷。所有 AC 与 NFR 均被正确覆盖；默认构建与 YAML 构建均编译链接通过，测试全绿，ABI 纯增量无破坏。

## 回归证据

| 构建 | 结果 |
| --- | --- |
| 默认构建（`hpcplatform_WITH_YAML=OFF`） | 14/14 通过 |
| YAML 构建（`hpcplatform_WITH_YAML=ON`，含 samples） | 15/15 通过；`test_load_operators` 内含 19 个子用例 |
| `samples/05_config_operator`（内联 + 文件） | 运行正确 |

## 发现与处置

| # | 级别 | 内容 | 处置 |
| --- | --- | --- | --- |
| 1 | 低 | AC-4 失败时 map 为空未被断言 | 已修复：新增 `FailureYieldsEmptyMap`，断言失败返回空 map |
| 2 | 低 | 非 CPU + 非 naive → kInvalidArgument 未在 operator 套件集成测试 | 暂不处理：该规则由 `FindKernel` 执行、已被既有 op 测试经 `ValidateInvocation` 同源覆盖；当前树未编译非 CPU 后端，无法构造场景，留待 OpenCV/CUDA 构建树补测 |
| 3 | 低 | per-entry 非法 backend / cpu_level 名字未测 | 已修复：新增 `PerEntryInvalidBackendNameFails`、`PerEntryInvalidCpuLevelFails`，并补 `PerEntryInvalidDeviceIndexFails` |
| 4 | 低 | 具体算子类声明未用 `hpcplatform_WITH_YAML` 门控，默认构建误用会链接错误 | 不处理：这是已批准 spec Task 4 选定的设计（在现有公共头追加声明）；默认构建中 `LoadOperators` 恒返回 `kUnsupported`，正常路径不会拿到对象 |
| 5 | 低 | 测试写死 `/tmp` 路径 | 不处理：ctest 默认串行且用例自行清理，实际安全 |
| 6 | 低 | `-fno-exceptions` 未在构建中实际启用 | 不处理：项目既有不一致（非本次引入），OpenCV 后端同样使用 try/catch；`ParamGet*` 解析器保持无异常 |
| 7 | 低 | AC-7 rubric 阈值（≥4）与显式登记清单模式存在张力 | 不处理：spec 自身 rubric 与 Task 设计的张力，实现忠实遵循已批准的 Task 设计 |
| 8 | 低 | NFR-7 不存在（spec 仅定义 NFR-1 ~ NFR-6） | 不处理：任务描述笔误，NFR-1 ~ NFR-6 均已覆盖 |

## 专项核对（均通过）

- fail-fast 不产生半成品对象：`ValidateAll` 在构造前完成全部校验；`BuildOperators` 任一条失败返回错误，局部 map 随函数销毁。
- 环境优先级 env > shared context > per-entry：命中高优先级即忽略低优先级，无静默回退。
- 错误码三分类：`kInvalidArgument` / `kUnsupported` / `kNotImplemented` 映射正确。
- 链接闭合：默认构建无 yaml-cpp 依赖；YAML 构建 yaml-cpp 以 `PUBLIC` 传播，测试/示例可链接；前置声明与定义链接性匹配。
- ABI：公共头仅增量追加，未变更既有签名与枚举值。

## 待人工验证（spec 已声明不在 CI 内）

AC-3 (b)(c) 的 CUDA 路径：需在 CUDA 构建树中验证 `hpcplatform_BACKEND=cuda` 与共享 CUDA `Context` 正确绑定对象。当前环境无 CUDA 构建，按 spec 留作人工验证。
