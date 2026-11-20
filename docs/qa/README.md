# mycv 问答与易错点（QA）

> 文档视角：**学习者与贡献者**——回答"这个模块为什么这么设计、关键组件是什么、它怎么和全局串起来、我该怎么调用/测试/扩展"。
> 配套阅读：[`docs/architecture/`](../architecture/) 从项目架构角度给出系统全貌。
> 本文档记录的是**容易理解混淆的点**，不是 API 罗列——API 以头文件注释为准。

---

## 索引

| 文档 | 解决的困惑 |
| --- | --- |
| [`cpubackends.md`](./cpubackends.md) | CPU 内存为什么不能只用 `std::vector`/`unique_ptr`？Buffer / MemoryManager / AllocatorState 三层各管什么？ |
| [`tensor_views.md`](./tensor_views.md) | 为什么 View/Slice/Permute 不拷贝？stride 和 offset 怎么算？Reshape 什么时候偷偷深拷贝？ |
| [`memory_lifecycle.md`](./memory_lifecycle.md) | Tensor 为什么能比 Context 活得久？Wrap 的"借用"和"收养"有什么区别？ |
| [`op_dispatch.md`](./op_dispatch.md) | 算子到底是怎么被选中的？`kUnsupported` / `kNotImplemented` / `kInvalidArgument` 到底什么时候返回哪个？ |
| [`context_copy.md`](./context_copy.md) | `Context::Copy` 拷贝的是字节还是元素？跨后端怎么选路？什么时候同步？ |
| [`cuda_stream_visibility.md`](./cuda_stream_visibility.md) | 为什么 Wrap 进来的 CUDA 内存数据不对？non-blocking 流和外部内存的可见性责任在谁？ |
| [`error_model.md`](./error_model.md) | 为什么不用异常？`Status` / `Result` 怎么用？六种错误码怎么选？ |
| [`how_to_extend.md`](./how_to_extend.md) | 新增一个算子 / 一个后端 / 一个 CPU 优化版本，分别要动哪些文件、守哪些规矩？ |
| [`testing.md`](./testing.md) | 三棵构建树怎么跑？怎么写对拍测试？容差怎么定？ |

---

## 这份文档的写法约定

- **先讲"为什么"，再讲"是什么"**：每个易错点先解释设计动机，因为混淆往往来自不知道取舍。
- **每条结论锚定到代码**：给出关键文件路径，方便读者核实（路径以仓库根为基准）。
- **不替代码说谎**：如果某处是"当前行为、未来可能变"，会明确标注，不包装成永恒真理。
- **PDF 友好**：标准 Markdown，可直接导出。
