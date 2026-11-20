cpu后端设计

Buffer 负责"所有权"，CpuMemoryManager 负责"分配策略"，CpuAllocatorState 负责"两者的生命周期解耦"。

| 层                 | 职责                   | 关键代码               |
| ----------------- | -------------------- | ------------------ |
| Buffer            | 谁在什么时候释放             | \~Buffer 调 deleter |
| CpuMemoryManager  | 怎么分配：对齐、统计、复用        | Allocate           |
| CpuAllocatorState | 管理器先死、Tensor 还活着时谁兜底 | shared\_ptr 捕获     |

<br />

## 为什么不能只用 RAII（比如 std::unique\_ptr\<float\[]>）

1. 对齐。 算子内核按 64 字节对齐做向量化假设，默认`alignment=64` 。`new` /`malloc` /`std::vector` 只保证`alignof(max_align_t)` ，通常 16 字节。所以必须走`posix_memalign` ，这已经超出标准容器的能力边界。
2. 无异常错误模型。`new` 失败抛`std::bad_alloc` ，而本库明确要保持`-fno-exceptions` 可用（ error.h 的注释写了这个动机）。`Allocate` 返回`Result<shared_ptr<Buffer>>` ，失败是`Status::OutOfMemory` 。`std::unique_ptr` +`new` 在这个约束下直接不可用。
3. 统计。`MemoryStats` 要给出`allocated_bytes / peak_bytes / allocation_count` ，其中峰值必须是 跨所有分配聚合 的。单个 RAII 对象只能知道自己那一份，没有共享计数器就无法算峰值。峰值用 CAS 循环维护，这天然要求一个共享状态。
4. 复用（这是引入空闲块池的直接动机）。 算子每次调用都新分配输出，一个 5 级链路的基准跑 20 轮就是上百次 alloc/free。空闲块按 精确字节数 留在池里，下一轮直接命中，上一轮实测`cache_reuse=yes` 就是靠这个。纯 RAII 的`unique_ptr` 语义是"析构即归还"，结构上做不到复用。
5. 后端多态。`Tensor` 只持有`shared_ptr<Buffer>` ，完全不知道内存来自`posix_memalign` 、`cudaMalloc` 还是 NPU 的 DMA 池。这正是 deleter 用类型擦除的原因：把"怎么释放"藏进闭包，`Buffer` 本身保持非虚、轻量。如果直接 RAII 成`unique_ptr<T[]>` ，分配方式就被写死在类型里，跨后端就分裂了。
6. 生命周期倒置。 这是最容易被忽略、但最容易写出悬垂指针的地方：

Result<Tensor> MakeFoo(); // 内部用某个 Context 分配
// Context 先析构，Tensor 后析构

如果`Buffer` 的 deleter 直接持有`CpuMemoryManager*` ，Context 一销毁就是 use-after-free。所以 deleter 捕获的是`shared_ptr<CpuAllocatorState>` ： 管理器只是 state 的一个持有者，不是 state 本身 。Manager 死了，state 靠还活着的 Buffer 续命，等最后一个持有者释放时才在 \~CpuAllocatorState 里把整池还给 OS。

一句话： RAII 保证"不泄漏"，共享 state 保证"不悬垂" ，两者解决的是不同问题。

## 具体机制值得注意的细节

- 复用时的对齐校验 ：命中空闲块要求`bytes` 完全相同 且 该地址满足本次`alignment` 。前者防越界，后者防"32 字节对齐的旧块被当 64 字节用"。
- `Release` 先入池再减计数 ：`allocated_bytes` 表示 活跃 字节，池里的不算，所以基准测试中峰值是在正确性校验前采样的。
- 加锁范围最小化 ：只在`free_blocks` 的查找/插入上持锁，`AlignedAlloc` 在锁外，避免慢速系统调用串行化所有线程。

## 这个设计的代价（已知局限，评审里也提过）

1. 池无上限、无淘汰 ：不同尺寸长期运行时历史块会一直驻留到 Context 销毁；`Stats()` 不包含缓存字节数，监控上看起来已释放。固定尺寸链路没问题，多尺寸长跑就是隐患。
2. 只做精确匹配 ，不做尺寸取整或合并，所以收益集中在"同一链路重复跑"，碎片化场景收益有限。
3. 分配路径有一把 mutex ，高并发小分配时会成为热点（原子计数本身无锁，但链表操作有锁）。
4. 视图共享整个 Buffer ：`Slice` /`Permute` 的视图会让整块底层内存存活，无法只释放片段——这是"轻量视图"换取"简单所有权"的取舍。

## 什么情况下"直接 RAII 就够"

如果这些 buffer 只在一个函数内部短暂存在、不需要对齐保证、不需要统计、不需要跨后端、也不需要复用——那确实用`std::vector<std::byte>` 就够了，管理器是纯负担。`CpuMemoryManager` 存在的理由是它服务于 算子返回值 ：这些 Tensor 会跨函数、跨 Context 生命周期、跨后端流动，还要被基准统计。所以这里多一层不是过度设计，而是把"分配策略"从"所有权"里剥出来的必然结果。
