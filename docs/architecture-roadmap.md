# mini_ml 框架重构与并行计算路线图

> 状态：讨论稿，不是已经批准的实现规格  
> 更新日期：2026-09-17  
> 当前动作：只记录思路，不移动或修改任何现有代码  
> 使用方式：后续每次只选择一个“待决定”问题讨论，确认后再更新决策记录

## 0. 文档标记

本文使用以下标记区分结论的确定程度：

- **已形成共识**：来自当前项目目标和已经明确表达的取舍，可以作为后续讨论的默认前提。
- **建议**：目前看来较合理，但仍然可以推翻。
- **待决定**：会明显影响后续设计，需要专门讨论后才能实施。
- **延后**：当前没有足够的真实需求，不提前设计。

这份文档不会把“像 PyTorch”理解成兼容 PyTorch 的 API、功能或性能。当前只借鉴其分层思想：Tensor、算子、kernel、执行层之间应有清楚的依赖方向。

---

## 1. 项目真正要解决的问题

### 1.1 主要目标

**已形成共识：** `mini_ml` 的重点不是把 Lab1 包装得越来越通用，而是学习并实现一条最终能够支撑 ViT 实验的 C++ 机器学习计算链，尤其是其中的 CPU 多线程并行部分。

目标依赖链可以概括为：

```text
ViT / 神经网络模块
        ↓
Tensor 与自动微分（若最终实验需要训练）
        ↓
公开 Tensor 算子
        ↓
CPU kernel
        ↓
parallel_for / parallel_reduce / 固定线程池
```

最终希望证明的不只是“能写出一个 ViT 类”，而是：

1. ViT 的主要计算确实经过自己的 Tensor 和算子接口。
2. Matmul、Softmax、LayerNorm、Attention 等热点确实使用自己的并行执行器。
3. 串行参考实现与并行实现的数值结果在合理误差范围内一致。
4. 能通过 benchmark 解释什么时候获得加速、什么时候被调度开销或内存带宽限制。
5. 框架层不依赖任何具体实验、模型、CSV 格式或命令行参数。

### 1.2 目前不追求的目标

以下内容当前不是必要目标：

- 完整复刻 PyTorch。
- 与 PyTorch API 或模型文件兼容。
- CUDA、ROCm、分布式训练或多机通信。
- 混合精度、量化、JIT、图编译和自动算子融合。
- 支持几十种 dtype 和任意设备。
- 通用 Trainer、实验管理、超参数搜索平台。
- 通用 CLI、CSV、画图和报告系统。
- 让每个实验的 `main.cpp` 都只有几行。
- 在所有输入规模、所有算子上获得线性加速。
- 并行浮点结果与串行结果逐 bit 相同。
- 为未来“可能出现”的功能预先建立空接口和空目录。

这些边界不是永久禁止，而是防止当前阶段把精力分散到与 ViT 并行主线无关的工作上。

---

## 2. 当前仓库的事实与诊断

### 2.1 当前代码是一条 Lab1 纵向原型

目前的代码已经完整跑通了：

```text
生成一维样本
  → 多项式模型
  → 标量 MSE
  → 手写 loss/gradient
  → SGD 或共轭梯度
  → CSV 与 Python 绘图
```

这条链适合作为 Lab1 的实验实现和学习记录，但不应自然地被认定为未来框架核心。

| 当前类型或文件 | 实际职责 | 当前判断 |
|---|---|---|
| `Sample1D` | 一维回归样本 | Lab1 专属 |
| `Dataset1D` | 对 `vector<Sample1D>` 的薄包装 | Lab1 专属，尚未形成通用 Dataset 协议 |
| `sample_curve*` | 一维函数曲线采样 | Lab1 专属 |
| `PolynomialRegression` | 多项式回归模型 | Lab1 专属 |
| 当前 `MSELoss` | 对两个 `double` 计算值和导数 | 不能直接服务 Tensor/autograd |
| `TrainResult` | 一次 loss/gradient 计算结果 | 名称过于通用，语义实际属于旧管线 |
| 当前 `SGD` | 更新一个 `vector<double>` | 无法直接管理未来 Tensor 参数 |
| `ConjugateGradient` | 二次目标优化算法 | 对 Lab1 有价值，但不是 ViT 主线 |
| `SerialBackend` | 多项式模型的全批量 loss/gradient | 不是通用 Backend |
| `RandomGenerator` | 串行随机数生成 | 有潜在复用价值，但并行 RNG 需要重新设计 |

### 2.2 主要问题不是文件数量，而是抽象边界

当前 `sample.hpp` 和 `result.hpp` 都只有极少量内容，`Dataset1D` 主要转发 `std::vector` 的接口；同时 `backend.hpp`、`cpu.hpp`、`cpu.cpp` 仍为空。

小文件本身并不是错误。真正的问题是：这些文件还没有形成能够独立使用、独立演化或隔离复杂度的稳定边界。

尤其是当前依赖方向：

```text
runtime/serial
    ├── Dataset1D
    ├── PolynomialRegression
    ├── MSELoss
    └── TrainResult
```

底层 `runtime` 反向依赖具体实验模型和损失，说明 `SerialBackend` 实际是 Lab1 的融合计算函数，而不是可复用的执行后端。

目标方向应当变为：

```text
实验 / 模型
    ↓ 调用
Tensor 公开算子
    ↓ 调用
CPU kernel
    ↓ 调用
并行执行原语
```

并行执行层不知道 Tensor，CPU kernel 不知道 Module，Tensor 算子不知道 ViT，框架也不知道某个实验的 CSV 格式。

### 2.3 对现有代码的暂定处理态度

**建议：** 先冻结 Lab1，保证它继续可以构建和运行；不要为了目录整齐立刻搬文件。新框架不再建立在旧 `SerialBackend` 上，等新核心稳定后再决定是否迁移或保留 Lab1 原样。

这样可以避免把“机械移动旧代码”误当成真正的架构进展。

---

## 3. 抽象与文件拆分原则

### 3.1 只有真实边界才进入公共框架

一个概念进入 `include/ml`，建议至少满足以下一项：

1. 已经出现两个真实使用者。
2. 它是多个上层共同依赖的稳定基础能力。
3. 它隔离了内存、并发、设备或数值实现方面的复杂性。
4. 不抽象会产生已经可以观察到的重复或依赖倒置。

如果只是某个实验使用，优先留在 `lab/labN`。如果只是未来可能有用，先不创建。

### 3.2 不采用“一种类型一个头文件”

彼此总是一起出现、一起变化、依赖相同的小类型应当放在一起。例如第一版 `tensor.hpp` 可以共同声明：

- Tensor
- Shape
- Strides
- TensorOptions
- 第一版需要的数据类型信息

但也不应把随机数、线程池、Tensor、日志等所有“基础内容”都塞进一个 `core.hpp`，因为它们没有相同的依赖方向和变化周期。

### 3.3 公共接口可以粗，内部实现可以细

用户看到的公共入口应当少而稳定；内部 kernel 可以按功能和优化方式细分。

```text
公开：少量概念级头文件
内部：多个 .cpp / private header / kernel 文件
```

内部拆分不会自动增加用户侧 API 的复杂度，反而有利于分别测试和优化。

### 3.4 头文件与 `.cpp` 的边界

适合放头文件：

- 小型值类型和声明。
- 简单内联函数。
- 必须在实例化位置可见的模板。
- 统一入口的聚合头文件。

适合放 `.cpp`：

- Tensor 内存管理的复杂实现。
- 线程池、任务队列与同步。
- CPU kernel。
- 非模板算子实现。
- autograd 图遍历。
- Module、Optimizer 的非模板实现。
- 文件系统、日志、序列化与实验输出。

本项目没有必要为了“调用代码短”而变成 header-only。

### 3.5 何时才拆一个头文件

满足以下条件之一时再考虑拆分：

- 某个概念可以被独立使用。
- 它引入了明显不同或很重的依赖。
- 它与同文件其他内容有不同的变化周期。
- 当前组织迫使大量调用者包含不需要的依赖。
- 文件已经承担多个不相关职责。

不因为以下原因拆分：

- 新增了一个小结构体。
- 文件超过某个随意的行数。
- 大型框架中存在同名目录。
- 想让项目“看起来完整”。
- 未来也许会需要。

---

## 4. 对参考项目的取舍

| 项目 | 借鉴什么 | 不照搬什么 |
|---|---|---|
| PyTorch / ATen | Tensor、算子、kernel、执行层的依赖方向；公共聚合入口；按算子做 CPU/CUDA 分派 | c10 全套 dispatcher、代码生成、几十种 dtype/device、完整 hook/serialization/JIT 体系 |
| Eigen | 按模块提供入口，内部模板实现可以细分；重视 layout、stride 和表达式成本 | 全面 header-only、重模板元编程和复杂表达式模板 |
| oneDNN | 把计算 primitive、设备 engine、执行 stream、memory 分开思考 | 直接复制它面向生产级多设备库的 API |
| ggml | 核心 Tensor/graph API 与 CPU/backend 实现分离；线程池属于 CPU 执行能力 | 直接照搬其 C API、图执行器和多后端注册系统 |
| Flashlight | 框架核心与具体研究 app 分离 | 在当前规模下提前建立大量 package/app 基础设施 |

PyTorch 用户代码短，是因为大量复杂度已经在编译后的库中实现，而不是因为把所有实现放进了头文件。

参考资料见文档末尾。

---

## 5. 候选总体分层

### 5.1 第一阶段候选结构

只有真正开始实现时才创建对应文件；下面只是候选布局，不要求现在一次性建好。

```text
include/ml/
  ml.hpp                 # 可选的用户聚合入口
  tensor.hpp             # Tensor 及紧密相关元数据
  ops.hpp                # 公开 Tensor 算子
  execution.hpp          # CPU 并行执行原语

src/
  tensor.cpp
  ops.cpp                # 初期可以存在，增长后再拆
  execution/
    thread_pool.cpp
  kernels/cpu/
    elementwise.cpp
    reduction.cpp
    matmul.cpp
    normalization.cpp
```

### 5.2 确认需要训练以后再增加

```text
include/ml/
  autograd.hpp
  nn.hpp
  optim.hpp

src/
  autograd.cpp
  nn.cpp
  optim.cpp
```

如果某个公共头文件后来确实过大，可以再增加 `ml/nn/...`、`ml/optim/...` 等子目录，同时保留 `nn.hpp`、`optim.hpp` 作为方便入口。

### 5.3 候选依赖规则

```text
execution
    ↑
CPU kernels
    ↑
Tensor ops
    ↑
autograd
    ↑
nn / optim
    ↑
labs
```

约束：

- `execution` 不依赖 Tensor、autograd、nn 或实验。
- CPU kernel 可以依赖 execution，但不能依赖某个模型。
- Tensor/ops 不依赖 nn、optimizer 或数据集。
- autograd 依赖 Tensor/ops。
- nn 和 optim 依赖 Tensor/autograd/ops。
- 实验可以依赖所有公开框架层。
- `include/ml` 永远不依赖 `lab/`。

---

## 6. 从 ViT 反推最小能力

### 6.1 第一版 ViT 的候选范围

以下只是为了约束设计的**建议范围**，必须在实现前确认：

- CPU-only。
- `float32` 训练。
- 固定图像尺寸和 patch 大小。
- 非重叠 patch。
- Encoder-only 分类 ViT。
- 支持 batch。
- AdamW。
- 主要采用算子内部并行。
- backward 图节点顺序调度，具体 backward kernel 内部并行。

第一版可能不需要：

- 通用 Conv2d：patch embedding 可先由 patchify + Linear 完成。
- causal mask、cross-attention、KV cache。
- 混合精度和梯度缩放。
- 并行 autograd 图调度。
- 通用 Trainer 或 DataLoader。
- Dropout 和 stochastic depth；是否需要由课程要求决定。

### 6.2 候选数据流

```text
Image [B, C, H, W]
  │
  ├─ patchify
  ├─ Linear patch embedding
  ├─ prepend CLS token
  ├─ add positional embedding
  │
  ├─ N × Transformer Encoder Block
  │    ├─ LayerNorm
  │    ├─ Multi-Head Self-Attention
  │    ├─ residual add
  │    ├─ LayerNorm
  │    ├─ Linear → GELU → Linear
  │    └─ residual add
  │
  ├─ select CLS token
  ├─ Linear classifier
  └─ CrossEntropyLoss
```

### 6.3 ViT 部件到框架能力的映射

| ViT 部件 | 最小 Tensor/算子能力 | backward 需求 | 并行优先级 |
|---|---|---|---|
| Patchify | reshape、permute、contiguous | 逆向 reshape/permute | 中 |
| Patch embedding | matmul、bias add | dX、dW、db | 高 |
| CLS token | expand/broadcast、concat | batch 方向归约、split | 低 |
| Position embedding | broadcast add | `sum_to_shape` | 低 |
| QKV projection | 大矩阵乘法 | matmul backward | 很高 |
| Attention score | batched matmul、transpose、scale | 两次 batched matmul | 很高 |
| Softmax | row max、exp、sum、divide | Softmax 专用 backward | 高 |
| Attention × V | batched matmul | 两次 batched matmul | 很高 |
| Output projection | matmul、bias add | matmul backward、bias reduce | 很高 |
| LayerNorm | mean、variance、rsqrt、逐元素运算 | LayerNorm backward | 高 |
| MLP | matmul、GELU、matmul | 对应 backward | 很高 |
| Residual | add | 分支梯度累积 | 中 |
| 分类头 | select、matmul | select backward、matmul | 中 |
| CrossEntropy | logsumexp、索引、mean | logits gradient | 中 |
| AdamW | 参数逐元素更新 | 不进入 autograd 图 | 中 |
| Accuracy | argmax、比较、reduce | 不需要 | 低 |

这张表意味着：真正应优先优化的是 Matmul/Batched Matmul、Softmax 和 LayerNorm，而不是 epoch 循环本身。

---

## 7. 各层候选设计

### 7.1 Tensor

Tensor 是代价最高、最不适合匆忙决定的部分。

候选元数据：

```text
Storage ownership
shape
strides
storage offset
numel
contiguous state
```

训练路线确认后还可能需要：

```text
requires_grad
grad
grad_fn / autograd metadata
```

#### 第一版候选约束

- 仅 CPU 内存。
- 动态 rank。
- 默认 row-major contiguous 布局。
- shape 和 stride 使用有符号或无符号整数需要专门决定。
- 支持 RAII、移动和共享 Storage。
- `reshape` 只有布局兼容时才产生 view。
- `transpose/permute` 可以产生 strided view。
- 暂不支持负 stride。
- 不支持非连续输入的 kernel 可以显式要求 `contiguous()`。
- 对参与 autograd 的 Tensor，第一版严格限制 in-place。

#### dtype 的候选路线

| 路线 | 优点 | 风险 |
|---|---|---|
| 固定 `float` Tensor | 最简单，适合先验证并行 | 以后加入标签和 double 参考时需要扩展 |
| 非模板 Tensor + 运行时 DType | 更接近主流框架，public API 稳定 | kernel dispatch、访问和测试复杂度较高 |
| `Tensor<T>` 模板 | 类型安全，编译期分派 | Module/autograd/容器和编译时间会更复杂 |

**当前建议但未决定：** public Tensor 不做 `Tensor<T>`；第一阶段只真正实现 Float32。是否从一开始存储 `DType` 标签，需要单独讨论。

#### Tensor 阶段必须明确的语义

- 空 Tensor、标量 Tensor、零长度维度分别是什么。
- copy 是共享 Storage 还是深拷贝。
- `clone()` 与普通复制的区别。
- view 的生命周期。
- 非连续 Tensor 的 `data()` 是否允许直接访问。
- shape 乘积溢出如何检查。
- 越界和非法 shape 使用异常、断言还是错误返回。
- 多个 view 是否允许并发写入同一 Storage。

这些决定一旦进入公共 API，后续修改成本很高。

### 7.2 Tensor 算子

第一批串行参考算子建议从 ViT 的真实需求中选择：

#### 基础逐元素

- add / subtract
- multiply / divide
- scalar 运算
- exp / log
- sqrt 或 rsqrt
- GELU 所需的 tanh 或 erf

#### Reduction

- sum
- mean
- max
- 指定维度
- keepdim
- `sum_to_shape`，服务广播 backward

#### 形状操作

- reshape / flatten
- transpose / permute
- contiguous
- select 或 slice
- concat
- expand / broadcast

#### 线性代数

- 2D matmul
- 3D batched matmul
- 是否支持转置标志，需结合 kernel 设计决定

#### 语义算子

- Softmax
- GELU
- LayerNorm
- CrossEntropy

Softmax、LayerNorm 和 CrossEntropy 应拥有明确的语义接口及稳定数值实现，而不是永久由几十个细碎算子拼接，因为专用 forward/backward 更容易保证数值稳定并优化。

第一版不需要一开始实现任意 rank、任意广播的完整通用 matmul。Attention 可以先把四维数据整理成 `[B * heads, tokens, head_dim]`，使用 3D batched matmul。

### 7.3 CPU 并行执行层

执行层只负责“如何执行一组工作”，不认识 Tensor、模型、Loss 或 Dataset。

第一版候选能力：

- 固定大小、可复用线程池。
- 线程数设置与查询。
- `parallel_for(begin, end, grain, fn)`。
- `parallel_reduce(...)`。
- 等待当前工作完成。
- worker 异常通过 `exception_ptr` 回传并在调用线程重抛。
- 安全关闭和析构。
- 单线程与小任务直接执行路径。
- 嵌套并行保护。

第一版不需要：

- 通用 Future/Promise 框架。
- 任意 DAG 调度。
- 多 Executor 虚基类体系。
- GPU stream。
- 每个算子创建自己的线程。

#### 需要重点讨论的策略

1. 全局默认线程池还是显式 `ExecutionContext`。
2. 调用线程是否参与计算。
3. 静态切块、原子索引分配还是 work stealing。
4. grain size 如何决定。
5. worker 内再次进入 `parallel_for` 时如何处理。
6. 一个任务失败后，其余任务是否继续完成。
7. 是否提供确定性 reduction 模式。

**初始建议：** 先用简单、可解释的固定线程池和静态/原子切块；worker 内嵌套调用退化为串行。不要在没有 benchmark 证据前实现复杂 work stealing。

### 7.4 CPU kernel

公开 op 与并行策略之间建议保持以下结构：

```text
公开 Tensor op
      ↓
CPU kernel
  ├── 小规模：串行路径
  └── 大规模：parallel_for / parallel_reduce
```

建议优先级：

1. Matmul。
2. Batched Matmul。
3. Softmax。
4. LayerNorm。
5. Reduction。
6. Elementwise。
7. AdamW 参数更新。

Elementwise 容易并行，但通常受内存带宽限制；它适合验证线程池，不应被当作最终性能代表。

#### Matmul 演进顺序

1. 最朴素的串行三重循环，作为 oracle。
2. 调整循环顺序，改善连续访问。
3. cache blocking。
4. 按输出 tile 并行，保证每个任务独占写入区域。
5. 有数据后再研究 packing、SIMD。

不要在正确性语义尚未稳定时直接实现复杂 packing kernel。

#### 自然并行粒度

- Elementwise：连续区间。
- Reduction：每线程局部结果，再按确定顺序合并。
- Matmul：输出矩阵 tile。
- Batched Matmul：batch/head 与输出 tile。
- Softmax：row。
- LayerNorm：token row。
- AdamW：参数连续区间。

### 7.5 Autograd

**待决定前提：** 最终 ViT 实验究竟要求训练还是只要求推理。如果只做推理，通用 autograd 可以延后；如果需要训练，autograd 会成为主线能力。

若实现训练，建议从动态反向模式开始：

```text
Tensor
  └─ AutogradMeta
       ├─ requires_grad
       ├─ accumulated grad
       └─ grad_fn

GradFn
  ├─ parent edges
  ├─ forward 保存的数据
  └─ backward(upstream_grad)
```

最小能力：

- 标量 loss 的 `backward()`。
- 非标量输出可以传入初始梯度。
- 逆拓扑遍历。
- 叶子参数梯度累积。
- fan-out/residual 图中的正确累加。
- broadcasting backward。
- `zero_grad()`。
- `no_grad` 和 `detach`。
- backward 后计算图释放策略。
- 避免 Tensor 与 GradFn 之间形成所有权环。

第一版 backward 图可以单线程按拓扑顺序调度；每个 backward kernel 内部仍然可以使用并行 kernel。这样不必过早构建并行图调度器。

#### 第一版 in-place 建议

- 禁止对正在参与计算图的 Tensor 做危险的 in-place 修改。
- 优化器更新在 `no_grad` 环境执行。
- 暂不复刻完整 version counter/view mutation 系统。
- 真正需要 in-place 性能后再引入版本检查。

### 7.6 Module 与 Parameter

ViT 组合时至少需要：

- Parameter 身份。
- 参数枚举。
- 子模块参数递归收集。
- `zero_grad()`。
- 参数初始化。
- 如果存在 Dropout，则需要 `train()/eval()`。

C++ 没有 Python 的动态属性系统，不必强行复制 PyTorch 的注册魔法。第一版可以显式注册参数/子模块，或者让模块明确返回自己的参数列表。

需要避免：

- 参数遗漏或重复。
- 临时 Tensor 被误当成 Parameter。
- optimizer state 因参数对象移动而失效。
- Parameter identity 依赖不稳定的裸指针。

### 7.7 Optimizer

未来 optimizer 应管理 Tensor Parameter，而不是继续扩展当前 `vector<double>` 版本。

ViT 主线优先需要 AdamW：

- 一阶动量。
- 二阶动量。
- bias correction。
- epsilon。
- decoupled weight decay。
- optimizer state 不进入 autograd 图。
- `no_grad` 下更新参数。

Tensor 版 SGD 可以作为较简单的正确性起点，但不必为了兼容 Lab1 而保留相同内部表示。

### 7.8 Data、CLI、日志和序列化

- 图片读取、增强和特定数据集逻辑优先留在 `lab/lab6`。
- 框架层只需要接收成批 Tensor。
- 只有第二个实验确实需要相同 Dataset/DataLoader 协议时，再晋升到公共 API。
- DataLoader 并行与计算线程池可能发生 oversubscription，真正引入时需统一线程所有权。
- CLI、CSV、绘图始终可以留在实验。
- 模型 checkpoint 对长时间训练有价值，但文件格式是高代价承诺，应在 Parameter/Optimizer 状态稳定后再设计。

---

## 8. 分阶段路线与验收门槛

每一阶段完成后都允许停下来修改设计。阶段目标以“可以验证的成果”描述，而不是以“创建了多少类和文件”描述。

### P0：确认目标与冻结基线

目标：先确定真正需求，保住已有 Lab1 成果。

工作：

- 确认最终 ViT 是训练还是推理。
- 确认 CPU-only 是否为课程范围。
- 确认是否允许 BLAS、OpenMP、TBB；标准库是否为硬约束。
- 确认大致数据集、输入尺寸和模型规模。
- 记录当前测试和 Lab1 的关键输出。
- 把旧管线视作 legacy vertical slice，新框架不再依赖它。

验收：

- 有明确目标范围。
- 现有 CTest 全部通过。
- Lab1 SGD/CG 仍能运行。
- 已记录 legacy 与新框架边界。

### P1：最小 Tensor + 串行参考算子

目标：建立后续所有计算共享的数据表示和正确性 oracle。

工作：

- 确认 Storage、shape、stride、offset、copy/view 语义。
- 实现 CPU Tensor 生命周期。
- 实现最少的 shape/view 操作。
- 实现串行 elementwise、reduction、2D matmul。
- 每个 op 写清 shape 规则和错误条件。

验收：

- 构造、复制、移动、clone、reshape、transpose 行为有测试。
- shape 乘积溢出、非法维度和空 Tensor 有测试。
- 手算小矩阵结果正确。
- 串行实现足够简单，可以作为并行 oracle。
- ASan/UBSan 下无明显内存和越界问题。
- Tensor 不依赖实验、nn、optimizer 或 execution。

### P2：并行执行器

目标：获得与机器学习语义无关、可重复使用的 CPU 并行基础。

工作：

- 固定线程池。
- parallel_for。
- parallel_reduce。
- 小任务串行阈值。
- 异常传播。
- 嵌套并行规则。
- 生命周期和安全关闭。

验收：

- 每个索引恰好执行一次。
- 空区间、小区间、非整除区间正确。
- 线程数为 1 时行为等价于串行。
- worker 异常能回到调用线程。
- 高频重复调用和析构不死锁。
- 不在每个算子调用时创建线程。
- 条件允许时通过 ThreadSanitizer 压力测试。

### P3：并行 CPU kernel

目标：让执行器服务真实的 Tensor 计算。

工作：

- 并行 elementwise，验证基本切分。
- 并行 reduction，验证局部结果合并。
- 并行 matmul，按输出 tile 分工。
- 再实现 batched matmul、Softmax、LayerNorm。
- 为小输入保留串行路径。

验收：

- 1 线程与串行 oracle 一致。
- N 线程结果在统一误差标准内一致。
- 非整除尺寸和极小输入正确。
- 输出写入区间不存在重叠竞争。
- Matmul、reduction、Softmax、LayerNorm 有独立 benchmark。
- 对没有加速的场景能解释原因。

### P4：Autograd（仅在训练需求确认后）

目标：模型由 Tensor 运算自然组成，不再为每个模型手写整条梯度。

工作：

- 动态反向图。
- 梯度累积和拓扑遍历。
- 基础算子 backward。
- broadcasting backward。
- matmul/batched matmul backward。
- Softmax、GELU、LayerNorm、CrossEntropy backward。
- no_grad、detach、zero_grad。

验收：

- 每个可微算子通过中心有限差分 gradient check。
- 广播 bias 梯度正确。
- residual/fan-out 图梯度正确累加。
- 同一参数重复使用时不丢梯度。
- backward 后没有明显引用环泄漏。
- 串行和并行 kernel 得到的梯度在容差内一致。

### P5：NN 与 Tensor optimizer

目标：提供组合 ViT 所需的最低层用户接口。

工作：

- Parameter 与 Module 参数枚举。
- Linear。
- LayerNorm。
- GELU。
- Tensor 版 SGD（可选的简单起点）。
- AdamW。
- 必要的参数初始化。

验收：

- Linear/LayerNorm/GELU forward 与 backward 正确。
- AdamW 单步更新与手算公式一致。
- 参数枚举不遗漏、不重复。
- 一个很小的 MLP 可以过拟合极小数据集。
- optimizer 不依赖任何具体模型类型。

### P6：Attention 与极小 ViT

目标：用前面各层组合出第一个真正的目标模型。

工作：

- patchify + patch embedding。
- CLS token 和 position embedding。
- Multi-Head Self-Attention。
- Transformer Encoder Block。
- 分类头和 CrossEntropy。
- 极小 batch 的完整 forward/backward/step。

验收：

- 所有中间 Tensor shape 有测试。
- Attention 前向可与独立朴素参考比较。
- Attention backward 有有限差分抽查。
- 单个 Encoder Block 能完整训练。
- 极小 ViT 能过拟合几张图片。
- profiler/计数证明热点实际调用并行 kernel。

### P7：端到端 ViT 与并行评估

目标：证明并行框架在最终实验中正确且有实际效果。

验收：

- 真实数据完成 forward、backward、AdamW step。
- loss 明显下降。
- 串行与并行训练趋势接近。
- 报告不同线程数、batch、token、embedding 维度下的结果。
- 分别报告 kernel 和完整 training step 的耗时。
- 能解释并行阈值、扩展上限和瓶颈。

### P8：最后处理 Legacy

新核心稳定后再决定：

- Lab1 是否值得迁移到 Tensor API。
- 旧 `SGD`/`CG` 是否移入 `lab/lab1`。
- `Sample1D`、`Dataset1D`、sampling、regression 是否聚合为一个 Lab1 私有文件。
- 旧 `SerialBackend` 是改名、内联为实验函数还是删除。
- 空 placeholder 文件是否删除。

在 P8 之前不要求为了“最终目录”搬动旧代码。

---

## 9. 测试策略

### 9.1 Tensor 测试

- 空 Tensor、标量 Tensor、零长度维度。
- shape 乘积溢出。
- copy/move/clone。
- view 的 Storage 生命周期。
- contiguous 判定。
- reshape 合法与非法情况。
- transpose/permute 后索引映射。
- `contiguous()` 的数据顺序。
- 广播兼容与不兼容。

### 9.2 算子测试

每个关键算子尽量比较三份结果：

```text
手算小样例
串行高精度或朴素 reference
被测试的实际 kernel
```

重点覆盖：

- 非方阵 matmul。
- 尺寸不是线程数或 tile 大小整数倍。
- batch 为 1。
- 很小 token 数。
- 不整齐的 embedding 维度。
- 极大/极小 logits。
- LayerNorm 接近零方差输入。

### 9.3 Autograd 测试

对随机小 Tensor 使用中心有限差分：

```text
df/dx ≈ [f(x + ε) - f(x - ε)] / (2ε)
```

重点测试：

- 广播 bias。
- matmul 两侧输入。
- batched matmul。
- Softmax。
- GELU。
- LayerNorm 的输入、weight、bias。
- CrossEntropy。
- residual 分支。
- reshape/transpose 后继续计算。

### 9.4 并发测试

- 1、2、多个线程。
- 大量短任务与少量长任务。
- 空区间和不均匀区间。
- worker 内再次调用并行原语。
- 任务抛异常。
- 线程池反复创建/销毁。
- 连续执行数千次，观察偶发死锁。
- TSan/ASan/UBSan 作为辅助验证。

### 9.5 集成测试

依次增加：

1. Linear 单步训练。
2. 两层 MLP 过拟合少量数据。
3. 单个 Attention Block forward/backward。
4. 单层极小 ViT 过拟合几张图片。
5. 完整实验配置训练若干 epoch。
6. 固定种子下比较串行与并行的 loss 和参数趋势。

“能过拟合极小数据集”是重要集成门槛。若这一步失败，通常说明 autograd、shape、参数注册或 optimizer 存在问题。

---

## 10. Benchmark 设计

### 10.1 Benchmark 必须来自目标形状

设：

```text
B  = batch size
N  = token count
D  = embedding dimension
H  = head count
Dh = D / H
M  = MLP hidden dimension
```

至少测量：

```text
QKV:        [B*N, D] × [D, 3D]
Projection: [B*N, D] × [D, D]
MLP 1:      [B*N, D] × [D, M]
MLP 2:      [B*N, M] × [M, D]
QKᵀ:        B*H 个 [N, Dh] × [Dh, N]
AttentionV: B*H 个 [N, N] × [N, Dh]
Softmax:    [B, H, N, N]
LayerNorm:  [B, N, D]
```

不要只测与目标模型无关的随意方阵。

### 10.2 记录内容

- 单线程耗时。
- 多线程耗时。
- speedup：`T1 / Tp`。
- parallel efficiency：`speedup / thread_count`。
- Matmul GFLOP/s。
- 完整 forward/backward/training step 耗时。
- 主要 Tensor 的内存规模。
- CPU 型号、编译器、构建类型、线程数、输入 shape。

### 10.3 测量约束

- 使用 Release 构建。
- 先 warm-up。
- 多次测量并报告 median，必要时报告波动范围。
- CSV、日志和数据加载不进入 kernel 计时。
- 正确性检查放在计时区外。
- 结果必须先通过正确性验证，再讨论速度。
- 性能数据不作为跨机器的固定单元测试断言。

第一阶段不设不现实的绝对速度目标。可接受的原则性要求是：

- 大规模 Matmul 的多线程路径应当比 1 线程更快。
- 小 Tensor 通过阈值避免严重的并行退化。
- 随线程数增加的变化应当能够解释。
- 未获得加速也必须保留真实数据，而不是隐藏结果。

---

## 11. 数值正确性标准

并行 reduction 改变浮点加法顺序，因此通常不能要求与串行结果 bitwise 相同。

建议统一采用：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

不同操作应设置不同的 `atol/rtol`：

- Elementwise：较严格。
- Matmul：随 reduction 长度适当放宽。
- Softmax：除逐元素误差外，每行和应接近 1。
- LayerNorm：检查输出均值和方差。
- Gradient check：使用小输入、高精度 reference 和合适的 ε。
- 训练：比较趋势和最终区间，不要求每个 epoch 完全相同。

是否提供确定性 reduction 模式是待决定功能，不应默认承诺。

---

## 12. 并行风险登记表

| 风险 | 可能表现 | 第一版应对方式 |
|---|---|---|
| 每次算子创建线程 | 小算子比串行更慢 | 固定线程池 |
| 任务过细 | 调度成本占主导 | grain size 和串行阈值 |
| 负载不均 | 部分线程空闲 | 分块策略与 benchmark 驱动调整 |
| cache locality 差 | 线程多但速度低 | Matmul tile、连续区间切分 |
| false sharing | reduction/状态写入抖动 | 每线程局部结果分离和适当 padding |
| 嵌套并行 | 死锁或线程爆炸 | worker 内退化串行或 caller-helping |
| 浮点非结合 | 不同线程数结果不同 | 容差测试、固定合并顺序 |
| 输出写入重叠 | data race、结果偶发错误 | 每个任务独占输出区间 |
| 重叠 view 并发写 | 隐蔽 data race | 第一版限制写入和 in-place |
| worker 抛异常 | 进程终止或任务悬挂 | 捕获 `exception_ptr`，join 后重抛 |
| pool 析构竞态 | 偶发死锁/悬空任务 | 明确 shutdown 和同步语义 |
| `hardware_concurrency()==0` | 默认线程数无效 | 明确 fallback |
| 外部 BLAS 自带线程 | oversubscription | 统一线程所有权策略 |
| DataLoader 与 kernel 同时并行 | CPU 过载 | 分开配置并记录线程预算 |
| 共享 RNG | 数据竞争、不可复现 | 不共享当前 RNG；以后设计 per-worker stream |
| 内存带宽饱和 | Elementwise 不线性加速 | 区分 bandwidth-bound 与 compute-bound |
| Matmul 仅按行切分 | 缓存和负载表现差 | 输出 tile + blocking |

使用 TSan/ASan/UBSan 有帮助，但通过 sanitizer 不等于已经证明并发正确。

---

## 13. 可逆决策与高代价决策

### 13.1 较容易调整

- 文件名和目录分组。
- 是否提供 `ml.hpp` 聚合入口。
- 内部 kernel `.cpp` 如何拆分。
- 第一版任务队列的具体数据结构，前提是隐藏在稳定 API 后。
- free function 或少量语法糖。
- 是否暂时保持 Lab1 原样。
- benchmark 输出格式。

### 13.2 修改成本较高，应延迟拍板

- Tensor Storage 所有权。
- copy、alias、view 的语义。
- shape/stride/layout 表示。
- runtime DType 与 `Tensor<T>` 模板二选一。
- 非连续 Tensor 与 in-place 规则。
- autograd 图节点所有权和生命周期。
- Parameter 身份和 optimizer state 的键。
- Device、Context、Stream 的语义。
- kernel dispatch 边界。
- 异常/错误模型。
- public API 是否暴露内部容器。
- checkpoint 文件格式。

这些问题应先写小型实验或最小原型验证，不宜只靠命名讨论后直接固定。

---

## 14. Lab1 Legacy 迁移策略

### 14.1 近期

- 保持 Lab1 能运行。
- 不再为 Lab1 新增公共框架抽象。
- 参数解析、CSV、绘图继续留在实验中。
- 不让 Lab2–Lab6 的新代码依赖旧 `SerialBackend`。

### 14.2 新核心稳定以后

可选择：

1. **完全不迁移**：把 Lab1 保留为项目早期实现记录。
2. **只整理目录**：把 Sample、Dataset、PolynomialRegression、标量 Loss 和旧 optimizer 聚合进少量 Lab1 私有文件。
3. **迁移到新 Tensor**：只有在它能验证新框架、且迁移工作量有教学价值时才做。

当前推荐先采用方案 1，未来再判断。

### 14.3 名称冲突

未来 Tensor 版 `SGD` 与当前 `vector<double>` 版 `SGD` 可能冲突。项目尚未稳定发布时，可以接受一次 breaking move/rename，把旧实现迁入 Lab1，而不是永久维护两套兼容层。

---

## 15. CMake 与构建层面的原则

- 当前规模继续使用一个 `mini_ml` 静态库即可。
- 不为每一个小目录建立单独 library target。
- 所有新测试应链接 `mini_ml`，避免依赖“碰巧 header-only”。
- 线程实现通过 CMake 的 `Threads::Threads` 表达依赖，不假定只在 WSL/Linux 使用。
- 正确性测试与 benchmark 分开。
- Sanitizer 配置与 Release benchmark 分开。
- 若未来确实出现独立 backend 或很重的可选依赖，再考虑拆 target。

候选测试组织：

```text
tests/
  tensor_test.cpp
  ops_test.cpp
  execution_test.cpp
  autograd_test.cpp
  nn_test.cpp
  optim_test.cpp

benchmark/
  execution_benchmark.cpp
  matmul_benchmark.cpp
  vit_benchmark.cpp
```

这同样只是候选布局，不需要预先创建空文件。

---

## 16. 需要逐项回答的开放问题

以下问题按推荐讨论顺序排列。

### A. 项目范围

1. 最终 ViT 实验是训练、推理，还是两者都要？
2. 课程是否规定数据集、准确率、模型规模或运行时间？
3. 是否确定只做 CPU？
4. 是否允许使用 BLAS/OpenMP/TBB，还是核心实现必须只依赖 C++20 标准库？
5. 支持范围是 WSL/Linux，还是也要原生 Windows？

### B. Tensor 高代价语义

6. 第一版固定 Float32，还是立即做运行时 DType？
7. 是否需要 Float64 Tensor，还是只在测试 reference 中使用 double？
8. copy 是否共享 Storage？
9. 第一版是否开放 strided view？
10. transpose 是 view 还是 copy？
11. 非连续 kernel 自动 contiguous，还是明确报错/由调用者转换？
12. 是否允许任何形式的 in-place？

### C. 并行执行语义

13. 使用全局默认线程池还是显式 ExecutionContext？
14. 调用线程是否参与工作？
15. 第一版采用静态分块、原子索引还是简单 work stealing？
16. worker 内嵌套并行统一退化串行是否可接受？
17. 需要确定性 reduction 模式吗？
18. 正确性目标是误差容限、固定种子趋势一致，还是更严格？

### D. Autograd 与模型

19. 通用动态图 autograd 是否是课程目标，还是允许为 ViT 手写 backward？
20. backward 后默认释放图吗？
21. 是否需要重复 backward/retain graph？
22. Parameter 使用显式注册还是显式返回列表？
23. 第一版是否需要 Dropout 和 train/eval？
24. checkpoint 是否属于最终实验硬需求？

### E. 性能验收

25. 目标机器有多少物理核/逻辑核？
26. 以自研串行实现还是外部 BLAS 作为主要性能对照？
27. 用哪些 ViT shape 作为固定 benchmark？
28. 是否有最低 speedup 要求，还是重点放在正确分析扩展性？

在 A 类问题没有回答前，不应固定完整 ViT 或 DataLoader 设计；在 B 类问题没有回答前，不应开始大规模编写 Tensor API。

---

## 17. 决策记录

后续每次讨论完成一个问题，在此增加一条记录。

| ID | 日期 | 问题 | 决定 | 原因 | 影响范围 | 是否可回滚 |
|---|---|---|---|---|---|---|
| D-001 | 2026-09-17 | Lab 专属代码是否进入公共框架 | 倾向不进入，尚待最终确认 | 当前没有复用，目标是 ViT 并行框架 | include/ml 与各 lab 边界 | 是 |
| D-002 | 2026-09-17 | 是否继续抽象 Lab1 CLI/CSV/Trainer | 暂停 | 与并行主线无直接关系 | Lab1 | 是 |
| D-003 | 2026-09-17 | 新 runtime 是否沿用旧 SerialBackend | 否，作为候选原则 | 旧接口依赖具体回归模型/数据/loss | runtime、ops、Lab1 | 中等 |

新记录模板：

```text
ID：D-XXX
日期：YYYY-MM-DD
问题：
可选方案：
决定：
原因：
放弃方案及原因：
影响范围：
验证方式：
何时重新评估：
```

---

## 18. 下一次建议只讨论什么

在开始任何代码重构前，建议下一次只回答下面三个范围问题：

1. 最终 ViT 必须训练，还是只需完成推理和并行对比？
2. 核心实现是否必须只使用 C++20 标准库，能否使用 BLAS/OpenMP 作为实现或仅作为对照？
3. 第一版是否明确限定为 CPU + Float32？

这三个答案会决定是否必须实现 autograd、Tensor dtype 复杂度以及并行执行器的边界。回答完成后，再只讨论 Tensor 的 Storage/copy/view 语义。

---

## 19. 参考资料

- [PyTorch C++ Frontend](https://docs.pytorch.org/cppdocs/frontend)
- [PyTorch C++ API 总览](https://docs.pytorch.org/cppdocs/)
- [ATen Parallel API](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Parallel.h)
- [ATen native operator 与 backend dispatch 说明](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md)
- [Eigen 模块与聚合头文件](https://eigen.tuxfamily.org/dox/group__QuickRefPage.html)
- [oneDNN 基本概念：Primitive、Engine、Stream、Memory](https://uxlfoundation.github.io/oneDNN/dev_guide_basic_concepts.html)
- [ggml backend 公共接口](https://github.com/ggml-org/ggml/blob/master/include/ggml-backend.h)
- [Flashlight 项目结构](https://github.com/flashlight/flashlight)

---

## 20. 一句话原则

> 实验代码知道模型，模型知道 Tensor 算子，算子知道 CPU kernel，kernel 知道并行执行器；并行执行器不应该知道上面的任何概念。

