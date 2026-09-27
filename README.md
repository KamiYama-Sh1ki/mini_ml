# mini_ml

一个用于机器学习课程实验的极简 C++ 机器学习库:仅依赖 C++20 标准库,提供一维多项式回归的完整训练管线(数据采样、模型、损失、优化器、后端),配合 Python 脚本完成可视化。

## 目录结构

```
include/ml/          头文件库(header-only 接口 + 少量 .cpp 实现)
  core/              数据集、样本、随机数、训练结果
  data/              曲线采样(sample_curve_noisy)
  model/             PolynomialRegression(Horner 求值)
  opt/               MSELoss、SGD、ConjugateGradient
  runtime/           SerialBackend(全批量损失与梯度)
src/backends/cpu/    后端实现
tests/               assert 风格单元测试(CTest 注册)
lab/lab1/           多项式回归实验:过拟合与正则化
benchmark/          预留
examples/           预留
```

## 优化器

`SGD` 与 `ConjugateGradient` 接口一致,均为 `step(weights, gradient)`:

- **SGD**:全批量梯度下降,支持 L2 weight decay。
- **ConjugateGradient**(Fletcher–Reeves):面向"关于权重为严格二次"的目标(线性模型 + MSE + L2)。利用相邻两轮梯度差 `Δg = α·H·d` 反解上一方向的精确线搜索步长并事后修正,无需额外损失评估。对病态问题(如 15 阶原始单项式基,条件数 ~10⁵)比 SGD 快若干个数量级。数值防护:β 截断到 [0, 1]、步长上限、试探步长复用上一轮精确步长。

## 快速开始

```bash
cmake -S . -B build && cmake --build build
ctest --test-dir build          # 运行全部单元测试
```

### lab1:多项式回归过拟合实验

```bash
./lab/lab1/run.sh --optimizer cg --samples 10 --degree 15 \
    --weight-decay 0 --learning-rate 0.5 --epochs 1000
```

- `--optimizer sgd|cg` 选择优化器,其余参数透传给 C++ 可执行文件;
- 每次运行在 `lab/lab1/output/<时间戳>/` 下生成 `samples/loss/checkpoints/curve/config.csv` 与 `fit.png`、`loss.png`;
- `bishop_plot.py DIR1 DIR2` 生成 Bishop 风格双面板对比图;`overlay_plot.py DIR...` 叠加多组拟合曲线;
- 典型结论:degree 15 + 10 样本 + λ=0 时,CG 收敛后训练 MSE ≪ 噪声方差而测试误差爆炸(过拟合);λ ≈ 0.005–0.01 时泛化最优。
