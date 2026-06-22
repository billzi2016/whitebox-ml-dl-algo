# RBF SVM 逐步优化动画

这个目录演示的是 RBF 特征空间里的 SVM 优化过程。这里不使用 `sklearn.svm.SVC`，因为库函数会把核矩阵、二次规划和内部优化过程全部封装掉，只给最终模型，不能展示算法一步一步怎么变。

当前实现只用 `numpy` 手写训练过程：

- 手写 RBF 特征映射。
- 手写 one-vs-rest 多分类。
- 手写 hinge loss。
- 手写 L2 正则。
- 手写 batch gradient descent。
- 每一帧保存一次参数快照，用于 `Previous`、`Next` 和进度条回放。

`sklearn.datasets` 只用于加载 Iris 数据，不参与模型训练。

## 1. 文件结构

- `data.py`：提供 `iris` 和 `xor` 两个数据入口。
- `model.py`：实现 RBF 特征映射和 SVM 优化器。
- `visualizer.py`：实现 Matplotlib 动画界面、按钮、速度条和进度条。
- `main.py`：命令行入口，只负责组装数据、模型和动画器。

## 2. 运行方式

Iris 三分类：

```bash
python3 svm_rbf/main.py iris
```

风车 XOR 二分类：

```bash
python3 svm_rbf/main.py xor
```

可调参数示例：

```bash
python3 svm_rbf/main.py xor --gamma 2.0 --c 1.0 --learning-rate 0.08
python3 svm_rbf/main.py iris --gamma 0.8 --c 1.5 --updates-per-frame 5
```

## 3. RBF 特征映射

普通线性 SVM 在原始二维空间里学习：

\[
f(x)=w^\top x+b
\]

二维空间里的线性模型只能形成直线边界。RBF 的想法是先把原始输入映射到一个由“相似度”组成的新特征空间。

当前实现把每个训练样本都当作一个 RBF 中心：

\[
c_1,c_2,\ldots,c_n
\]

对任意输入 \(x\)，第 \(j\) 个 RBF 特征是：

\[
\phi_j(x)=\exp(-\gamma \lVert x-c_j\rVert^2)
\]

于是一个二维点会被映射成：

\[
\phi(x)=
[\phi_1(x),\phi_2(x),\ldots,\phi_n(x)]
\]

然后在这个 RBF 特征空间里训练线性 SVM：

\[
f(x)=w^\top \phi(x)+b
\]

这就是“非线性边界”的来源：模型在 \(\phi(x)\) 空间是线性的，但映射回原始二维平面后，边界会变成弯曲曲线。

## 4. SVM 优化目标

二分类 SVM 的标签写成：

\[
y_i \in \{-1,+1\}
\]

margin 是：

\[
y_i f(x_i)
\]

如果：

\[
y_i f(x_i) < 1
\]

说明这个样本在 margin 内部，或者已经分错，需要产生 hinge loss：

\[
\max(0,1-y_i f(x_i))
\]

当前实现优化的是软间隔 SVM 目标：

\[
\min_{w,b}
\frac{1}{2}\lVert w\rVert^2
+
C\sum_i \max(0,1-y_i(w^\top\phi(x_i)+b))
\]

梯度下降更新是：

\[
w \leftarrow w-\eta\nabla_w L
\]

\[
b \leftarrow b-\eta\nabla_b L
\]

每一帧都会执行若干次这样的更新，然后重画分类区域。

## 5. 三分类怎么处理

Iris 是三分类，所以使用 one-vs-rest：

- setosa vs others
- versicolor vs others
- virginica vs others

每个类别都有自己的参数：

\[
(w_0,b_0),(w_1,b_1),(w_2,b_2)
\]

预测时计算三个分数：

\[
s_k(x)=w_k^\top\phi(x)+b_k
\]

最终取分数最大的类别：

\[
\hat{y}=\arg\max_k s_k(x)
\]

## 6. 风车 XOR 为什么适合 RBF

`xor` 入口会生成 16 个点，围成旋转风车，并且红蓝交替。点的极坐标半径是：

\[
r(\theta)=2.0+0.45\sin(4\theta)
\]

坐标是：

\[
x=r(\theta)\cos(\theta+\alpha)
\]

\[
y=r(\theta)\sin(\theta+\alpha)
\]

标签是：

\[
0,1,0,1,\ldots
\]

这类数据的关键是：相邻方向上的类别一直交替，线性边界没法处理。RBF 特征会让模型根据局部相似度形成弯曲边界，所以更适合这个场景。

## 7. gamma 和 C

RBF 核是：

\[
K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)
\]

`gamma` 控制局部影响范围：

- `gamma` 小：每个中心影响范围大，边界更平滑。
- `gamma` 大：每个中心影响范围小，边界更局部，也更容易过拟合。

`C` 控制 margin violation 的惩罚强度：

- `C` 小：更能容忍分错或落入 margin 的样本。
- `C` 大：更努力把训练样本分对，边界可能更复杂。

## 8. 动画每一帧表示什么

每一帧执行的是算法优化步骤：

1. 用当前 \(w,b\) 计算所有样本的 RBF 特征空间分数。
2. 计算 hinge loss。
3. 找出所有 margin violation。
4. 计算 \(w,b\) 的梯度。
5. 用梯度下降更新参数。
6. 用新参数预测整个二维网格。
7. 重新绘制背景分类区域和样本点。

标题中的指标：

- `frame`：当前第几帧。
- `updates`：累计梯度下降更新次数。
- `loss`：当前目标函数值。
- `margin violations`：当前仍然违反 margin 的 one-vs-rest 样本数量。

这才是算法过程动画，不是逐渐加点，也不是调用库直接给最终结果。
