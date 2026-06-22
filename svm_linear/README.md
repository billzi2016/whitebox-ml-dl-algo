# Iris 三分类线性 SVM 逐步优化演示

这个目录里的 `svm_linear_iris_animation.py` 不是调用 `sklearn.svm.SVC` 后只画最终结果，而是手写了一个用于教学演示的线性 SVM 优化过程。界面里的每一帧表示参数 \(w, b\) 又做了若干次梯度下降更新，所以它展示的是“模型怎么一步一步优化”，不是“样本一点一点加入”。

## 1. 数据和二维展示

Iris 数据集有三类花：

- setosa
- versicolor
- virginica

原始 Iris 有 4 个特征。为了能画在普通的二维 `plt` 图里，这个演示只取前两个特征：

- sepal length
- sepal width

也就是说，每个样本可以写成：

\[
x_i =
\begin{bmatrix}
x_{i1} \\
x_{i2}
\end{bmatrix}
\]

因为 SVM 对尺度敏感，代码先手写标准化：

\[
\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
\]

后续优化都在标准化后的特征 \(\tilde{x}\) 上进行，但画图时坐标轴仍然显示原始厘米单位。

## 2. 二分类线性 SVM 的目标

最基础的线性 SVM 是二分类模型。它学习一个线性打分函数：

\[
f(x) = w^\top x + b
\]

其中：

- \(w\) 是权重向量
- \(b\) 是偏置
- \(x\) 是输入特征

二分类标签通常写成：

\[
y_i \in \{-1, +1\}
\]

SVM 希望正确样本不只是分对，还要离边界有足够间隔：

\[
y_i(w^\top x_i + b) \ge 1
\]

如果这个值小于 1，说明样本在 margin 里面或者已经被分错，需要受到惩罚。这个惩罚就是 hinge loss：

\[
\max(0, 1 - y_i(w^\top x_i + b))
\]

软间隔线性 SVM 的优化目标是：

\[
\min_{w,b}
\frac{1}{2}\lVert w\rVert^2
+
C \sum_{i=1}^{n}
\max(0, 1 - y_i(w^\top x_i + b))
\]

两部分含义分别是：

- \(\frac{1}{2}\lVert w\rVert^2\)：让间隔变大，避免边界过于激进。
- \(C \sum hinge\ loss\)：惩罚分错或者进入间隔内的样本。

## 3. 为什么这里不用 `sklearn.svm.SVC`

传统 SVM 训练通常是批处理优化。像 `sklearn.svm.SVC` 这类接口更适合输入整批数据，然后直接得到最终模型。它不会把内部二次规划求解过程一帧一帧暴露出来。

这个演示的目标是看“优化过程”，所以代码没有用 `SVC` 来训练。它手写了线性 SVM 的 hinge loss 和梯度下降更新，让每一帧都能看到参数变化后的边界。

严格说，这不是 LIBSVM 的精确二次规划求解轨迹，而是用 batch gradient descent 对线性 SVM 目标进行教学式优化。这样更适合可视化每一步。

## 4. 一步梯度下降在做什么

对某个二分类器来说，如果样本满足：

\[
y_i(w^\top x_i + b) < 1
\]

它就是一个违反 margin 的样本，会参与梯度。

代码使用的目标是：

\[
L(w,b)
=
\frac{1}{2}\lVert w\rVert^2
+
C \sum_i \max(0, 1 - y_i(w^\top x_i + b))
\]

对 \(w\) 的梯度可以理解为：

\[
\nabla_w L
=
w
-
C \sum_{i \in V} y_i x_i
\]

对 \(b\) 的梯度是：

\[
\nabla_b L
=
-
C \sum_{i \in V} y_i
\]

其中 \(V\) 是所有违反 margin 的样本集合：

\[
V = \{i \mid y_i(w^\top x_i + b) < 1\}
\]

然后用梯度下降更新：

\[
w \leftarrow w - \eta \nabla_w L
\]

\[
b \leftarrow b - \eta \nabla_b L
\]

这里 \(\eta\) 是学习率。动画中的每一帧会执行若干次这样的更新，所以你看到的边界会随着 loss 下降逐步移动。

## 5. 三分类怎么做

SVM 原始形式是二分类。Iris 是三分类，所以这里用 one-vs-rest：

- 分类器 0：setosa 是 \(+1\)，其他两类是 \(-1\)
- 分类器 1：versicolor 是 \(+1\)，其他两类是 \(-1\)
- 分类器 2：virginica 是 \(+1\)，其他两类是 \(-1\)

因此会学到三组参数：

\[
(w_0, b_0), (w_1, b_1), (w_2, b_2)
\]

预测时，对同一个点分别计算三个分数：

\[
s_k(x) = w_k^\top x + b_k
\]

最终类别取分数最大的那个：

\[
\hat{y} = \arg\max_k s_k(x)
\]

图中的三种背景颜色就是在二维网格上对每个点计算 \(\arg\max\) 后得到的分类区域。

## 6. 动画每一帧代表什么

每一帧做的是：

1. 从当前 \(w, b\) 出发。
2. 对三个 one-vs-rest 分类器分别计算 hinge loss。
3. 找出违反 margin 的样本。
4. 根据公式计算梯度。
5. 更新 \(w, b\)。
6. 用更新后的参数重新预测整张二维网格。
7. 重新绘制三分类背景区域和样本点。

标题里的指标含义：

- `frame`：当前第几帧。
- `updates`：到当前为止已经做了多少次参数更新。
- `loss`：当前 SVM 目标函数值。
- `margin violations`：当前仍然违反 margin 的 one-vs-rest 样本数量。

如果优化有效，通常可以看到 loss 逐渐下降，分类区域逐渐稳定。

## 7. 界面控件

界面底部有四个控件：

- `Previous`：回到上一帧的参数状态。
- `Next`：前进一步，执行新的优化更新。
- `Auto Play` / `Pause`：自动播放或暂停。
- `Speed`：调节自动播放速度。

`Previous` 和 `Next` 是逐帧控制组；`Auto Play` 和 `Speed` 是播放控制组。
