# K-Means 逐步优化动画

这个目录演示 K-Means 的算法过程，不调用 `sklearn.cluster.KMeans`，也不是直接给最终结果。

![K-Means 逐步优化过程演示](kmeans-demo.png)

当前实现只用 `numpy` 手写：

- 复杂二维高斯混合数据生成。
- 中心初始化。
- assignment step。
- update step。
- SSE 目标函数计算。
- Matplotlib 逐帧动画和回放。

## 1. 运行方式

```bash
python3 kmeans/main.py
```

可以调整簇数量和随机种子：

```bash
python3 kmeans/main.py --k 4
python3 kmeans/main.py --k 5 --seed 31 --init-seed 2
```

## 2. 数据怎么生成

数据不是用现成数据集，而是用多个二维高斯分布自己生成。每个高斯簇都有不同的均值和协方差矩阵：

\[
x \sim \mathcal{N}(\mu, \Sigma)
\]

协方差矩阵不同，意味着簇的形状不同：

- 有的簇更宽。
- 有的簇更窄。
- 有的簇倾斜。
- 有的簇互相重叠。

另外还加入了一些均匀噪声点，让聚类过程不那么理想化。

## 3. K-Means 在优化什么

K-Means 想把样本分成 \(K\) 个簇，每个簇有一个中心：

\[
\mu_1,\mu_2,\ldots,\mu_K
\]

它优化的目标函数是 SSE：

\[
J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2
\]

其中：

- \(x_i\)：第 \(i\) 个样本。
- \(c_i\)：第 \(i\) 个样本当前所属的簇。
- \(\mu_{c_i}\)：这个簇的中心。

目标是让每个点到自己所属中心的平方距离总和尽量小。

## 4. Assignment Step

固定中心不动，把每个样本分给最近的中心：

\[
c_i = \arg\min_j \lVert x_i - \mu_j \rVert^2
\]

动画里 `phase assigned` 表示刚完成这一步。你会看到点的颜色根据最近中心重新变化。

## 5. Update Step

固定样本分配不动，把每个中心移动到自己簇内样本的均值：

\[
\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i
\]

动画里 `phase updated` 表示刚完成这一步。黑色 `X` 中心会移动，黑色轨迹线记录中心移动路径。

## 6. 为什么会收敛

K-Means 交替执行两步：

1. 固定中心，选择最近中心会让 SSE 不增加。
2. 固定分配，用均值作为中心会让 SSE 不增加。

因此：

\[
J^{(t+1)} \le J^{(t)}
\]

它会收敛到一个稳定状态。

但 K-Means 不保证找到全局最优，只保证找到局部最优。不同初始化可能得到不同结果。

## 7. 动画每帧表示什么

每一帧执行半步：

- 一帧做 assignment。
- 下一帧做 update。
- 再下一帧继续 assignment。

标题里显示：

- `frame`：当前帧。
- `iteration`：完成了多少轮中心更新。
- `phase`：当前是初始、分配后，还是更新后。
- `SSE`：当前目标函数值。

这展示的是 K-Means 的算法过程，不是直接调用库得到最终聚类。
