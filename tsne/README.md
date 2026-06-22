# t-SNE 逐步优化动画

这个目录演示 t-SNE 的优化过程，不调用 `sklearn.manifold.TSNE`。重点是看点云在二维 embedding 中被高维相似度不断拉扯，逐渐形成结构。

![t-SNE 几何点云优化过程演示](tsne-demo.png)

## 1. 运行方式

```bash
python3 tsne/main.py
```

可调参数：

```bash
python3 tsne/main.py --perplexity 20 --learning-rate 100
python3 tsne/main.py --perplexity 35 --updates-per-frame 8
```

## 2. 数据是什么

数据由几何图案点云生成：

- 外层玫瑰曲线。
- 中层星形。
- 内层螺旋。

然后把二维几何坐标扩展到五维：

\[
[x,y,xy,\sin(1.4x),\cos(1.2y)]
\]

t-SNE 的输入是这个五维数据。动画中展示的是算法如何从随机二维坐标开始，把相似点慢慢拉近，把不相似点推开。

## 3. 高维相似度 P

t-SNE 先在高维空间中计算条件概率：

\[
p_{j|i} =
\frac{\exp(-\lVert x_i-x_j\rVert^2 / 2\sigma_i^2)}
\sum_{k\ne i}\exp(-\lVert x_i-x_k\rVert^2 / 2\sigma_i^2)}
\]

每个点都有自己的 \(\sigma_i\)，通过二分搜索让 perplexity 接近目标值。

然后对称化：

\[
p_{ij} = \frac{p_{j|i}+p_{i|j}}{2n}
\]

## 4. 低维相似度 Q

低维 embedding 里使用 Student-t 分布：

\[
q_{ij} =
\frac{(1+\lVert y_i-y_j\rVert^2)^{-1}}
\sum_{k\ne l}(1+\lVert y_k-y_l\rVert^2)^{-1}}
\]

这个重尾分布能缓解 crowding problem，让不相似点更容易被推开。

## 5. 优化目标

t-SNE 最小化 KL 散度：

\[
KL(P\Vert Q)=
\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}
\]

直观上：

- 如果高维里相似但低维里很远，惩罚大。
- 如果高维里不相似但低维里很近，也会通过排斥力被推开。

梯度形式可以写成：

\[
\frac{\partial C}{\partial y_i}
=
4\sum_j
(p_{ij}-q_{ij})
(y_i-y_j)
(1+\lVert y_i-y_j\rVert^2)^{-1}
\]

代码里每一步都根据这个梯度更新二维坐标。

## 6. 动画每帧表示什么

每一帧会执行多次梯度下降：

1. 计算低维相似度 \(Q\)。
2. 计算 \(P-Q\) 形成的吸引/排斥力。
3. 更新每个点的二维坐标。
4. 去掉整体平移。
5. 重新绘制 embedding。

标题里的 `KL` 是当前目标函数值。动画开始时点都挤在原点附近，随后会不断被拉扯，慢慢形成几何结构。
