# DBSCAN 逐步扩展动画

这个目录演示 DBSCAN 的密度聚类过程，不调用 `sklearn.cluster.DBSCAN`。

![DBSCAN 密度扩展过程演示](dbscan-demo.png)

当前实现手写：

- eps 邻域查询。
- core point 判断。
- noise 标记。
- density-reachable 簇扩展。
- Matplotlib 逐帧动画。

## 1. 运行方式

```bash
python3 dbscan/main.py
```

可调参数：

```bash
python3 dbscan/main.py --eps 0.34 --min-samples 5
python3 dbscan/main.py --eps 0.28 --min-samples 5
```

## 2. 数据是什么

数据由多个复杂几何结构组成：

- 两个月牙形簇。
- 一个螺旋簇。
- 一个紧密小岛。
- 一批均匀背景噪声点。

这个数据故意不是圆形高斯团，因为 DBSCAN 的优势是识别非凸形状和噪声点。

## 3. 核心定义

eps 邻域：

\[
N_\varepsilon(x_i)=\{x_j:\lVert x_i-x_j\rVert \le \varepsilon\}
\]

核心点：

\[
|N_\varepsilon(x_i)| \ge minPts
\]

如果一个点不是核心点，并且不能被任何簇密度到达，就会被标记为 noise。

## 4. 动画每帧表示什么

DBSCAN 会从未访问点开始：

1. 检查它的 eps 邻域。
2. 如果不是核心点，暂时标记为噪声。
3. 如果是核心点，开启新簇。
4. 把它邻域里的点加入扩展队列。
5. 如果队列中又遇到核心点，继续把它的邻域加入队列。
6. 直到队列耗尽，当前簇完成。

动画里黄色点是当前正在处理的点，黑色圆圈是它的 eps 邻域。

## 5. 为什么 DBSCAN 不需要指定 K

K-Means 必须提前指定簇数量 \(K\)。DBSCAN 不需要指定 \(K\)，它通过密度连通自动发现簇：

- 密度连通区域会变成簇。
- 稀疏孤立点会变成噪声。
- 非凸形状也能被沿着密度路径扩展出来。
