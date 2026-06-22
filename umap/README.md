# UMAP 逐步优化动画

这个目录演示 UMAP 风格的低维嵌入优化过程，不调用 `umap-learn`，也不调用现成降维库。

当前实现手写：

- 几何高维点云。
- 高维 k 近邻图。
- 局部 fuzzy graph 权重。
- 低维吸引/排斥优化。
- Matplotlib 动画。

## 1. 运行方式

```bash
python3 umap/main.py
```

可调参数：

```bash
python3 umap/main.py --n-neighbors 8
python3 umap/main.py --n-neighbors 18 --learning-rate 0.06
```

## 2. 数据是什么

数据由二维几何图案生成：

- 外层波纹双环。
- 内层环。
- 对角波浪桥。

然后扩展到六维：

\[
[x,y,\sin(0.9x),\cos(0.9y),xy,\sin(\sqrt{x^2+y^2})]
\]

UMAP 输入的是六维数据，动画展示二维 embedding 如何逐步形成。

## 3. 高维邻域图

UMAP 的核心不是保留所有距离，而是保留局部邻域结构。

对每个点 \(x_i\)，找到 \(k\) 个近邻，并计算权重：

\[
w_{ij}=\exp\left(-\frac{\max(0,d(x_i,x_j)-\rho_i)}{\sigma_i}\right)
\]

其中：

- \(\rho_i\)：最近邻距离，保证局部连通性。
- \(\sigma_i\)：局部尺度。

然后把有向邻域合成无向 fuzzy union：

\[
w = a + b - ab
\]

## 4. 低维优化

低维空间里，相近点的相似度近似为：

\[
q_{ij}=\frac{1}{1+\lVert y_i-y_j\rVert^2}
\]

高维邻居会产生吸引力，非邻居会产生较弱排斥力。动画中的点就是在这种吸引和排斥之间不断移动。

## 5. 动画每帧表示什么

每一帧执行多次低维坐标更新：

1. 根据当前 embedding 计算低维相似度。
2. 高维邻居产生吸引力。
3. 非邻居产生弱排斥力。
4. 更新二维坐标。
5. 重新绘制点云。

标题里的 `cross-entropy` 是当前优化目标的近似值。

这不是直接调用 UMAP 得最终结果，而是把 UMAP 的“邻域图 + 吸引排斥优化”过程拆开演示。
