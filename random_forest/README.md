# 随机森林逐步构建动画

这个目录演示随机森林的算法过程，不调用 `sklearn.ensemble.RandomForestClassifier`。

![随机森林逐步构建演示](rf-demo.png)

当前实现手写：

- bootstrap 抽样。
- CART 决策树。
- 随机特征子集。
- Gini impurity。
- 节点 split 搜索。
- 多棵树投票边界。

## 1. 运行方式

```bash
python3 random_forest/main.py
```

可调参数：

```bash
python3 random_forest/main.py --n-trees 13 --max-depth 6
```

## 2. 数据是什么

数据是二维复杂二分类点云，包括：

- 两个斜向椭圆岛。
- 一个环形区域。
- 一条波浪带。
- 少量噪声点。

这个数据用来展示随机森林如何用很多轴对齐切分组合出复杂非线性边界。

## 3. Gini impurity

一个节点的数据集为 \(D\)，类别比例为 \(p_k\)，Gini impurity 是：

\[
Gini(D)=1-\sum_k p_k^2
\]

越纯的节点，Gini 越小。

## 4. Split gain

一次切分会把父节点分成左右子节点：

\[
Gain =
Gini(D)
-
\frac{|D_L|}{|D|}Gini(D_L)
-
\frac{|D_R|}{|D|}Gini(D_R)
\]

每个节点会在随机选中的特征子集上尝试多个阈值，选择 gain 最大的 split。

## 5. 随机森林为什么随机

随机森林有两层随机性：

- 每棵树使用 bootstrap 抽样的数据。
- 每个节点只看一部分随机特征。

这些随机性会让不同树犯不同错误，最后通过投票降低方差。

## 6. 动画每帧表示什么

每一帧会构建若干个节点：

1. 当前树从 bootstrap 数据开始。
2. 取一个叶子节点。
3. 随机选特征。
4. 搜索最佳阈值。
5. 用 Gini gain 分裂节点。
6. 森林投票边界更新。

背景颜色表示当前森林对二维平面每个位置的平均投票概率，黑色线是 0.5 决策边界，虚线是最新节点 split。
