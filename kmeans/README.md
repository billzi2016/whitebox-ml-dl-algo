# K-Means Optimization Animation

English version. Chinese version: [README_CN.md](README_CN.md)

![K-Means optimization demo](kmeans-demo.png)

Run:

```bash
python3 kmeans/main.py
```

This demo does not call `sklearn.cluster.KMeans`. It implements assignment, center update, and SSE by hand. Frames alternate between assigning points to the nearest center and moving centers to cluster means.
