# t-SNE Optimization Animation

English version. Chinese version: [README_CN.md](README_CN.md)

![t-SNE geometric point cloud demo](tsne-demo.png)

Run:

```bash
python3 tsne/main.py
```

This demo does not call `sklearn.manifold.TSNE`. It implements high-dimensional similarities, low-dimensional Student-t similarities, KL divergence, and gradient descent by hand.
