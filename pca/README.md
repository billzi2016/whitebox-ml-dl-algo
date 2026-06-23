# PCA Power Iteration Animation

English version. Chinese version: [README_CN.md](README_CN.md)

![PCA power iteration demo](pca-demo.png)

Run:

```bash
python3 pca/main.py
```

This demo does not call `sklearn.decomposition.PCA`. It implements centering, covariance matrix construction, power iteration, orthogonalization, and projection onto the current PC1/PC2 directions.
