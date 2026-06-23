# Whitebox ML/DL Algorithms

This repository is a collection of whitebox algorithm animations. The goal is not to call a library and show the final result. The goal is to expose the internal steps of each algorithm and visualize how parameters, centers, embeddings, trees, or decision boundaries evolve over time.

Chinese version: [README_CN.md](README_CN.md)

## Demos

### 1. Linear SVM

Directory:

```bash
svm_linear/
```

![Linear SVM optimization demo](svm_linear/svm_linear-demo.png)

Run:

```bash
python3 svm_linear/svm_linear_iris_animation.py
```

Highlights:

- Uses the first two features of the Iris dataset.
- Implements hinge loss, L2 regularization, and gradient descent by hand.
- Uses one-vs-rest for the three-class problem.
- Each frame shows how the boundary changes as \(w,b\) are optimized.

### 2. RBF SVM

Directory:

```bash
svm_rbf/
```

![RBF SVM windmill XOR demo](svm_rbf/svm_rbf-demo.png)

Run:

```bash
python3 svm_rbf/main.py iris
python3 svm_rbf/main.py xor
```

Highlights:

- Does not call `sklearn.svm.SVC`.
- Implements the RBF feature map manually:

\[
\phi_j(x)=\exp(-\gamma\lVert x-c_j\rVert^2)
\]

- Optimizes a linear SVM in the RBF feature space.
- Supports both Iris classification and a 16-point windmill XOR dataset.

### 3. K-Means

Directory:

```bash
kmeans/
```

![K-Means optimization demo](kmeans/kmeans-demo.png)

Run:

```bash
python3 kmeans/main.py
```

Highlights:

- Does not call `sklearn.cluster.KMeans`.
- Generates a complex 2D Gaussian mixture dataset.
- Implements assignment, center update, and SSE manually.
- Each frame shows either point reassignment or center movement.

### 4. t-SNE

Directory:

```bash
tsne/
```

![t-SNE geometric point cloud demo](tsne/tsne-demo.png)

Run:

```bash
python3 tsne/main.py
```

Highlights:

- Does not call `sklearn.manifold.TSNE`.
- Generates a geometric point cloud.
- Implements high-dimensional similarity \(P\), low-dimensional Student-t similarity \(Q\), KL divergence, and gradient descent manually.
- Each frame shows the embedding being pulled and pushed by attraction and repulsion forces.

### 5. UMAP

Directory:

```bash
umap/
```

![UMAP neighborhood optimization demo](umap/umap-demo.png)

Run:

```bash
python3 umap/main.py
```

Highlights:

- Does not call `umap-learn`.
- Generates a geometric high-dimensional manifold.
- Implements k-nearest-neighbor graph construction, fuzzy graph weights, and low-dimensional attraction/repulsion optimization.
- Each frame shows how the embedding unfolds while preserving neighborhood structure.

### 6. DBSCAN

Directory:

```bash
dbscan/
```

![DBSCAN density expansion demo](dbscan/dbscan-demo.png)

Run:

```bash
python3 dbscan/main.py
```

Highlights:

- Does not call `sklearn.cluster.DBSCAN`.
- Generates a density-based dataset with moons, a spiral, an island, and noise.
- Implements epsilon neighborhoods, core point detection, noise marking, and density-reachable expansion.
- Each frame shows how DBSCAN grows clusters from core points.

### 7. PCA

Directory:

```bash
pca/
```

![PCA power iteration demo](pca/pca-demo.png)

Run:

```bash
python3 pca/main.py
```

Highlights:

- Does not call `sklearn.decomposition.PCA`.
- Generates a correlated 3D point cloud.
- Implements centering, covariance matrix construction, power iteration, and orthogonalization.
- Each frame shows the principal directions rotating toward maximum-variance directions.

### 8. Random Forest

Directory:

```bash
random_forest/
```

![Random Forest construction demo](random_forest/rf-demo.png)

Run:

```bash
python3 random_forest/main.py
```

Highlights:

- Does not call `sklearn.ensemble.RandomForestClassifier`.
- Generates a complex 2D classification dataset.
- Implements bootstrap sampling, CART node splitting, Gini impurity, and forest voting.
- Each frame shows new node splits and the evolving vote boundary.

### 9. XGBoost-Style Gradient Boosted Trees

Directory:

```bash
xgboost/
```

![XGBoost-style boosting demo](xgboost/xgboost-demo.png)

Run:

```bash
python3 xgboost/main.py
```

Highlights:

- Does not call the `xgboost` library.
- Generates a complex 2D binary classification dataset.
- Implements logloss gradients/Hessians, second-order CART trees, split gain, and leaf weights manually.
- Each frame shows how boosted trees gradually correct the current probability boundary.

### 10. Linear Regression

Directory:

```bash
linear_regression/
```

Run:

```bash
python3 linear_regression/main.py
```

Highlights:

- Does not call `sklearn.linear_model.LinearRegression`.
- Generates noisy one-dimensional regression data with a few outliers.
- Implements MSE, gradients, and gradient descent manually.
- Each frame shows how the fitted line moves as \(w,b\) are optimized.

### 11. Optimizer Trajectories

Directory:

```bash
optimizers/
```

Run:

```bash
python3 optimizers/main.py
```

Highlights:

- Compares GD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, and AdamW.
- Uses PyTorch optimizer implementations step by step when available.
- Falls back to NumPy update rules if PyTorch cannot be imported.
- Each frame moves colored balls across the same 3D loss surface.

## Design Principles

- Keep the training process as explicit as practical instead of hiding it inside library calls.
- Use Matplotlib for consistent interactive visualizations.
- Animations include `Previous`, `Next`, `Auto Play`, `Speed`, and `Frame` controls.
- READMEs explain the formulas, the algorithmic process, and what each frame represents.
- Datasets are selected or generated to reveal the behavior of each algorithm, not merely to maximize a score.
