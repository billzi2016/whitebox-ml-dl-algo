# Linear SVM Optimization on Iris

English version. Chinese version: [README_CN.md](README_CN.md)

![Linear SVM optimization demo](svm_linear-demo.png)

Run:

```bash
python3 svm_linear/svm_linear_iris_animation.py
```

This demo does not call `sklearn.svm.SVC`. It implements a linear SVM objective with hinge loss, L2 regularization, gradient descent, and one-vs-rest classification for the three Iris classes.
