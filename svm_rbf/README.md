# RBF SVM Optimization Animation

English version. Chinese version: [README_CN.md](README_CN.md)

![RBF SVM windmill XOR demo](svm_rbf-demo.png)

Run:

```bash
python3 svm_rbf/main.py iris
python3 svm_rbf/main.py xor
```

This demo does not call `sklearn.svm.SVC`. It maps inputs into RBF features centered on training samples, then optimizes a linear SVM in that feature space.
