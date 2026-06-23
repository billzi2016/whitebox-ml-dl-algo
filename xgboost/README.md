# XGBoost-Style Boosting Animation

English version. Chinese version: [README_CN.md](README_CN.md)

![XGBoost-style boosting demo](xgboost-demo.png)

Run:

```bash
python3 xgboost/main.py
```

This demo does not call the `xgboost` library. It implements logloss gradients/Hessians, second-order CART regression trees, split gain, leaf weights, shrinkage, and additive logit updates.
