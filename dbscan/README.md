# DBSCAN Expansion Animation

English version. Chinese version: [README_CN.md](README_CN.md)

![DBSCAN density expansion demo](dbscan-demo.png)

Run:

```bash
python3 dbscan/main.py
```

This demo does not call `sklearn.cluster.DBSCAN`. It implements epsilon neighborhoods, core point detection, noise marking, and density-reachable cluster expansion. The yellow point is the current point, and the circle shows its epsilon neighborhood.
