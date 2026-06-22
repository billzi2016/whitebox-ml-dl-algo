import numpy as np


class DBSCANOptimizer:
    """手写 DBSCAN 的逐步状态机：找核心点、扩展簇、标记噪声。"""

    UNVISITED = -99
    NOISE = -1

    def __init__(self, x, eps=0.32, min_samples=6):
        self.x = x
        self.eps = eps
        self.min_samples = min_samples
        self.labels = np.full(len(x), self.UNVISITED, dtype=int)
        self.visited = np.zeros(len(x), dtype=bool)
        self.neighbors = self._precompute_neighbors()
        self.core_points = np.array(
            [len(items) >= self.min_samples for items in self.neighbors],
            dtype=bool,
        )

        self.cluster_id = 0
        self.queue = []
        self.current_point = None
        self.phase = "search"
        self.finished = False

    def _precompute_neighbors(self):
        """预计算 eps 邻域，动画时只展示过程，不重复算距离。"""
        diff = self.x[:, None, :] - self.x[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        return [np.where(distances[i] <= self.eps)[0].tolist() for i in range(len(self.x))]

    def _next_unvisited(self):
        """找到下一个还没处理过的点。"""
        candidates = np.where(~self.visited)[0]
        if len(candidates) == 0:
            return None
        return int(candidates[0])

    def _start_or_mark_noise(self, point):
        """处理一个新点：核心点开新簇，非核心点暂时标为噪声。"""
        self.current_point = point
        self.visited[point] = True

        if not self.core_points[point]:
            self.labels[point] = self.NOISE
            self.phase = "mark noise"
            return

        self.labels[point] = self.cluster_id
        self.queue = [idx for idx in self.neighbors[point] if idx != point]
        self.phase = "start cluster"

    def _expand_one(self):
        """从当前簇的队列中弹出一个点，执行一次密度可达扩展。"""
        if not self.queue:
            self.cluster_id += 1
            self.phase = "finish cluster"
            return

        point = self.queue.pop(0)
        self.current_point = point

        if not self.visited[point]:
            self.visited[point] = True
            if self.core_points[point]:
                # 核心点会把自己的邻居继续并入扩展队列。
                for neighbor in self.neighbors[point]:
                    if neighbor not in self.queue:
                        self.queue.append(neighbor)

        if self.labels[point] in {self.UNVISITED, self.NOISE}:
            self.labels[point] = self.cluster_id

        self.phase = "expand cluster"

    def step(self):
        """执行 DBSCAN 的一个可视化小步骤。"""
        if self.finished:
            return

        if self.queue:
            self._expand_one()
            return

        # 当前簇的扩展队列已经耗尽，说明这个簇完整结束。
        # 必须先递增 cluster_id，再去寻找下一个未访问核心点；
        # 否则后续互不连通的簇会被错误地继续标成 cluster 0。
        if self.phase in {"start cluster", "expand cluster"}:
            self.cluster_id += 1
            self.phase = "finish cluster"
            self.current_point = None
            return

        point = self._next_unvisited()
        if point is None:
            self.finished = True
            self.phase = "finished"
            self.current_point = None
            return

        self._start_or_mark_noise(point)

    def snapshot(self):
        """保存当前 DBSCAN 状态。"""
        assigned = int(np.sum(self.labels >= 0))
        noise = int(np.sum(self.labels == self.NOISE))
        return {
            "labels": self.labels.copy(),
            "visited": self.visited.copy(),
            "queue": list(self.queue),
            "cluster_id": self.cluster_id,
            "current_point": self.current_point,
            "phase": self.phase,
            "finished": self.finished,
            "assigned": assigned,
            "noise": noise,
        }

    def restore(self, snapshot):
        """恢复某一帧状态。"""
        self.labels = snapshot["labels"].copy()
        self.visited = snapshot["visited"].copy()
        self.queue = list(snapshot["queue"])
        self.cluster_id = snapshot["cluster_id"]
        self.current_point = snapshot["current_point"]
        self.phase = snapshot["phase"]
        self.finished = snapshot["finished"]
