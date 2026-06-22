import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umap.data import make_geometric_manifold
from umap.model import UMAPOptimizer
from umap.visualizer import UMAPAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written UMAP optimization.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-neighbors", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--updates-per-frame", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    x_high, labels = make_geometric_manifold(seed=args.seed)
    optimizer = UMAPOptimizer(
        x_high=x_high,
        n_neighbors=args.n_neighbors,
        learning_rate=args.learning_rate,
    )
    animator = UMAPAnimator(
        optimizer=optimizer,
        labels=labels,
        updates_per_frame=args.updates_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
