import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsne.data import make_geometric_point_cloud
from tsne.model import TSNEOptimizer
from tsne.visualizer import TSNEAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written t-SNE optimization.")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--perplexity", type=float, default=28.0)
    parser.add_argument("--learning-rate", type=float, default=120.0)
    parser.add_argument("--updates-per-frame", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    x_high, labels = make_geometric_point_cloud(seed=args.seed)
    optimizer = TSNEOptimizer(
        x_high=x_high,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
    )
    animator = TSNEAnimator(
        optimizer=optimizer,
        labels=labels,
        updates_per_frame=args.updates_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
