import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmeans.data import make_complex_gaussian_data
from kmeans.model import KMeansOptimizer
from kmeans.visualizer import KMeansAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written K-Means.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--init-seed", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    x = make_complex_gaussian_data(seed=args.seed)
    optimizer = KMeansOptimizer(x=x, k=args.k, seed=args.init_seed)
    animator = KMeansAnimator(optimizer=optimizer, max_frames=args.max_frames)
    animator.show()


if __name__ == "__main__":
    main()
