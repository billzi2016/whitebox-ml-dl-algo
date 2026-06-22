import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pca.data import make_correlated_3d_data
from pca.model import PCAOptimizer
from pca.visualizer import PCAAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written PCA power iteration.")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--init-seed", type=int, default=3)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    x = make_correlated_3d_data(seed=args.seed)
    optimizer = PCAOptimizer(x=x, seed=args.init_seed)
    animator = PCAAnimator(
        optimizer=optimizer,
        steps_per_frame=args.steps_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
