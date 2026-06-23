import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.engines import OptimizerComparison
from optimizers.visualizer import OptimizerAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Compare optimizers step by step on a 3D surface.")
    parser.add_argument("--lr", type=float, default=0.055)
    parser.add_argument("--max-frames", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    comparison = OptimizerComparison(lr=args.lr)
    animator = OptimizerAnimator(comparison=comparison, max_frames=args.max_frames)
    animator.show()


if __name__ == "__main__":
    main()
