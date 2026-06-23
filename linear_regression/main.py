import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_regression.data import make_regression_data
from linear_regression.model import LinearRegressionOptimizer
from linear_regression.visualizer import LinearRegressionAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written linear regression.")
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--updates-per-frame", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = make_regression_data(seed=args.seed)
    optimizer = LinearRegressionOptimizer(x=x, y=y, learning_rate=args.learning_rate)
    animator = LinearRegressionAnimator(
        optimizer=optimizer,
        updates_per_frame=args.updates_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
