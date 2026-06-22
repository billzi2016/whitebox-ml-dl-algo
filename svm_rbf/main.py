import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svm_rbf.data import load_dataset
from svm_rbf.model import RBFKernelSVMOptimizer
from svm_rbf.visualizer import RBFSVMAnimator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate hand-written RBF-feature SVM optimization."
    )
    parser.add_argument("dataset", choices=["iris", "xor"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--updates-per-frame", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=160)
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, names, title = load_dataset(args.dataset)
    optimizer = RBFKernelSVMOptimizer(
        x_raw=x,
        y=y,
        gamma=args.gamma,
        c=args.c,
        learning_rate=args.learning_rate,
    )
    animator = RBFSVMAnimator(
        optimizer=optimizer,
        names=names,
        title=title,
        updates_per_frame=args.updates_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
