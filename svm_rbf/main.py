import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svm_rbf.data import load_dataset
from svm_rbf.model import RBFKernelSVMOptimizer
from svm_rbf.visualizer import RBFSVMAnimator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate hand-written RBF-feature SVM optimization."
    )
    parser.add_argument("dataset", choices=["iris", "xor"])
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--updates-per-frame", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=160)
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, names, title = load_dataset(args.dataset)

    # XOR 风车数据是刻意构造的强非线性问题，需要更局部的 RBF 和更强的惩罚；
    # Iris 的二维投影更噪声化，默认参数保守一些，避免边界过度贴点。
    if args.dataset == "xor":
        gamma = 6.0 if args.gamma is None else args.gamma
        c = 10.0 if args.c is None else args.c
        learning_rate = 0.1 if args.learning_rate is None else args.learning_rate
    else:
        gamma = 1.0 if args.gamma is None else args.gamma
        c = 1.0 if args.c is None else args.c
        learning_rate = 0.08 if args.learning_rate is None else args.learning_rate

    optimizer = RBFKernelSVMOptimizer(
        x_raw=x,
        y=y,
        gamma=gamma,
        c=c,
        learning_rate=learning_rate,
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
