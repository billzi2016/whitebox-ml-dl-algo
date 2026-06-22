import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_forest.data import make_forest_classification_data
from random_forest.forest import RandomForestBuilder
from random_forest.visualizer import RandomForestAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written random forest.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--n-trees", type=int, default=9)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=160)
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = make_forest_classification_data(seed=args.seed)
    forest = RandomForestBuilder(
        x=x,
        y=y,
        n_trees=args.n_trees,
        max_depth=args.max_depth,
    )
    animator = RandomForestAnimator(
        forest=forest,
        steps_per_frame=args.steps_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
