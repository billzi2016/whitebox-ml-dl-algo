import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dbscan.data import make_density_geometry
from dbscan.model import DBSCANOptimizer
from dbscan.visualizer import DBSCANAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written DBSCAN.")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--eps", type=float, default=0.32)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    x = make_density_geometry(seed=args.seed)
    optimizer = DBSCANOptimizer(x=x, eps=args.eps, min_samples=args.min_samples)
    animator = DBSCANAnimator(
        optimizer=optimizer,
        steps_per_frame=args.steps_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
