import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost.data import make_boosting_classification_data
from xgboost.model import XGBoostBuilder
from xgboost.visualizer import XGBoostAnimator


def parse_args():
    parser = argparse.ArgumentParser(description="Animate hand-written XGBoost-style boosting.")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--n-estimators", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--steps-per-frame", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=160)
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = make_boosting_classification_data(seed=args.seed)
    booster = XGBoostBuilder(
        x=x,
        y=y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    animator = XGBoostAnimator(
        booster=booster,
        steps_per_frame=args.steps_per_frame,
        max_frames=args.max_frames,
    )
    animator.show()


if __name__ == "__main__":
    main()
