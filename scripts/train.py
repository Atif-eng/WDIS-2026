"""CLI script for training the WiDS wildfire model."""

from __future__ import annotations

import argparse
from pathlib import Path

from wids_2026.config import DATA_DIR
from wids_2026.pipeline import train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WiDS wildfire prediction model.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Directory containing train.csv and test.csv.")
    parser.add_argument("--target", type=str, default="event", help="Target column name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores, _, _ = train_pipeline(args.data_dir, target_column=args.target)
    print("Cross-validation F1 scores:")
    for name, value in scores.items():
        print(f"- {name}: {value:.4f}")


if __name__ == "__main__":
    main()
