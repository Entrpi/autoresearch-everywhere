"""
One-time data preparation for autoresearch MLX experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare_mlx.py
    python prepare_mlx.py --num-shards 8
"""

import argparse

from autoresearch_mlx.constants import CACHE_DIR, MAX_SHARD
from autoresearch_mlx.data import download_data, train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch MLX")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of training shards to download (-1 = all). Val shard is always pinned.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards
    print(f"Cache directory: {CACHE_DIR}")
    print()
    download_data(num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer()
    print()
    print("Done! Ready to train with MLX.")


if __name__ == "__main__":
    main()
