"""
One-time data preparation for autoresearch MLX experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare_mlx.py
    python prepare_mlx.py --num-shards 8
"""

import argparse

from autoresearch_mlx.constants import CACHE_DIR, DEFAULT_PREPACKED_SEQ_LENS, MAX_SHARD
from autoresearch_mlx.data import (
    Tokenizer,
    build_prepacked_cache,
    build_token_cache,
    download_data,
    train_tokenizer,
)


def parse_seq_lens(raw: str) -> list[int]:
    seq_lens = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Prepacked cache sequence lengths must be positive integers.")
        seq_lens.append(value)
    if not seq_lens:
        raise ValueError("Provide at least one sequence length for prepacked caches.")
    return seq_lens


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
    parser.add_argument(
        "--skip-token-cache",
        action="store_true",
        help="Skip building the pretokenized shard cache.",
    )
    parser.add_argument(
        "--build-prepacked-cache",
        action="store_true",
        help="Build optional prepacked row caches for the requested sequence lengths.",
    )
    parser.add_argument(
        "--prepacked-seq-lens",
        type=str,
        default=",".join(str(seq_len) for seq_len in DEFAULT_PREPACKED_SEQ_LENS),
        help="Comma-separated sequence lengths to prepack when --build-prepacked-cache is set.",
    )
    args = parser.parse_args()
    if args.skip_token_cache and args.build_prepacked_cache:
        raise ValueError("Prepacked cache generation requires token caches; remove --skip-token-cache.")

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards
    print(f"Cache directory: {CACHE_DIR}")
    print()
    download_data(num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer()
    if not args.skip_token_cache:
        print()
        tokenizer = Tokenizer.from_directory()
        build_token_cache(tokenizer)
        if args.build_prepacked_cache:
            print()
            build_prepacked_cache(tokenizer, parse_seq_lens(args.prepacked_seq_lens))
    print()
    print("Done! Ready to train with MLX.")


if __name__ == "__main__":
    main()
