from pathlib import Path

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288
PROXY_EVAL_TOKENS = 262144
CANONICAL_EVAL_SEQ_LEN = 2048
CANONICAL_EVAL_TOKENS = 262144
CANONICAL_EVAL_STEP_TOKENS = 4096

CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
TOKENIZER_DIR = CACHE_DIR / "tokenizer"
TOKEN_CACHE_DIR = CACHE_DIR / "token_cache"
PREPACKED_CACHE_DIR = CACHE_DIR / "prepacked_cache"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints"
TOKEN_CACHE_VERSION = 1
PREPACKED_CACHE_VERSION = 1
DEFAULT_PREPACKED_SEQ_LENS = (256, 512, 1024, 2048)
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"
