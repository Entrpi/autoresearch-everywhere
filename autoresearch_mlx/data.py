import hashlib
import json
import math
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken

from .constants import (
    BASE_URL,
    BOS_TOKEN,
    DATA_DIR,
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    MAX_SHARD,
    SPECIAL_TOKENS,
    SPLIT_PATTERN,
    TOKEN_CACHE_DIR,
    TOKEN_CACHE_VERSION,
    TOKENIZER_DIR,
    VAL_FILENAME,
    VAL_SHARD,
    VOCAB_SIZE,
)


def download_single_shard(index: int) -> bool:
    filename = f"shard_{index:05d}.parquet"
    filepath = DATA_DIR / filename
    if filepath.exists():
        return True

    url = f"{BASE_URL}/{filename}"
    temp_path = filepath.with_suffix(".parquet.tmp")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {exc}")
            for path in (temp_path, filepath):
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2**attempt)
    return False


def download_data(num_shards: int, download_workers: int = 8) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    shard_ids = list(range(num_train))
    if VAL_SHARD not in shard_ids:
        shard_ids.append(VAL_SHARD)

    existing = sum(1 for shard_id in shard_ids if (DATA_DIR / f"shard_{shard_id:05d}.parquet").exists())
    if existing == len(shard_ids):
        print(f"Data: all {len(shard_ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(shard_ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, shard_ids)

    ok = sum(1 for ready in results if ready)
    print(f"Data: {ok}/{len(shard_ids)} shards ready at {DATA_DIR}")
    if ok != len(shard_ids):
        raise RuntimeError("Data download did not complete successfully.")


def list_parquet_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.suffix == ".parquet" and not path.name.endswith(".tmp")
    )


def text_iterator(max_chars: int = 1_000_000_000, doc_cap: int = 10_000):
    parquet_paths = [path for path in list_parquet_files() if path.name != VAL_FILENAME]
    if not parquet_paths:
        raise RuntimeError("Tokenizer training needs at least one non-validation parquet shard.")

    nchars = 0
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            for text in row_group.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer() -> None:
    tokenizer_pkl = TOKENIZER_DIR / "tokenizer.pkl"
    token_bytes_path = TOKENIZER_DIR / "token_bytes.npy"
    if tokenizer_pkl.exists() and token_bytes_path.exists():
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        raise RuntimeError("Tokenizer training needs at least 2 shards (1 train + 1 val).")

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks = {bytes(token): rank for token, rank in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {
        name: tokens_offset + idx
        for idx, name in enumerate(SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with tokenizer_pkl.open("wb") as handle:
        pickle.dump(enc, handle)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes = np.zeros((enc.n_vocab,), dtype=np.int32)
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str not in special_set:
            token_bytes[token_id] = len(token_str.encode("utf-8"))
    np.save(token_bytes_path, token_bytes)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    if decoded != test:
        raise RuntimeError(f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}")
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


class Tokenizer:
    def __init__(self, encoding: tiktoken.Encoding, *, fingerprint: str | None = None):
        self.enc = encoding
        self.bos_token_id = encoding.encode_single_token(BOS_TOKEN)
        self.fingerprint = fingerprint

    @classmethod
    def from_directory(cls, tokenizer_dir: Path = TOKENIZER_DIR) -> "Tokenizer":
        tokenizer_pkl = tokenizer_dir / "tokenizer.pkl"
        payload = tokenizer_pkl.read_bytes()
        fingerprint = hashlib.sha256(payload).hexdigest()[:16]
        return cls(pickle.loads(payload), fingerprint=fingerprint)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads: int = 8):
        prepend_id = None
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            return ids
        if isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            return ids
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def load_token_bytes() -> np.ndarray:
    return np.load(TOKENIZER_DIR / "token_bytes.npy")


def _split_parquet_paths(split: str) -> list[Path]:
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise RuntimeError("No parquet files found. Run prepare_mlx.py first.")

    val_path = DATA_DIR / VAL_FILENAME
    if split == "train":
        parquet_paths = [path for path in parquet_paths if path != val_path]
        if not parquet_paths:
            raise RuntimeError("No training shards available. Run prepare_mlx.py with at least 1 training shard.")
    else:
        parquet_paths = [val_path]
    return parquet_paths


def _token_cache_paths(parquet_path: Path) -> dict[str, Path]:
    base = TOKEN_CACHE_DIR / parquet_path.stem
    return {
        "tokens": base.with_suffix(".tokens.bin"),
        "offsets": base.with_suffix(".offsets.npy"),
        "meta": base.with_suffix(".meta.json"),
    }


def _load_token_cache_meta(parquet_path: Path) -> dict | None:
    meta_path = _token_cache_paths(parquet_path)["meta"]
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _has_valid_token_cache(parquet_path: Path, tokenizer: Tokenizer) -> bool:
    paths = _token_cache_paths(parquet_path)
    if not all(path.exists() for path in paths.values()):
        return False
    meta = _load_token_cache_meta(parquet_path)
    if meta is None:
        return False
    return (
        meta.get("version") == TOKEN_CACHE_VERSION
        and meta.get("tokenizer_fingerprint") == tokenizer.fingerprint
        and meta.get("bos_token_id") == tokenizer.get_bos_token_id()
        and meta.get("source_parquet") == parquet_path.name
    )


def _split_has_token_cache(split: str, tokenizer: Tokenizer) -> bool:
    parquet_paths = _split_parquet_paths(split)
    return all(_has_valid_token_cache(parquet_path, tokenizer) for parquet_path in parquet_paths)


def build_token_cache(tokenizer: Tokenizer, tokenizer_batch_size: int = 128) -> None:
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_paths = list_parquet_files()
    if not parquet_paths:
        raise RuntimeError("No parquet files found. Run prepare_mlx.py first.")
    if tokenizer.fingerprint is None:
        raise RuntimeError("Tokenizer fingerprint missing; token cache cannot be validated.")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise RuntimeError("Token cache expects vocab_size <= 65535 for uint16 storage.")

    ready = sum(1 for parquet_path in parquet_paths if _has_valid_token_cache(parquet_path, tokenizer))
    if ready == len(parquet_paths):
        print(f"Token cache: all {ready} shard caches already built at {TOKEN_CACHE_DIR}")
        return

    print(f"Token cache: building {len(parquet_paths) - ready} shard caches at {TOKEN_CACHE_DIR}...")
    for parquet_path in parquet_paths:
        if _has_valid_token_cache(parquet_path, tokenizer):
            continue
        _build_single_token_cache(parquet_path, tokenizer, tokenizer_batch_size=tokenizer_batch_size)


def _build_single_token_cache(parquet_path: Path, tokenizer: Tokenizer, tokenizer_batch_size: int = 128) -> None:
    paths = _token_cache_paths(parquet_path)
    temp_tokens = paths["tokens"].with_name(paths["tokens"].name + ".tmp")
    temp_offsets = paths["offsets"].with_name(paths["offsets"].name + ".tmp")
    temp_meta = paths["meta"].with_name(paths["meta"].name + ".tmp")

    offsets = [0]
    total_tokens = 0
    doc_count = 0
    bos_token = tokenizer.get_bos_token_id()

    print(f"  Caching {parquet_path.name}")
    parquet_file = pq.ParquetFile(parquet_path)
    with temp_tokens.open("wb") as handle:
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            batch = row_group.column("text").to_pylist()
            for start in range(0, len(batch), tokenizer_batch_size):
                token_lists = tokenizer.encode(batch[start:start + tokenizer_batch_size], prepend=bos_token)
                for doc in token_lists:
                    doc_array = np.asarray(doc, dtype=np.uint16)
                    doc_array.tofile(handle)
                    total_tokens += int(doc_array.size)
                    offsets.append(total_tokens)
                    doc_count += 1

    with temp_offsets.open("wb") as handle:
        np.save(handle, np.asarray(offsets, dtype=np.int64))
    meta = {
        "version": TOKEN_CACHE_VERSION,
        "source_parquet": parquet_path.name,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "bos_token_id": bos_token,
        "doc_count": doc_count,
        "token_count": total_tokens,
        "dtype": "uint16",
    }
    temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    temp_tokens.replace(paths["tokens"])
    temp_offsets.replace(paths["offsets"])
    temp_meta.replace(paths["meta"])


def _document_text_batches(parquet_paths: list[Path], tokenizer_batch_size: int = 128):
    epoch = 1
    while True:
        for filepath in parquet_paths:
            parquet_file = pq.ParquetFile(filepath)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_idx)
                batch = row_group.column("text").to_pylist()
                for start in range(0, len(batch), tokenizer_batch_size):
                    yield batch[start:start + tokenizer_batch_size], epoch
        epoch += 1


def _document_token_batches(
    split: str,
    tokenizer: Tokenizer,
    tokenizer_batch_size: int = 128,
    *,
    use_cache: bool | None = None,
):
    parquet_paths = _split_parquet_paths(split)
    if use_cache is None:
        use_cache = _split_has_token_cache(split, tokenizer)
    if use_cache:
        yield from _cached_document_batches(parquet_paths, tokenizer_batch_size=tokenizer_batch_size)
        return

    bos_token = tokenizer.get_bos_token_id()
    for text_batch, epoch in _document_text_batches(parquet_paths, tokenizer_batch_size=tokenizer_batch_size):
        yield tokenizer.encode(text_batch, prepend=bos_token), epoch


def _cached_document_batches(parquet_paths: list[Path], tokenizer_batch_size: int = 128):
    epoch = 1
    while True:
        for parquet_path in parquet_paths:
            paths = _token_cache_paths(parquet_path)
            offsets = np.load(paths["offsets"], mmap_mode="r")
            tokens = np.memmap(paths["tokens"], dtype=np.uint16, mode="r")
            batch: list[np.ndarray] = []
            for doc_idx in range(len(offsets) - 1):
                start = int(offsets[doc_idx])
                end = int(offsets[doc_idx + 1])
                batch.append(tokens[start:end])
                if len(batch) == tokenizer_batch_size:
                    yield batch, epoch
                    batch = []
            if batch:
                yield batch, epoch
        epoch += 1


class PackedDataLoader:
    def __init__(self, tokenizer: Tokenizer, batch_size: int, seq_len: int, split: str, buffer_size: int = 1000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.row_capacity = seq_len + 1
        self.buffer_size = buffer_size
        self.using_token_cache = _split_has_token_cache(split, tokenizer)
        source = "token cache" if self.using_token_cache else "parquet + tokenizer fallback"
        print(f"Data loader ({split}): using {source}.")
        self.batches = _document_token_batches(split, tokenizer, use_cache=self.using_token_cache)
        self.doc_buffer: list[np.ndarray | list[int]] = []
        self.epoch = 1
        self.row_buffer = np.empty((batch_size, self.row_capacity), dtype=np.int32)
        self.inputs = np.empty((batch_size, seq_len), dtype=np.int32)
        self.targets = np.empty((batch_size, seq_len), dtype=np.int32)

    def __iter__(self) -> "PackedDataLoader":
        return self

    def refill_buffer(self) -> None:
        token_batch, self.epoch = next(self.batches)
        self.doc_buffer.extend(token_batch)

    def __next__(self):
        for row_idx in range(self.batch_size):
            pos = 0
            while pos < self.row_capacity:
                while len(self.doc_buffer) < self.buffer_size:
                    self.refill_buffer()

                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = idx
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    self.row_buffer[row_idx, pos:pos + len(doc)] = np.asarray(doc, dtype=np.int32)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(self.doc_buffer)), key=lambda idx: len(self.doc_buffer[idx]))
                    doc = self.doc_buffer.pop(shortest_idx)
                    self.row_buffer[row_idx, pos:pos + remaining] = np.asarray(doc[:remaining], dtype=np.int32)
                    pos += remaining

        self.inputs[...] = self.row_buffer[:, :-1]
        self.targets[...] = self.row_buffer[:, 1:]
        return mx.array(self.inputs), mx.array(self.targets), self.epoch


def make_dataloader(tokenizer: Tokenizer, batch_size: int, seq_len: int, split: str, buffer_size: int = 1000) -> PackedDataLoader:
    if split not in {"train", "val"}:
        raise ValueError(f"Invalid split: {split}")
    return PackedDataLoader(tokenizer, batch_size, seq_len, split, buffer_size=buffer_size)


@mx.compile
def _masked_token_sums(loss_flat, target_ids, token_bytes):
    nbytes = token_bytes[target_ids]
    mask = nbytes > 0
    return mx.sum(loss_flat * mask.astype(loss_flat.dtype)), mx.sum(nbytes.astype(mx.int64))


def evaluate_bpb(
    model,
    tokenizer: Tokenizer,
    batch_size: int,
    *,
    seq_len: int = MAX_SEQ_LEN,
    eval_tokens: int = EVAL_TOKENS,
) -> float:
    token_bytes = mx.array(load_token_bytes())
    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val")
    steps = max(1, eval_tokens // (batch_size * seq_len))
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").reshape((-1,))
        y_flat = y.reshape((-1,))
        batch_nats, batch_bytes = _masked_token_sums(loss_flat, y_flat, token_bytes)
        mx.eval(batch_nats, batch_bytes)
        total_nats += batch_nats.item()
        total_bytes += batch_bytes.item()

    return total_nats / (math.log(2) * total_bytes)
