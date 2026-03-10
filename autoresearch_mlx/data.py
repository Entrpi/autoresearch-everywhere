import bisect
import hashlib
import json
import math
import pickle
import time
from collections import deque
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
    EVAL_SLICE_TARGET_STEPS,
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    MAX_SHARD,
    PREPACKED_CACHE_DIR,
    PREPACKED_CACHE_VERSION,
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


def _prepacked_cache_paths(split: str, seq_len: int) -> dict[str, Path]:
    base = PREPACKED_CACHE_DIR / f"{split}_seq{seq_len}"
    return {
        "rows": base.with_suffix(".rows.bin"),
        "meta": base.with_suffix(".meta.json"),
    }


def _load_prepacked_cache_meta(split: str, seq_len: int) -> dict | None:
    meta_path = _prepacked_cache_paths(split, seq_len)["meta"]
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _has_valid_prepacked_cache(split: str, seq_len: int, tokenizer: Tokenizer) -> bool:
    paths = _prepacked_cache_paths(split, seq_len)
    if not all(path.exists() for path in paths.values()):
        return False
    meta = _load_prepacked_cache_meta(split, seq_len)
    if meta is None:
        return False
    parquet_paths = _split_parquet_paths(split)
    return (
        meta.get("version") == PREPACKED_CACHE_VERSION
        and meta.get("split") == split
        and meta.get("seq_len") == seq_len
        and meta.get("row_capacity") == seq_len + 1
        and meta.get("tokenizer_fingerprint") == tokenizer.fingerprint
        and meta.get("bos_token_id") == tokenizer.get_bos_token_id()
        and meta.get("source_parquets") == [path.name for path in parquet_paths]
        and meta.get("dtype") == "uint16"
        and int(meta.get("row_count", 0)) > 0
    )


def _split_has_prepacked_cache(split: str, seq_len: int, tokenizer: Tokenizer) -> bool:
    return _has_valid_prepacked_cache(split, seq_len, tokenizer)


def build_prepacked_cache(
    tokenizer: Tokenizer,
    seq_lens: list[int] | tuple[int, ...],
    *,
    splits: tuple[str, ...] = ("train", "val"),
    buffer_size: int = 1000,
) -> None:
    PREPACKED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if tokenizer.fingerprint is None:
        raise RuntimeError("Tokenizer fingerprint missing; prepacked cache cannot be validated.")
    if not all(_split_has_token_cache(split, tokenizer) for split in splits):
        raise RuntimeError("Prepacked cache requires token caches for all requested splits.")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise RuntimeError("Prepacked cache expects vocab_size <= 65535 for uint16 storage.")

    work = [(split, seq_len) for split in splits for seq_len in seq_lens]
    ready = sum(1 for split, seq_len in work if _has_valid_prepacked_cache(split, seq_len, tokenizer))
    if ready == len(work):
        print(f"Prepacked cache: all {ready} caches already built at {PREPACKED_CACHE_DIR}")
        return

    print(f"Prepacked cache: building {len(work) - ready} caches at {PREPACKED_CACHE_DIR}...")
    for split, seq_len in work:
        if _has_valid_prepacked_cache(split, seq_len, tokenizer):
            continue
        _build_single_prepacked_cache(split, seq_len, tokenizer, buffer_size=buffer_size)


def _iter_cached_documents_once(parquet_paths: list[Path]):
    for parquet_path in parquet_paths:
        paths = _token_cache_paths(parquet_path)
        offsets = np.load(paths["offsets"], mmap_mode="r")
        tokens = np.memmap(paths["tokens"], dtype=np.uint16, mode="r")
        for doc_idx in range(len(offsets) - 1):
            start = int(offsets[doc_idx])
            end = int(offsets[doc_idx + 1])
            yield tokens[start:end]


def _build_single_prepacked_cache(split: str, seq_len: int, tokenizer: Tokenizer, *, buffer_size: int = 1000) -> None:
    parquet_paths = _split_parquet_paths(split)
    paths = _prepacked_cache_paths(split, seq_len)
    temp_rows = paths["rows"].with_name(paths["rows"].name + ".tmp")
    temp_meta = paths["meta"].with_name(paths["meta"].name + ".tmp")
    row_capacity = seq_len + 1
    doc_iter = iter(_iter_cached_documents_once(parquet_paths))
    doc_buffer = _PackingBuffer()
    source_exhausted = False
    row_count = 0

    def refill_buffer() -> None:
        nonlocal source_exhausted
        while len(doc_buffer) < buffer_size and not source_exhausted:
            try:
                doc_buffer.add(next(doc_iter))
            except StopIteration:
                source_exhausted = True

    print(f"  Prepacking {split} seq_len={seq_len}")
    with temp_rows.open("wb") as handle:
        while True:
            refill_buffer()
            if not doc_buffer:
                break

            row = np.empty((row_capacity,), dtype=np.uint16)
            pos = 0
            while pos < row_capacity:
                refill_buffer()
                if not doc_buffer:
                    break

                remaining = row_capacity - pos
                doc = doc_buffer.pop_best_fit(remaining)
                if doc is not None:
                    row[pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    doc = doc_buffer.pop_shortest()
                    row[pos:pos + remaining] = doc[:remaining]
                    pos += remaining

            if pos == row_capacity:
                row.tofile(handle)
                row_count += 1

    meta = {
        "version": PREPACKED_CACHE_VERSION,
        "split": split,
        "seq_len": seq_len,
        "row_capacity": row_capacity,
        "buffer_size": buffer_size,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "bos_token_id": tokenizer.get_bos_token_id(),
        "source_parquets": [path.name for path in parquet_paths],
        "row_count": row_count,
        "dtype": "uint16",
    }
    temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    temp_rows.replace(paths["rows"])
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


class _CachedTokenBatchSource:
    def __init__(self, parquet_paths: list[Path], tokenizer_batch_size: int = 128):
        if not parquet_paths:
            raise RuntimeError("PackedDataLoader requires at least one parquet shard.")
        self.parquet_paths = parquet_paths
        self.tokenizer_batch_size = tokenizer_batch_size
        self.parquet_index = 0
        self.doc_index = 0
        self.epoch = 1
        self._offsets = None
        self._tokens = None
        self._doc_count = 0
        self._open_current_parquet()

    def _open_current_parquet(self) -> None:
        paths = _token_cache_paths(self.parquet_paths[self.parquet_index])
        self._offsets = np.load(paths["offsets"], mmap_mode="r")
        self._tokens = np.memmap(paths["tokens"], dtype=np.uint16, mode="r")
        self._doc_count = len(self._offsets) - 1

    def _advance_parquet(self) -> None:
        self.parquet_index += 1
        self.doc_index = 0
        if self.parquet_index >= len(self.parquet_paths):
            self.parquet_index = 0
            self.epoch += 1
        self._open_current_parquet()

    def next_batch(self):
        batch = []
        batch_epoch = self.epoch
        while True:
            while self.doc_index < self._doc_count and len(batch) < self.tokenizer_batch_size:
                start = int(self._offsets[self.doc_index])
                end = int(self._offsets[self.doc_index + 1])
                batch.append(self._tokens[start:end])
                self.doc_index += 1
            if batch:
                if self.doc_index >= self._doc_count:
                    self._advance_parquet()
                return batch, batch_epoch
            self._advance_parquet()
            batch_epoch = self.epoch

    def state_dict(self) -> dict:
        return {
            "kind": "cached_token",
            "parquet_index": self.parquet_index,
            "doc_index": self.doc_index,
            "epoch": self.epoch,
            "tokenizer_batch_size": self.tokenizer_batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        self.parquet_index = int(state["parquet_index"])
        self.doc_index = int(state["doc_index"])
        self.epoch = int(state["epoch"])
        self._open_current_parquet()


class _TextTokenBatchSource:
    def __init__(self, parquet_paths: list[Path], tokenizer: Tokenizer, tokenizer_batch_size: int = 128):
        if not parquet_paths:
            raise RuntimeError("PackedDataLoader requires at least one parquet shard.")
        self.parquet_paths = parquet_paths
        self.tokenizer = tokenizer
        self.tokenizer_batch_size = tokenizer_batch_size
        self.bos_token = tokenizer.get_bos_token_id()
        self.parquet_index = 0
        self.row_group_index = 0
        self.row_offset = 0
        self.epoch = 1

    def _current_file(self) -> pq.ParquetFile:
        return pq.ParquetFile(self.parquet_paths[self.parquet_index])

    def _advance_parquet(self) -> None:
        self.parquet_index += 1
        self.row_group_index = 0
        self.row_offset = 0
        if self.parquet_index >= len(self.parquet_paths):
            self.parquet_index = 0
            self.epoch += 1

    def next_batch(self):
        while True:
            parquet_file = self._current_file()
            if self.row_group_index >= parquet_file.num_row_groups:
                self._advance_parquet()
                continue

            texts = parquet_file.read_row_group(self.row_group_index).column("text").to_pylist()
            if self.row_offset >= len(texts):
                self.row_group_index += 1
                self.row_offset = 0
                continue

            batch_epoch = self.epoch
            end = min(self.row_offset + self.tokenizer_batch_size, len(texts))
            text_batch = texts[self.row_offset:end]
            self.row_offset = end
            if self.row_offset >= len(texts):
                self.row_group_index += 1
                self.row_offset = 0
                if self.row_group_index >= parquet_file.num_row_groups:
                    self._advance_parquet()
            return self.tokenizer.encode(text_batch, prepend=self.bos_token), batch_epoch

    def state_dict(self) -> dict:
        return {
            "kind": "text_tokenized",
            "parquet_index": self.parquet_index,
            "row_group_index": self.row_group_index,
            "row_offset": self.row_offset,
            "epoch": self.epoch,
            "tokenizer_batch_size": self.tokenizer_batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        self.parquet_index = int(state["parquet_index"])
        self.row_group_index = int(state["row_group_index"])
        self.row_offset = int(state["row_offset"])
        self.epoch = int(state["epoch"])


def _make_document_batch_source(
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
        return _CachedTokenBatchSource(parquet_paths, tokenizer_batch_size=tokenizer_batch_size)
    return _TextTokenBatchSource(parquet_paths, tokenizer, tokenizer_batch_size=tokenizer_batch_size)


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
    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int,
        seq_len: int,
        split: str,
        buffer_size: int = 1000,
        *,
        prepacked_requested: bool = False,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.row_capacity = seq_len + 1
        self.buffer_size = buffer_size
        self.using_token_cache = _split_has_token_cache(split, tokenizer)
        if prepacked_requested:
            if self.using_token_cache:
                source = f"token cache (prepacked cache missing for seq_len={seq_len})"
            else:
                source = f"parquet + tokenizer fallback (prepacked cache missing for seq_len={seq_len})"
        else:
            source = "token cache" if self.using_token_cache else "parquet + tokenizer fallback"
        print(f"Data loader ({split}): using {source}.")
        self.batch_source = _make_document_batch_source(split, tokenizer, use_cache=self.using_token_cache)
        self.doc_buffer = _PackingBuffer()
        self.epoch = 1
        self.row_buffer = np.empty((batch_size, self.row_capacity), dtype=np.int32)
        self.inputs = np.empty((batch_size, seq_len), dtype=np.int32)
        self.targets = np.empty((batch_size, seq_len), dtype=np.int32)

    def __iter__(self) -> "PackedDataLoader":
        return self

    def refill_buffer(self) -> None:
        token_batch, self.epoch = self.batch_source.next_batch()
        self.doc_buffer.extend(token_batch)

    def checkpoint_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        doc_buffer_tokens, doc_buffer_offsets = _pack_documents(self.doc_buffer.documents_for_checkpoint())
        return (
            {
                "loader_type": "packed",
                "split": self.split,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "buffer_size": self.buffer_size,
                "using_token_cache": self.using_token_cache,
                "epoch": self.epoch,
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
                "batch_source": self.batch_source.state_dict(),
            },
            {
                "doc_buffer_tokens": doc_buffer_tokens,
                "doc_buffer_offsets": doc_buffer_offsets,
            },
        )

    def load_checkpoint_state(self, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
        self.epoch = int(metadata["epoch"])
        self.doc_buffer.load_documents(_unpack_documents(arrays["doc_buffer_tokens"], arrays["doc_buffer_offsets"]))
        self.batch_source.load_state_dict(metadata["batch_source"])

    def __next__(self):
        for row_idx in range(self.batch_size):
            pos = 0
            while pos < self.row_capacity:
                while len(self.doc_buffer) < self.buffer_size:
                    self.refill_buffer()

                remaining = self.row_capacity - pos
                doc = self.doc_buffer.pop_best_fit(remaining)
                if doc is not None:
                    self.row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    doc = self.doc_buffer.pop_shortest()
                    self.row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        self.inputs[...] = self.row_buffer[:, :-1]
        self.targets[...] = self.row_buffer[:, 1:]
        return mx.array(self.inputs), mx.array(self.targets), self.epoch


class PrepackedDataLoader:
    def __init__(self, tokenizer: Tokenizer, batch_size: int, seq_len: int, split: str):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.row_capacity = seq_len + 1
        self.epoch = 1
        self.row_index = 0

        paths = _prepacked_cache_paths(split, seq_len)
        meta = _load_prepacked_cache_meta(split, seq_len)
        if meta is None:
            raise RuntimeError(f"Missing prepacked cache metadata for {split} seq_len={seq_len}.")

        self.row_count = int(meta["row_count"])
        if self.row_count <= 0:
            raise RuntimeError(f"Invalid prepacked cache row count for {split} seq_len={seq_len}.")

        self.rows = np.memmap(
            paths["rows"],
            dtype=np.uint16,
            mode="r",
            shape=(self.row_count, self.row_capacity),
        )
        self._batch_offsets = np.arange(batch_size, dtype=np.int64)
        self.inputs = np.empty((batch_size, seq_len), dtype=np.int32)
        self.targets = np.empty((batch_size, seq_len), dtype=np.int32)
        print(f"Data loader ({split}): using prepacked cache.")

    def __iter__(self) -> "PrepackedDataLoader":
        return self

    def __next__(self):
        for row_idx in range(self.batch_size):
            self.inputs[row_idx, :] = self.rows[self.row_index, :-1]
            self.targets[row_idx, :] = self.rows[self.row_index, 1:]
            self.row_index += 1
            if self.row_index >= self.row_count:
                self.row_index = 0
                self.epoch += 1
        return mx.array(self.inputs), mx.array(self.targets), self.epoch

    def batch_at_row_index(self, row_index: int):
        indices = (row_index + self._batch_offsets) % self.row_count
        batch_rows = self.rows[indices]
        self.inputs[...] = batch_rows[:, :-1]
        self.targets[...] = batch_rows[:, 1:]
        return mx.array(self.inputs), mx.array(self.targets)

    def checkpoint_state(self) -> tuple[dict, dict[str, np.ndarray]]:
        return (
            {
                "loader_type": "prepacked",
                "split": self.split,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "row_index": self.row_index,
                "epoch": self.epoch,
                "tokenizer_fingerprint": self.tokenizer.fingerprint,
            },
            {},
        )

    def load_checkpoint_state(self, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
        self.row_index = int(metadata["row_index"])
        self.epoch = int(metadata["epoch"])


def _pack_documents(documents: list[np.ndarray | list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    chunks = []
    for document in documents:
        array = np.asarray(document, dtype=np.uint16)
        chunks.append(array)
        offsets.append(offsets[-1] + len(array))
    flat = np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.uint16)
    return flat, np.asarray(offsets, dtype=np.int64)


def _unpack_documents(tokens: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [
        np.array(tokens[int(offsets[idx]):int(offsets[idx + 1])], dtype=np.uint16, copy=True)
        for idx in range(len(offsets) - 1)
    ]


class _PackingBuffer:
    def __init__(self):
        # Bucket by document length so best-fit and shortest-pop stay sublinear
        # while preserving FIFO tie-breaking within a given length.
        self._buckets: dict[int, deque[np.ndarray | list[int]]] = {}
        self._lengths: list[int] = []
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, doc: np.ndarray | list[int]) -> None:
        doc_len = len(doc)
        bucket = self._buckets.get(doc_len)
        if bucket is None:
            bucket = deque()
            self._buckets[doc_len] = bucket
            bisect.insort(self._lengths, doc_len)
        bucket.append(doc)
        self._size += 1

    def extend(self, docs) -> None:
        for doc in docs:
            self.add(doc)

    def load_documents(self, docs: list[np.ndarray | list[int]]) -> None:
        self._buckets.clear()
        self._lengths.clear()
        self._size = 0
        self.extend(docs)

    def documents_for_checkpoint(self) -> list[np.ndarray | list[int]]:
        documents: list[np.ndarray | list[int]] = []
        for doc_len in self._lengths:
            documents.extend(self._buckets[doc_len])
        return documents

    def pop_best_fit(self, max_len: int):
        idx = bisect.bisect_right(self._lengths, max_len) - 1
        if idx < 0:
            return None
        return self._pop_length(self._lengths[idx])

    def pop_shortest(self):
        if not self._lengths:
            raise IndexError("Packing buffer is empty.")
        return self._pop_length(self._lengths[0])

    def _pop_length(self, doc_len: int):
        bucket = self._buckets[doc_len]
        doc = bucket.popleft()
        self._size -= 1
        if not bucket:
            del self._buckets[doc_len]
            idx = bisect.bisect_left(self._lengths, doc_len)
            del self._lengths[idx]
        return doc


def serialize_loader_state(loader) -> tuple[dict, dict[str, np.ndarray]]:
    return loader.checkpoint_state()


def restore_loader_state(loader, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    expected = {
        "loader_type": "prepacked" if isinstance(loader, PrepackedDataLoader) else "packed",
        "split": loader.split,
        "batch_size": loader.batch_size,
        "seq_len": loader.seq_len,
        "tokenizer_fingerprint": loader.tokenizer.fingerprint,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint loader metadata mismatch for {key}: expected {value!r}, got {metadata.get(key)!r}")
    loader.load_checkpoint_state(metadata, arrays)


def make_dataloader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    buffer_size: int = 1000,
    *,
    prefer_prepacked_cache: bool = True,
):
    if split not in {"train", "val"}:
        raise ValueError(f"Invalid split: {split}")
    has_prepacked_cache = _split_has_prepacked_cache(split, seq_len, tokenizer)
    if prefer_prepacked_cache and has_prepacked_cache:
        return PrepackedDataLoader(tokenizer, batch_size, seq_len, split)
    return PackedDataLoader(
        tokenizer,
        batch_size,
        seq_len,
        split,
        buffer_size=buffer_size,
        prepacked_requested=prefer_prepacked_cache and not has_prepacked_cache,
    )


@mx.compile
def _masked_token_sums(loss_flat, target_ids, token_bytes):
    nbytes = token_bytes[target_ids]
    mask = nbytes > 0
    return mx.sum(loss_flat * mask.astype(loss_flat.dtype)), mx.sum(nbytes.astype(mx.int64))


def _iter_eval_batches(loader, steps: int, eval_slices: int, reference_steps: int | None = None):
    if isinstance(loader, PrepackedDataLoader) and eval_slices > 1:
        available_batches = max(1, loader.row_count // loader.batch_size)
        horizon_batches = available_batches
        if reference_steps is not None:
            # For cheap canonical evals, spread slices across the same prefix
            # horizon that the full upstream contract would traverse.
            horizon_batches = max(1, min(max(steps, reference_steps), available_batches))
        slice_count = max(1, min(eval_slices, max(1, steps // EVAL_SLICE_TARGET_STEPS), horizon_batches))
        for slice_idx in range(slice_count):
            slice_steps = steps // slice_count + (1 if slice_idx < steps % slice_count else 0)
            if slice_steps <= 0:
                continue
            row_index = ((slice_idx * horizon_batches) // slice_count) * loader.batch_size
            for _ in range(slice_steps):
                yield loader.batch_at_row_index(row_index)
                row_index = (row_index + loader.batch_size) % loader.row_count
        return

    for _ in range(steps):
        x, y, _ = next(loader)
        yield x, y


def evaluate_bpb(
    model,
    tokenizer: Tokenizer,
    batch_size: int,
    *,
    seq_len: int = MAX_SEQ_LEN,
    eval_tokens: int = EVAL_TOKENS,
    prefer_prepacked_cache: bool = True,
    eval_slices: int = 1,
    reference_eval_tokens: int | None = None,
) -> float:
    token_bytes = mx.array(load_token_bytes())
    val_loader = make_dataloader(
        tokenizer,
        batch_size,
        seq_len,
        "val",
        prefer_prepacked_cache=prefer_prepacked_cache,
    )
    steps = max(1, eval_tokens // (batch_size * seq_len))
    reference_steps = None
    if reference_eval_tokens is not None:
        reference_steps = max(1, reference_eval_tokens // (batch_size * seq_len))
    total_nats = 0.0
    total_bytes = 0

    for x, y in _iter_eval_batches(val_loader, steps, eval_slices, reference_steps=reference_steps):
        loss_flat = model(x, y, reduction="none").reshape((-1,))
        y_flat = y.reshape((-1,))
        batch_nats, batch_bytes = _masked_token_sums(loss_flat, y_flat, token_bytes)
        mx.eval(batch_nats, batch_bytes)
        total_nats += batch_nats.item()
        total_bytes += batch_bytes.item()

    return total_nats / (math.log(2) * total_bytes)
