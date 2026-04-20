import os
import random
import torch
import pyarrow.parquet as pq
import time
import psutil

#lagacy! this is the pre-processing dataloader for parquet files, kept for reference. superior to streaming.
#however not realistic for personal hardware
def extract_text(obj):
    """
    Recursively extract text from arbitrary parquet objects.
    main issue I think
    Handles:
    - strings
    - dicts with 'content'
    - lists of dicts
    """
    texts = []

    if obj is None:
        return texts

    if isinstance(obj, str):
        if obj.strip():
            texts.append(obj)
        return texts

    if isinstance(obj, dict):
        if "content" in obj and isinstance(obj["content"], str):
            texts.append(obj["content"])
        for v in obj.values():
            texts.extend(extract_text(v))
        return texts

    if isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))
        return texts

    return texts


class DatasetLoaderParquet:
    """
    Parquet dataset loader with automatic text-column detection.
    
    - Works with LMSYS-Chat-1M
    - Supports flat chat datasets
    """

    def __init__(
        self,
        path,
        tokenizer,
        block_size,
        batch_size,
        device,
        text_column=None,   # ← OPTIONAL
        shuffle=True,
        seed=1337,
        log_interval_sec=60,
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        self.process = psutil.Process(os.getpid())
        self.log_interval_sec = log_interval_sec
        self._last_log_time = time.time()

        # Resolve parquet files
        if os.path.isdir(path):
            parquet_files = [
                os.path.join(path, f)
                for f in sorted(os.listdir(path))
                if f.endswith(".parquet")
            ]
        else:
            parquet_files = [path]

        if not parquet_files:
            raise RuntimeError("No parquet files found.")

        print(f"[ParquetDataset] Found {len(parquet_files)} parquet files")

        self.blocks = []
        start_time = time.time()

        for parquet_path in parquet_files:
            print(f"[ParquetDataset] Reading {parquet_path}")
            pf = pq.ParquetFile(parquet_path)

            # Read schema once
            schema = pf.schema_arrow
            column_names = schema.names

            print(f"[ParquetDataset] Columns found: {column_names}")

            # ------------------------------
            # Auto-detect text column
            # ------------------------------
            if text_column is None:
                preferred = ["content", "text", "message", "prompt", "completion"]
                candidates = [c for c in preferred if c in column_names]

                if not candidates:
                    raise RuntimeError(
                        "Could not auto-detect text column. "
                        "Please pass text_column explicitly."
                    )

                text_column = candidates[0]
                print(f"[ParquetDataset] Using column: {text_column}")

            elif text_column not in column_names:
                raise RuntimeError(
                    f"Requested text_column '{text_column}' not found. "
                    f"Available columns: {column_names}"
                )

            # ------------------------------
            # Load data
            # ------------------------------
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=[text_column])
                texts = table[text_column].to_pylist()

                for text in texts:
                    if not isinstance(text, str) or not text.strip():
                        continue

                    tokens = tokenizer.encode(text)

                    if len(tokens) <= block_size:
                        continue

                    # Sliding window (document-local)
                    for i in range(0, len(tokens) - block_size):
                        block = tokens[i : i + block_size + 1]
                        if len(block) == block_size + 1:
                            self.blocks.append(block)

                self._maybe_log()

        if not self.blocks:
            raise RuntimeError("No training blocks were created.")

        if self.shuffle:
            self.rng.shuffle(self.blocks)

        self.idx = 0

        elapsed = time.time() - start_time
        print(
            f"[ParquetDataset] Loaded {len(self.blocks):,} blocks "
            f"in {elapsed:.1f}s"
        )
        self._log_stats(force=True)
