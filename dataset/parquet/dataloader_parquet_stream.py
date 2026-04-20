# dataset/streaming/dataloader_parquet_stream.py

import torch
import random
import multiprocessing as mp
from collections import deque

from .parquet_iterator import ParquetTextIterator


# ==============================
# BACKGROUND PREFETCH WORKER (ZERO-COPY SLIDING WINDOW)
# ==============================

def prefetch_worker(
    path,
    tokenizer,
    block_size,
    queue,
    max_queue_blocks,
):
    """
    Background process:
    - Reads parquet dynamically
    - Tokenizes lazily
    - Pushes sliding windows into queue
    """

    iterator = ParquetTextIterator(path)
    token_buffer = deque()  # rolling buffer for tokens

    while True:
        try:
            text = next(iterator)
        except StopIteration:
            break

        tokens = tokenizer.encode(text)
        token_buffer.extend(tokens)

        # Sliding window (zero-copy approach)
        while len(token_buffer) >= block_size + 1:
            # Only copy once per block for queue push
            block = list(token_buffer)[: block_size + 1]
            token_buffer.popleft()  # O(1)

            # Backpressure: blocking put
            queue.put(block)

            if queue.qsize() >= max_queue_blocks:
                # Could sleep here or just let queue.put() block
                pass

    # Signal end
    queue.put(None)


# ==============================
# DATASET LOADER (TRAINING SIDE)
# ==============================

class DatasetLoader:  # for parquet
    """
    TRUE streaming parquet dataloader.

    ✔ bounded memory
    ✔ background prefetch
    ✔ dynamic parquet reading
    ✔ training-loop compatible
    """

    def __init__(
        self,
        path,
        tokenizer,
        block_size,
        batch_size,
        device,
        shuffle=True,
        seed=1337,
        max_queue_blocks=4096,
        local_shuffle_buffer=1024,
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        self.queue = mp.Queue(maxsize=max_queue_blocks)
        self.buffer = deque(maxlen=local_shuffle_buffer)

        # Start background prefetch process
        self.prefetch_proc = mp.Process(
            target=prefetch_worker,
            args=(path, tokenizer, block_size, self.queue, max_queue_blocks),
            daemon=True,
        )
        self.prefetch_proc.start()

    # --------------------------
    # INTERNAL BUFFER FILL
    # --------------------------
    def _fill_buffer(self):
        while len(self.buffer) < self.buffer.maxlen:
            item = self.queue.get()
            if item is None:
                break
            self.buffer.append(item)

    # --------------------------
    # PUBLIC API (TRAIN LOOP)
    # --------------------------
    def get_batch(self):
        self._fill_buffer()

        if len(self.buffer) < self.batch_size:
            raise RuntimeError("Streaming dataset exhausted.")

        # Sample batch
        if self.shuffle:
            batch = self.rng.sample(self.buffer, self.batch_size)
        else:
            batch = [self.buffer.popleft() for _ in range(self.batch_size)]

        x = torch.tensor(
            [b[:-1] for b in batch],
            dtype=torch.long,
            device=self.device,
            pin_memory=True,
        )
        y = torch.tensor(
            [b[1:] for b in batch],
            dtype=torch.long,
            device=self.device,
            pin_memory=True,
        )

        return x, y
