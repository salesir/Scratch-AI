# dataset/dataloader.py
import torch
from dataset.parquet.parquet_iterator import ParquetTextIterator

# Conservative streaming buffer sizes
TOKEN_BUFFER_TARGET = 100_000
TOKEN_BUFFER_MIN = 20_000
MAX_FILL_ATTEMPTS = 1000  # max iterations to try filling buffer

class DatasetLoader:
    def __init__(
        self,
        path,
        tokenizer,
        block_size=128,
        batch_size=16,
        device="cpu",
        shuffle=True
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.token_buffer = []
        self.iterator = ParquetTextIterator(self.path)

    def _fill_token_buffer(self):
        if len(self.token_buffer) >= TOKEN_BUFFER_MIN:
            return

        attempts = 0
        while len(self.token_buffer) < TOKEN_BUFFER_TARGET:
            attempts += 1
            if attempts > MAX_FILL_ATTEMPTS:
                # stop trying further to avoid infinite loops
                break

            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator.reset()
                continue

            for col_name in batch.column_names:
                column = batch[col_name]
                if column.num_chunks == 0:
                    continue

                for text in column.to_pylist():
                    if not isinstance(text, str):
                        continue

                    text = text.strip()
                    if not text:
                        continue

                    # Hard cap extremely long documents
                    text = text[:10_000]

                    try:
                        tokens = self.tokenizer.encode(text)
                    except Exception:
                        continue

                    self.token_buffer.extend(tokens)

        if len(self.token_buffer) < TOKEN_BUFFER_MIN:
            raise RuntimeError("Unable to fill token buffer from parquet files.")

    def get_batch(self):
        self._fill_token_buffer()

        needed = self.batch_size * (self.block_size + 1)
        if len(self.token_buffer) < needed:
            raise RuntimeError(
                f"Not enough tokens to form a batch "
                f"(have {len(self.token_buffer)}, need {needed})"
            )

        batch_tokens = self.token_buffer[:needed]
        self.token_buffer = self.token_buffer[needed:]

        tokens = torch.tensor(
            batch_tokens,
            dtype=torch.long,
            device=self.device
        ).view(self.batch_size, self.block_size + 1)

        x = tokens[:, :-1]
        y = tokens[:, 1:]

        return x, y
