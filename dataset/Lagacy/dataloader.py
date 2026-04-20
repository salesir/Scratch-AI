import numpy as np
import torch
#considered lagacy, transforms .txt files into .bin files for dataloader
class DatasetLoader:
    def __init__(self, path="dataset/train.bin", block_size=128, batch_size=16, shuffle=True, device="cpu"):
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.data = np.fromfile(path, dtype=np.uint16)
        self.num_tokens = len(self.data)

        self.num_sequences = self.num_tokens // block_size
        self.sequences = self.data[:self.num_sequences * block_size].reshape(
            self.num_sequences, block_size
        )

        self.indices = np.arange(self.num_sequences)
        self.ptr = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.ptr = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.num_sequences:
            raise StopIteration

        batch_idx = self.indices[self.ptr:self.ptr + self.batch_size]
        batch = self.sequences[batch_idx]

        self.ptr += self.batch_size

        return (
            torch.tensor(batch[:, :-1], dtype=torch.long, device=self.device),
            torch.tensor(batch[:, 1:], dtype=torch.long, device=self.device),
        )

    def get_batch(self):
        try:
            return next(self)
        except StopIteration:
            # 🔥 RESET AND CONTINUE (NO ESCAPE)
            self.ptr = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            return next(self)
