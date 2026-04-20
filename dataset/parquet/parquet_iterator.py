# dataset/parquet/parquet_iterator.py
import os
import pyarrow.parquet as pq
import glob

class ParquetTextIterator:
    def __init__(self, parquet_path):
        # Find all parquet files dynamically
        self.files = glob.glob(os.path.join(parquet_path, "*.parquet"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No parquet files found in {parquet_path}")
        self.file_idx = 0
        self.row_group_idx = 0
        self.current_file = None
        self._open_file()

    def _open_file(self):
        self.current_file = pq.ParquetFile(self.files[self.file_idx])
        self.num_row_groups = self.current_file.num_row_groups
        self.row_group_idx = 0

    def reset(self):
        self.file_idx = 0
        self.row_group_idx = 0
        self._open_file()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.row_group_idx >= self.num_row_groups:
                self.file_idx += 1
                if self.file_idx >= len(self.files):
                    raise StopIteration
                self._open_file()

            batch = self.current_file.read_row_group(self.row_group_idx)
            self.row_group_idx += 1
            if batch.num_rows == 0:
                continue  # skip empty row groups
            return batch
