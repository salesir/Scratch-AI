from dataset.parquet.parquet_iterator import ParquetTextIterator

it = ParquetTextIterator("Data")
print(next(it))
