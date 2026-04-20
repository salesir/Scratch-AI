import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
#does not quite work yet due to the complexity of parquet, Lagacy for now but might return to it later
# ------------------------
# Paths
# ------------------------
DATA_DIR = Path("E:/Local_AI_Scratch/data")       # folder with your original parquet files
OUT_DIR = DATA_DIR / "small_split"               # folder where the smaller files will go, it will create it automatically if not present
OUT_DIR.mkdir(exist_ok=True)

# ------------------------
# Parameters
# ------------------------
rows_per_file = 50000  # number of rows per small parquet file

# ------------------------
# Process files
# ------------------------
for parquet_file in DATA_DIR.glob("*.parquet"):
    print(f"Reading {parquet_file.name}...")
    df = pd.read_parquet(parquet_file, columns=["token_id"])  # <-- only what we need
    df["token_id"] = df["token_id"].astype("uint16")          # <-- smaller dtype

    base_name = parquet_file.stem
    for i in range(0, len(df), rows_per_file):
        chunk = df.iloc[i:i + rows_per_file]
        out_path = OUT_DIR / f"{base_name}_part{i//rows_per_file}.parquet"
        pq.write_table(pa.Table.from_pandas(chunk), out_path, compression="snappy")

    print(f"Done: {parquet_file.name} -> {len(df)//rows_per_file + 1} chunks created")

print(f"All done! Small parquet files are in {OUT_DIR}")
