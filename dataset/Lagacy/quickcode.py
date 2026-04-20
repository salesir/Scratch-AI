from pathlib import Path
import pyarrow.parquet as pq

data_folder = Path(r"E:\Local_AI_Scratch\Data")
files = list(data_folder.glob("*.parquet"))
print("Files found:", files)

for f in files:
    try:
        pf = pq.ParquetFile(f)
        print(f"{f.name} -> {pf.schema.names}")
    except Exception as e:
        print(f"Error reading {f.name}: {e}")
