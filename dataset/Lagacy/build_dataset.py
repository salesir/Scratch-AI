# dataset/build_dataset.py
#lagacy! works with dataloader and .bin files to train off .txt files
import os
import struct
import re
from pathlib import Path
from tokenizer.bpe import BPETokenizer
import spacy
from concurrent.futures import ProcessPoolExecutor

# ------------------------------
# Paths and constants
# ------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FILE = "train.bin"
CONTEXT = 1024
NUM_WORKERS = 4  # Adjust based on your CPU cores

# ------------------------------
# Tokenizer
# ------------------------------
tokenizer = BPETokenizer(
    ROOT / "tokenizer" / "vocab.json",
    ROOT / "tokenizer" / "merges.txt"
)

# ------------------------------
# Load spaCy for entity recognition
# ------------------------------
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 50000000  # or higher if needed

# ------------------------------
# Text abstraction function
# ------------------------------
def abstract_text(text: str) -> str:
    text = re.sub(r'"[^"]*"', '', text)
    doc = nlp(text)
    abstracted_tokens = []
    for token in doc:
        if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART"}:
            abstracted_tokens.append("<ENTITY>")
        else:
            abstracted_tokens.append(token.text)
    abstracted_text = " ".join(abstracted_tokens)
    return re.sub(r"\s+", " ", abstracted_text).strip()

# ------------------------------
# Tone amplifier
# ------------------------------
def amplify_tone(text: str) -> str:
    replacements = {
        "powerful": "omnipotent",
        "strong": "unyielding",
        "control": "domination",
        "struggle": "torment",
        "fear": "dread",
        "suffer": "endure agony",
        "pain": "torment",
        "death": "obliteration"
    }
    for k, v in replacements.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text

# ------------------------------
# Encode function for multiprocessing
# ------------------------------
def process_file(fname):
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Step 1: Abstract entities
    text = abstract_text(text)

    # Step 2: Amplify tone
    text = amplify_tone(text)

    # Step 3: Tokenize
    return tokenizer.encode(text)

# ------------------------------
# Write tokens in CONTEXT-sized chunks
# ------------------------------
def write_buffer(out, token_list):
    buffer = []
    for t in token_list:
        buffer.append(t)
        if len(buffer) == CONTEXT + 1:
            for i, target in zip(buffer[:-1], buffer[1:]):
                out.write(struct.pack("II", i, target))
            buffer = []
    return buffer  # return leftover tokens

# ------------------------------
# Build dataset
# ------------------------------
def build():
    token_buffer = []

    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    with open(OUT_FILE, "wb") as out, ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Map files to processes
        for token_list in executor.map(process_file, txt_files):
            token_buffer = write_buffer(out, token_list + token_buffer)  # include leftover from previous
        # Write any remaining tokens
        if len(token_buffer) > 1:
            for i, target in zip(token_buffer[:-1], token_buffer[1:]):
                out.write(struct.pack("II", i, target))

    print(f"Dataset built and saved to {OUT_FILE}")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    build()
