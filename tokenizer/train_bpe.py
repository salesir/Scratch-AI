# tokenizer/train_bpe.py
import os
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

VOCAB_SIZE = 32000

def text_to_tokens(text):
    return [bytes([b]) for b in text.encode("utf-8")]

def load_corpus():
    sequences = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                text = f.read()
                sequences.append(text_to_tokens(text))
    return sequences

def count_pairs(sequences):
    counts = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i+1])] += 1
    return counts

def merge_pair(sequences, pair, new_token):
    out = []
    for seq in sequences:
        i = 0
        merged = []
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                merged.append(new_token)
                i += 2
            else:
                merged.append(seq[i])
                i += 1
        out.append(merged)
    return out

def train_bpe():
    sequences = load_corpus()

    vocab = {bytes([i]): i for i in range(256)}
    merges = []

    while len(vocab) < VOCAB_SIZE:
        pair_counts = count_pairs(sequences)
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=pair_counts.get)
        new_token = best_pair[0] + best_pair[1]

        vocab[new_token] = len(vocab)
        merges.append(best_pair)

        sequences = merge_pair(sequences, best_pair, new_token)

        if len(vocab) % 1000 == 0:
            print(f"Vocab size: {len(vocab)}")

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({k.hex(): v for k, v in vocab.items()}, f, indent=2)

    with open("merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

if __name__ == "__main__":
    train_bpe()
