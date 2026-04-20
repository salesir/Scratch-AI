# tokenizer/bpe.py
import json

class BPETokenizer:
    def __init__(self, vocab_path, merges_path):
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []

        self._load_vocab(vocab_path)
        self._load_merges(merges_path)

    def _load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        for hex_str, idx in raw_vocab.items():
            token = bytes.fromhex(hex_str)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def _load_merges(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                a_hex, b_hex = line.strip().split()
                self.merges.append(
                    (bytes.fromhex(a_hex), bytes.fromhex(b_hex))
                )

    def encode(self, text: str):
        tokens = [bytes([b]) for b in text.encode("utf-8")]

        for a, b in self.merges:
            i = 0
            merged = []
            ab = a + b
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged.append(ab)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        return [self.token_to_id[t] for t in tokens]
    def decode(self, token_ids):
        byte_stream = b"".join(self.id_to_token[i] for i in token_ids)
        return byte_stream.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return len(self.token_to_id)
