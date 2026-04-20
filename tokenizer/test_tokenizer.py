if __name__ == "__main__":
    tok = BPETokenizer("vocab.json", "merges.txt")

    s = "Hello 🌍! This is a test.\nNew line."
    ids = tok.encode(s)
    out = tok.decode(ids)

    assert s == out
    print("OK")
