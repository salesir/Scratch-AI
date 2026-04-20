import torch
from tokenizer.bpe import BPETokenizer
from model.model import TransformerLM

@torch.no_grad()
def generate_sample(model, idx, max_new_tokens, block_size, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


if __name__ == "__main__":
    device = "cpu"
    checkpoint_path = "model/checkpoint.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"]

    vocab_size, d_model = state["token_emb.weight"].shape
    block_size = state["pos_emb.weight"].shape[0]

    num_layers = len({
        k.split(".")[1]
        for k in state.keys()
        if k.startswith("blocks.")
    })

    num_heads = 4  # MUST match training

    print("[INFO] Inferred model config:")
    print(f" vocab_size = {vocab_size}")
    print(f" d_model    = {d_model}")
    print(f" block_size = {block_size}")
    print(f" layers     = {num_layers}")
    print(f" heads      = {num_heads}")

    tokenizer = BPETokenizer(
        vocab_path="tokenizer/vocab.json",
        merges_path="tokenizer/merges.txt"
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(state)

    prompt = "introduce yourself"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    out = generate_sample(
        model=model,
        idx=idx,
        max_new_tokens=200,
        block_size=block_size,
        temperature=1.0
    )

    print("\n=== Generated Text ===\n")
    print(tokenizer.decode(out[0].tolist()))
