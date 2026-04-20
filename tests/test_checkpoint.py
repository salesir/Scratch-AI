import torch
import os   
from model.model import TransformerLM

def test_checkpoint_roundtrip(tmp_path):
    vocab_size = 50
    block_size = 8

    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=32,
        num_heads=2,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_file = tmp_path / "ckpt.pth"

    torch.save({
        "step": 123,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss_history": [(0, 2.3)]
    }, checkpoint_file)

    new_model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=32,
        num_heads=2,
        num_layers=2
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    ckpt = torch.load(checkpoint_file)
    new_model.load_state_dict(ckpt["model_state"])
    new_optimizer.load_state_dict(ckpt["optimizer_state"])

    assert ckpt["step"] == 123
