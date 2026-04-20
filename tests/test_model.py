import torch
from model.model import TransformerLM

def test_forward_shape():
    vocab_size = 100
    block_size = 16
    d_model = 32
    num_heads = 4
    num_layers = 2

    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    batch_size = 4
    idx = torch.randint(0, vocab_size, (batch_size, block_size))

    logits = model(idx)

    assert logits.shape == (batch_size, block_size, vocab_size)
    #############################################################################
def test_no_nan_logits():
    vocab_size = 50
    block_size = 8
    d_model = 32
    num_heads = 4
    num_layers = 2

    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    idx = torch.randint(0, vocab_size, (1, block_size))
    logits = model(idx)

    assert torch.isfinite(logits).all()
##############################################################################
def test_backward_pass():
    vocab_size = 100
    block_size = 8
    d_model = 32
    num_heads = 4
    num_layers = 2

    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    idx = torch.randint(0, vocab_size, (2, block_size))
    targets = torch.randint(0, vocab_size, (2, block_size))

    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    loss.backward()

    # -------------------------------
    # Gradient existence check (must be inside function)
    grads_exist = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.parameters()
    )
    assert grads_exist

print("Model is functioning correctly. Yippie!")
