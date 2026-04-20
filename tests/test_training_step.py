import torch
import torch.nn as nn
from model.model import TransformerLM

def test_training_step_updates_weights():
    vocab_size = 50
    block_size = 8
    batch_size = 4

    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=32,
        num_heads=2,
        num_layers=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))

    # Save initial weights
    before = model.head.weight.clone().detach()

    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = model.head.weight.detach()

    assert not torch.allclose(before, after)
    