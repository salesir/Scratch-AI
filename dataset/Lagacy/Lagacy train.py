#this is lagacy code for pre-processing
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import pyarrow.parquet as pq

from tokenizer.bpe import BPETokenizer
from model.model import TransformerLM
# This now imports the correctly selected loader (Streaming or Legacy)
from dataset.dataloader import DatasetLoader

# ------------------------------
# Multiprocessing
# ------------------------------
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  

    # ------------------------------
    # Hyperparameters
    # ------------------------------
    tokenizer = BPETokenizer(
        "tokenizer/vocab.json",
        "tokenizer/merges.txt"
    )

    block_size = 128
    batch_size = 16
    learning_rate = 1e-3
    num_steps = 10000   # total training steps, increase for more training per run

    vocab_size = tokenizer.vocab_size 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = "model/checkpoint.pth"
    save_every = 50

    # ------------------------------
    # Reproducibility
    # ------------------------------
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ------------------------------
    # Initialize model
    # ------------------------------
    model = TransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=128,
        num_heads=2,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # ------------------------------
    # Load checkpoint if exists
    # ------------------------------
    start_step = 0
    loss_history = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint.get("step", 0)
        loss_history = checkpoint.get("loss_history", [])
        print(f"Resuming training from step {start_step}")

    # ------------------------------
    # Dataset loader
    # ------------------------------
    data_path = "Data"

    dataset = DatasetLoader(
        path=data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        shuffle=True,
    )

    model.train()

    # ------------------------------
    # Training loop
    # ------------------------------
    step = start_step

    while step < num_steps:
        inputs, targets = dataset.get_batch()

        # --------------------------
        # Safety checks
        # --------------------------
        assert inputs.shape == targets.shape
        assert inputs.dim() == 2
        assert inputs.min() >= 0
        assert inputs.max() < vocab_size

        optimizer.zero_grad()

        logits = model(inputs)
        assert torch.isfinite(logits).all(), "NaN/Inf in logits"

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --------------------------
        # Logging
        # --------------------------
        if step % 10 == 0:
            print(f"Step {step:5d} | Loss {loss.item():.4f}")
            loss_history.append((step, loss.item()))

        # --------------------------
        # Checkpoint
        # --------------------------
        if step > 0 and step % save_every == 0:
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss_history": loss_history,
                    "config": {
                        "vocab_size": vocab_size,
                        "block_size": block_size,
                        "d_model": 128,
                        "num_heads": 2,
                        "num_layers": 2,
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved @ step {step}")

        step += 1

    # ------------------------------
    # Plot training loss
    # ------------------------------
    if loss_history:
        steps, losses = zip(*loss_history)
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses)
        plt.xlabel("Training Step")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.show()