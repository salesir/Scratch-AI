import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import threading
import queue
import math

from tokenizer.bpe import BPETokenizer
from model.model import TransformerLM
from dataset.dataloader import DatasetLoader
from dataset.metrics import (
    token_accuracy,
    log_metrics
)

# ==========================================================
# Hyperparameters
# ==========================================================
block_size   = 256
batch_size   = 16
learning_rate = 1e-3

d_model    = 256
num_heads  = 4
num_layers = 4

gradient_accumulation_steps = 8
warmup_steps   = 2_000
total_steps    = 5_000_000   # Full training horizon for the cosine schedule.
                            # Should be the TOTAL steps you ever intend to train,
                            # not just one session. Increase if you plan to go further.
min_lr_ratio   = 0.1       # LR floor = learning_rate * min_lr_ratio

steps_to_run   = 50_000    # How many steps to run THIS session before stopping.
min_run_steps  = 5_000     # Minimum steps before plateau check activates.
plateau_window = 500        # Number of recent losses to watch.
plateau_eps    = 1e-3       # Plateau threshold. 1e-4 was too tight for streaming noise.
save_every     = 500

checkpoint_path = "model/checkpoint.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Reproducibility
# ==========================================================
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ==========================================================
# Tokenizer
# ==========================================================
tokenizer  = BPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
vocab_size = tokenizer.vocab_size

# ==========================================================
# Model & Optimisation
# ==========================================================
model = TransformerLM(
    vocab_size  = vocab_size,
    block_size  = block_size,
    d_model     = d_model,
    num_heads   = num_heads,
    num_layers  = num_layers
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn   = nn.CrossEntropyLoss()
scaler    = GradScaler()

# ==========================================================
# LR Scheduler  (Warmup + Cosine over the full training horizon)
#
# lr_lambda receives the number of times scheduler.step() has
# been called — i.e. the number of *optimiser* steps taken so
# far in this process. We add start_step later (after the
# checkpoint is loaded) so that the cosine position is correct
# regardless of how many times training has been resumed.
# ==========================================================
def make_lr_lambda(offset: int):
    """
    Returns a lambda that maps optimiser-step count → LR multiplier,
    with `offset` added so the schedule continues from where it left off.
    """
    def lr_lambda(local_step: int) -> float:
        global_step = local_step + offset
        if global_step < warmup_steps:
            return global_step / max(1, warmup_steps)
        progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(min_lr_ratio, cosine)
    return lr_lambda

# Placeholder — will be replaced after the checkpoint is loaded.
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(0))

# ==========================================================
# Resume Checkpoint
# ==========================================================
start_step   = 0
loss_history = []

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step   = checkpoint.get("step", 0)
    loss_history = checkpoint.get("loss_history", [])
    print(f"[RESUME] Resuming from step {start_step}")

    # Rebuild scheduler with the correct offset so the cosine curve
    # continues from the right position instead of restarting.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(start_step)
    )

    # Restore scheduler state if it was saved (older checkpoints won't have it).
    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("[RESUME] Scheduler state restored from checkpoint.")
    else:
        print("[RESUME] No scheduler state in checkpoint — cosine offset applied from step count.")
else:
    print("[INFO] No checkpoint found. Starting from scratch.")

target_step = start_step + steps_to_run
step        = start_step

print(f"[INFO] Training from step {start_step} to {target_step}  |  device={device}")
print(f"[INFO] LR at current step: {optimizer.param_groups[0]['lr']:.6f}")

# ==========================================================
# Dataset  (Streaming + Prefetch thread)
# ==========================================================
dataset = DatasetLoader(
    path       = "Data",
    tokenizer  = tokenizer,
    block_size = block_size,
    batch_size = batch_size,
    device     = device,
    shuffle    = True
)

batch_queue = queue.Queue(maxsize=4)  # slightly larger buffer than before

def prefetch_batches():
    consecutive_errors = 0
    while True:
        try:
            batch = dataset.get_batch()
            batch_queue.put(batch)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"[PREFETCH ERROR #{consecutive_errors}] {e}")
            if consecutive_errors >= 10:
                print("[PREFETCH] Too many consecutive errors — pausing 5 s before retrying.")
                time.sleep(5)
                consecutive_errors = 0

threading.Thread(target=prefetch_batches, daemon=True).start()

# ==========================================================
# Training Loop
# ==========================================================
model.train()
recent_losses = []
optimizer.zero_grad(set_to_none=True)

while step < target_step:
    # 1. Get batch from prefetch queue
    inputs, targets = batch_queue.get()

    # 2. Forward pass with mixed precision
    with autocast():
        logits = model(inputs)
        loss   = loss_fn(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        ) / gradient_accumulation_steps

    # 3. Backward pass
    scaler.scale(loss).backward()

    # 4. Optimiser step every N accumulation steps
    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # 5. Metrics bookkeeping
    loss_value = loss.item() * gradient_accumulation_steps
    recent_losses.append(loss_value)
    if len(recent_losses) > plateau_window:
        recent_losses.pop(0)

    step += 1

    if step % 10 == 0:
        print(f"Step {step:6d} | Loss {loss_value:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")
        loss_history.append((step, loss_value))

    # 6. Checkpoint + validation snapshot
    if step % save_every == 0:
        torch.save({
            "step":            step,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),   # saved now
            "loss_history":    loss_history,
        }, checkpoint_path)
        print(f"[CHECKPOINT] Saved @ step {step}")

        try:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = dataset.get_batch()
                val_logits  = model(val_inputs)
                val_loss    = loss_fn(
                    val_logits.reshape(-1, vocab_size),
                    val_targets.reshape(-1)
                ).item()
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                val_acc = token_accuracy(val_logits, val_targets)

                log_metrics(
                    step        = step,
                    loss        = val_loss,
                    perplexity  = val_ppl,
                    accuracy    = val_acc,
                    lr          = optimizer.param_groups[0]["lr"],
                    tokens_seen = step * batch_size * block_size,
                    split       = "val",
                )
            print(f"[METRICS] Val Loss {val_loss:.4f} | PPL {val_ppl:.2f} | Acc {val_acc:.2%}")
            model.train()
        except Exception as e:
            print(f"[METRICS ERROR] {e}")
            model.train()

    # 7. Early stopping on plateau
    if step >= (start_step + min_run_steps) and len(recent_losses) == plateau_window:
        spread = max(recent_losses) - min(recent_losses)
        if spread < plateau_eps:
            print(f"[STOP] Plateau detected at step {step}  (spread={spread:.6f})")
            break

print(f"[DONE] Session ended @ step {step}")