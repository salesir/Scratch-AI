import torch
import time
import json
import os

_METRICS_FILE = "metrics.jsonl"


# ------------------------------
# Logging
# ------------------------------

def log_metrics(
    step,
    loss,
    perplexity,
    accuracy,
    lr,
    tokens_seen,
    split="train"
):
    record = {
        "time": time.time(),
        "step": step,
        "split": split,
        "loss": loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "lr": lr,
        "tokens_seen": tokens_seen,
    }

    with open(_METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------
# Core Metrics
# ------------------------------

def compute_loss(model, dataloader, device, criterion=torch.nn.CrossEntropyLoss()):
    """
    Compute average loss over a dataloader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss


def compute_perplexity(loss):
    """
    Convert cross-entropy loss to perplexity.
    """
    return torch.exp(torch.tensor(loss)).item()


def token_accuracy(logits, targets, top_k=1):
    """
    Compute top-k token-level accuracy.
    """
    with torch.no_grad():
        _, top_preds = logits.topk(top_k, dim=-1)
        correct = (top_preds == targets.unsqueeze(-1)).any(dim=-1).float()
        accuracy = correct.mean().item()
    return accuracy


# ------------------------------
# Inference Profiling
# ------------------------------

def profile_inference(model, batch, device):
    """
    Measure latency and memory usage for a single batch.
    """
    batch = batch.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    with torch.no_grad():
        _ = model(batch)
    end_time = time.time()

    latency = end_time - start_time
    memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    return latency, memory


# ------------------------------
# Sample Generation
# ------------------------------

def generate_samples(model, tokenizer, prompts, max_length=50, temperature=1.0, device='cpu'):
    """
    Generate text samples from a list of prompts using greedy decoding.
    Uses the project's BPETokenizer (returns plain token id lists).
    """
    model.eval()
    samples = []

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)

        for _ in range(max_length):
            idx_cond = idx[:, -model.block_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        text = tokenizer.decode(idx[0].tolist())
        samples.append(text)

    return samples