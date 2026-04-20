import torch
from dataset.dataloader import DatasetLoader

def test_dataloader_shapes():
    loader = DatasetLoader(
        path="dataset/train.bin",
        block_size=8,
        batch_size=4,
        shuffle=False,
        device="cpu"
    )

    x, y = loader.get_batch()

    assert x.shape == (4, 7)
    assert y.shape == (4, 7)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_dataloader_shift():
    loader = DatasetLoader(
        path="dataset/train.bin",
        block_size=8,
        batch_size=1,
        shuffle=False,
        device="cpu"
    )

    x, y = loader.get_batch()

    # targets should be inputs shifted by 1
    assert torch.all(x[0, 1:] == y[0, :-1])
