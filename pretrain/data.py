import os
import numpy as np
import torch


class BinDataset:
    """
    Reads a flat uint16 binary file (produced by download_data.py).
    Each call to get_batch() samples random non-overlapping-start windows.
    """

    def __init__(self, path: str, block_size: int):
        self.block_size = block_size
        # memmap: reads from disk on demand — no RAM overhead for 20GB file
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str):
        # sample random start positions
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        # y is x shifted right by 1 — next-token prediction target
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


def get_datasets(data_dir: str, block_size: int):
    train = BinDataset(os.path.join(data_dir, "train.bin"), block_size)
    val   = BinDataset(os.path.join(data_dir, "val.bin"),   block_size)
    return train, val
