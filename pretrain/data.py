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
        # Sample random start positions — one per sequence in the batch
        start_indices = torch.randint(len(self), (batch_size,))

        # input_tokens: the token sequence fed into the model
        input_tokens = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in start_indices
        ])
        # targets: the same window shifted right by 1 — the next-token prediction target.
        # At each position t, the model must predict token t+1 given tokens 0..t.
        targets = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in start_indices
        ])
        return input_tokens.to(device), targets.to(device)


def get_datasets(data_dir: str, block_size: int):
    train = BinDataset(os.path.join(data_dir, "train.bin"), block_size)
    val   = BinDataset(os.path.join(data_dir, "val.bin"),   block_size)
    return train, val
