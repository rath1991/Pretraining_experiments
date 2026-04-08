"""
download_data.py

Downloads and tokenizes training datasets to binary format.

Why binary format?
  - Raw text (FineWeb-Edu 10B) would be ~38GB
  - Tokenized uint16 binary is ~20GB (2 bytes per token)
  - Training reads directly from binary — no tokenization overhead at runtime
  - This is the same approach used by nanoGPT (Karpathy)

Datasets downloaded:
  1. FineWeb-Edu 10B sample  → data/fineweb_edu/train.bin + val.bin
  2. Pile: Wikipedia (en)    → data/pile/wikipedia_train.bin + val.bin
  3. Pile: Books1            → data/pile/books_train.bin + val.bin
  4. Pile: HackerNews        → data/pile/hackernews_train.bin + val.bin

Total on disk: ~37GB

Usage:
  uv run python download_data.py               # download all
  uv run python download_data.py --dataset fineweb
  uv run python download_data.py --dataset pile
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VAL_FRACTION = 0.005          # 0.5% validation split — same as GPT-2 paper
SHARD_SIZE = 100_000_000      # tokens per shard (~200MB each) before merging

enc = tiktoken.get_encoding("gpt2")   # vocab size 50,257
EOT = enc.eot_token                    # <|endoftext|> token = 50256
                                       # inserted between documents


# ---------------------------------------------------------------------------
# Core: tokenize a stream of documents → write train.bin + val.bin
# ---------------------------------------------------------------------------

def tokenize_and_save(stream, out_dir: str, name: str, val_fraction: float = VAL_FRACTION):
    """
    Streams documents, tokenizes each one, and writes to binary files.

    Binary format:
      - dtype: uint16 (each token is a 16-bit integer, vocab fits in 0–65535)
      - layout: flat sequence of token ids, no padding, no length headers
      - EOT token (<|endoftext|>) is prepended to each document
        so the model learns that documents start fresh after EOT

    Args:
        stream:       iterable of {"text": str, ...} dicts
        out_dir:      folder to write train.bin and val.bin
        name:         display name for progress bar
        val_fraction: fraction of tokens to hold out for validation
    """
    os.makedirs(out_dir, exist_ok=True)

    train_tokens = []
    val_tokens   = []
    total_tokens = 0

    print(f"\n{'='*60}")
    print(f"  Tokenizing: {name}")
    print(f"  Output:     {out_dir}")
    print(f"  Val split:  {val_fraction*100:.1f}%")
    print(f"{'='*60}")

    pbar = tqdm(stream, desc=name, unit=" docs")

    for doc in pbar:
        text = doc["text"]

        # Tokenize: EOT prepended to mark document boundary
        tokens = [EOT] + enc.encode_ordinary(text)

        # Route to train or val based on a deterministic hash of the first
        # token — this gives a stable split across multiple runs
        if (tokens[1] % 1000) < int(val_fraction * 1000):
            val_tokens.extend(tokens)
        else:
            train_tokens.extend(tokens)

        total_tokens += len(tokens)
        pbar.set_postfix({"total_tokens": f"{total_tokens/1e6:.1f}M"})

        # Flush to disk in shards to avoid running out of RAM
        if len(train_tokens) >= SHARD_SIZE:
            _flush(train_tokens, out_dir, "train")
            train_tokens = []
        if len(val_tokens) >= SHARD_SIZE // 10:
            _flush(val_tokens, out_dir, "val")
            val_tokens = []

    # Final flush
    if train_tokens:
        _flush(train_tokens, out_dir, "train")
    if val_tokens:
        _flush(val_tokens, out_dir, "val")

    # Merge shards into single train.bin and val.bin
    _merge_shards(out_dir, "train")
    _merge_shards(out_dir, "val")

    # Stats
    train_size = os.path.getsize(os.path.join(out_dir, "train.bin")) / 1e9
    val_size   = os.path.getsize(os.path.join(out_dir, "val.bin"))   / 1e9
    print(f"\n  Done — {total_tokens/1e9:.2f}B tokens total")
    print(f"  train.bin: {train_size:.2f} GB")
    print(f"  val.bin:   {val_size:.2f} GB")


def _flush(tokens: list, out_dir: str, split: str):
    """Append a list of tokens to a shard file."""
    shard_idx = len([f for f in os.listdir(out_dir) if f.startswith(f"{split}_shard")])
    path = os.path.join(out_dir, f"{split}_shard_{shard_idx:04d}.bin")
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)


def _merge_shards(out_dir: str, split: str):
    """Concatenate all shard files into one final bin file, then delete shards."""
    shards = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(f"{split}_shard")
    ])
    if not shards:
        return

    print(f"  Merging {len(shards)} {split} shards...")
    out_path = os.path.join(out_dir, f"{split}.bin")
    with open(out_path, "wb") as out_f:
        for shard_path in tqdm(shards, desc=f"  merging {split}"):
            data = np.fromfile(shard_path, dtype=np.uint16)
            data.tofile(out_f)
            os.remove(shard_path)


# ---------------------------------------------------------------------------
# Dataset-specific downloaders
# ---------------------------------------------------------------------------

def download_fineweb_edu():
    """
    FineWeb-Edu 10B sample from HuggingFace.
    High-quality educational web content. ~20GB tokenized.

    Why FineWeb-Edu?
      - Filtered subset of Common Crawl, scored for educational quality
      - Much higher signal-to-noise than raw web crawl
      - 10B tokens is enough for GPT-2 Small to learn meaningfully without
        overfitting in 8-10 hour runs
    """
    print("\nLoading FineWeb-Edu 10B sample (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    tokenize_and_save(
        stream=ds,
        out_dir=os.path.join(DATA_DIR, "fineweb_edu"),
        name="FineWeb-Edu 10B",
    )


def download_pile_subsets():
    """
    Three subsets from The Pile (monology/pile-uncopyrighted).
    Each has a different character — useful for Exp 7 (data quality sensitivity).

      Wikipedia (en) — factual, encyclopedic, clean structure
      Books1         — long-form narrative, coherent paragraphs
      HackerNews     — short, opinionated, informal web text

    Why these three?
      Comparing FineWeb-Edu (curated educational) vs Wikipedia (factual) vs
      Books (narrative) vs HackerNews (noisy web) gives a controlled gradient
      from clean → noisy without artificially corrupting data.
    """
    subsets = {
        "Wikipedia (en)": os.path.join(DATA_DIR, "pile", "wikipedia"),
        "Books1":          os.path.join(DATA_DIR, "pile", "books"),
        "HackerNews":      os.path.join(DATA_DIR, "pile", "hackernews"),
    }

    for pile_name, out_dir in subsets.items():
        print(f"\nLoading Pile subset: {pile_name} (streaming)...")

        ds = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        # Filter to only this subset
        filtered = (
            doc for doc in ds
            if doc.get("meta", {}).get("pile_set_name") == pile_name
        )

        tokenize_and_save(
            stream=filtered,
            out_dir=out_dir,
            name=f"Pile/{pile_name}",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["fineweb", "pile", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    if args.dataset in ("fineweb", "all"):
        download_fineweb_edu()

    if args.dataset in ("pile", "all"):
        download_pile_subsets()

    print("\nAll downloads complete.")
    print(f"Data stored in: {DATA_DIR}")
