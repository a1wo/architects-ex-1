import os
import numpy as np
import torch
import tiktoken


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """Streams batches from pre-tokenized .npy shards in `data_root`."""

    def __init__(self, B, T, process_rank, num_processes, split, data_root, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


class TextDataLoader:
    """Tokenizes a single raw text file with the GPT-2 tokenizer and serves batches.

    Useful for local/Mac runs where the fineweb shards are not available. Holds
    the whole file in memory and wraps around when it reaches the end.
    """

    def __init__(self, B, T, process_rank, num_processes, filename, split="train", master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process

        enc = tiktoken.get_encoding("gpt2")
        with open(filename, "r") as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens from {filename}")
            print(f"1 epoch = {len(self.tokens) // (B * T * num_processes)} batches")
        self.reset()

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, wrap back to the start
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


def make_train_loader(cfg, process_rank, num_processes, master_process=True):
    """Builds the right train loader for a TrainConfig: raw text file or shards."""
    if cfg.text_file:
        return TextDataLoader(
            B=cfg.B, T=cfg.T, process_rank=process_rank, num_processes=num_processes,
            filename=cfg.text_file, split="train", master_process=master_process,
        )
    return DataLoaderLite(
        B=cfg.B, T=cfg.T, process_rank=process_rank, num_processes=num_processes,
        split="train", data_root=cfg.data_root, master_process=master_process,
    )
