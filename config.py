from dataclasses import dataclass, field
from typing import Optional

# -----------------------------------------------------------------------------
# Model architecture config

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

# -----------------------------------------------------------------------------
# Training config: bundles the model + data + optimization schedule for a run

@dataclass
class TrainConfig:
    name: str = "gpt2_124M"

    # model
    model: GPTConfig = field(default_factory=lambda: GPTConfig(vocab_size=50304))

    # data: use pre-tokenized .npy shards in `data_root`, OR a single raw `text_file`.
    # if `text_file` is set it takes precedence (handy for local/Mac runs).
    data_root: str = "/mnt/data/edu_fineweb10B"
    text_file: Optional[str] = None

    # batch / sequence
    total_batch_size: int = 524288 # 2**19, ~0.5M, in number of tokens
    B: int = 4   # micro batch size
    T: int = 64  # sequence length

    # optimization schedule
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1   # min_lr = max_lr * min_lr_ratio
    warmup_steps: int = 715
    max_steps: int = 100        # 19,073 steps is ~1 epoch for 10B tokens at 0.5M batch
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # runtime / logging
    log_dir: str = "log"
    log_every: int = 10
    seed: int = 1337
    matmul_precision: str = "high"

    @property
    def min_lr(self) -> float:
        return self.max_lr * self.min_lr_ratio


# -----------------------------------------------------------------------------
# Preset registry

def gpt2_124M() -> TrainConfig:
    """The full GPT-2 (124M) reproduction run (needs a CUDA box + fineweb shards)."""
    return TrainConfig()


def small_mac() -> TrainConfig:
    """A tiny model + tiny batch that trains on input.txt on a laptop (CPU / Apple MPS).

    ~few M params, reads the raw Shakespeare `input.txt` so no pre-tokenized
    shards are required. Meant for sanity-checking the pipeline locally.
    """
    return TrainConfig(
        name="small_mac",
        model=GPTConfig(
            block_size=128,
            vocab_size=50304,  # GPT-2 vocab padded to a nice number
            n_layer=4,
            n_head=4,
            n_embd=128,
        ),
        data_root="",          # unused; text_file takes precedence
        text_file="input.txt",
        total_batch_size=8 * 128 * 4,  # B*T*grad_accum -> 4 accumulation steps
        B=8,
        T=128,
        max_lr=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=10,
        max_steps=50,
        log_every=1,
    )


CONFIGS = {
    "gpt2_124M": gpt2_124M,
    "small_mac": small_mac,
}


def get_config(name: str) -> TrainConfig:
    if name not in CONFIGS:
        raise ValueError(f"unknown config '{name}'. available: {list(CONFIGS)}")
    return CONFIGS[name]()
