import os
import math
import time
import argparse

import torch

from config import get_config, CONFIGS
from model import GPT
from dataloader import make_train_loader

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py                      # full gpt2_124M run (needs CUDA + shards)
# python train_gpt2.py --config small_mac   # tiny run on input.txt (CPU / Apple MPS)
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT-2 style model.")
    parser.add_argument("--config", default="gpt2_124M", choices=list(CONFIGS),
                        help="which training preset to use")
    return parser.parse_args()


def setup_ddp():
    """Returns (ddp, rank, local_rank, world_size, master_process, device)."""
    # torchrun sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        from torch.distributed import init_process_group
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def get_lr(it, cfg):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_steps:
        return cfg.max_lr * (it + 1) / cfg.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.max_steps:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)


def main():
    args = parse_args()
    cfg = get_config(args.config)

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp()

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    # gradient accumulation
    assert cfg.total_batch_size % (cfg.B * cfg.T * ddp_world_size) == 0, \
        "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = cfg.total_batch_size // (cfg.B * cfg.T * ddp_world_size)
    if master_process:
        print(f"running config: {cfg.name}")
        print(f"total desired batch size: {cfg.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = make_train_loader(cfg, ddp_rank, ddp_world_size, master_process)

    torch.set_float32_matmul_precision(cfg.matmul_precision)

    # create model
    model = GPT(cfg.model)
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)

    # optimize!
    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay, learning_rate=cfg.max_lr,
        device_type=device_type, master_process=master_process,
    )

    # create the log directory we will write checkpoints to and log to
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, "log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    for step in range(cfg.max_steps):
        t0 = time.time()
        last_step = (step == cfg.max_steps - 1)

        # TODO: Implement the training step (grad accumulation, lr schedule, DDP sync)
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)

        logits, loss = model.forward(x, targets=y)
        lr = optimizer.param_groups[0]['lr']
        loss.backward()
        clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = loss.detach()
        norm = clipped_grad_norm.item()

        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work

        # Print loss and token throughput
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process and (step % cfg.log_every == 0 or last_step):
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        from torch.distributed import destroy_process_group
        destroy_process_group()


if __name__ == "__main__":
    main()
