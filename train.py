"""
train.py
--------
Trains whichever architecture in model.py matches your data, produced by
data_cleaning.py. Pick a mode:

    python train.py --mode encdec                    # Transformer (encoder-decoder)
    python train.py --mode decoder                   # TransformerDecoderOnly (GPT-style)
    python train.py --mode encoder --task classify    # TransformerEncoderOnly (classification)
    python train.py --mode encoder --task mlm         # TransformerEncoderOnly (masked-LM pretrain)
    python train.py --mode encoder --task regression  # TransformerRegression (predict a float from text)

Each mode reads from ./data/<matching_subfolder>/ (created by data_cleaning.py)
and writes checkpoints to ./checkpoints/<mode_name>/{last,best}.pt.

Shared training recipe (matches the original paper) for encdec / decoder / encoder-mlm:
  - Adam (betas=0.9, 0.98, eps=1e-9) + Noam warmup/decay LR schedule
  - label smoothing
  - gradient clipping

encoder --task classify and encoder --task regression are the exception: they use a flat
AdamW (--lr, --weight-decay) instead of Noam. Noam's peak lr scales as d_model**-0.5 *
warmup_steps**-0.5, and with a small d_model / small dataset / short warmup (as
classification/regression typically use) that peak is reached within the first few epochs
and is high enough to blow the model up into a degenerate state (classification: "predicts
everything as 50/50", train_loss stuck at ln(2); regression: predicts the target mean for
everything, no matter the input). AdamW with a small flat lr is the standard recipe for
training/fine-tuning small Transformer classifiers/regressors and avoids that failure mode.

Multi-GPU: this script uses DistributedDataParallel (DDP), one process per GPU. Launch
with torchrun instead of plain `python`:

    torchrun --nproc_per_node=<NUM_GPUS> train.py --mode decoder ...

Running it as plain `python train.py ...` still works exactly as before - it just runs
as a single process on one GPU (or CPU), no DDP involved. See setup_distributed() below
for how the two modes are told apart.
"""

import os
import json
import pickle
import math
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import (
    Transformer, TransformerDecoderOnly, TransformerEncoderOnly, TransformerRegression,
    default_num_kv_heads,
)

# These five are finalized by setup_distributed() once (called at the top of main()),
# before any training function runs. DEVICE, in particular, is *not* just "cuda if
# available" anymore once DDP is in play - each process gets pinned to its own single
# GPU (cuda:LOCAL_RANK), which is why every place below that used to say
# "torch.cuda.is_available()" now reads from these globals instead.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0
IS_DISTRIBUTED = False


def resolve_arch_args(args):
    """Builds the architecture dict shared by all four training modes. Fills in a
    sensible Grouped-Query-Attention group size (--num-kv-heads) when it isn't given
    explicitly, and always carries the RoPE base (--rope-theta) along too, so both
    end up saved into the checkpoint config and reused verbatim by inference.py."""
    num_kv_heads = args.num_kv_heads or default_num_kv_heads(args.num_heads)
    if args.num_heads % num_kv_heads != 0:
        raise ValueError(
            f"--num-heads ({args.num_heads}) must be divisible by --num-kv-heads ({num_kv_heads})"
        )
    return {
        "d_model": args.d_model, "num_layers": args.num_layers, "num_heads": args.num_heads,
        "num_kv_heads": num_kv_heads, "d_ff": args.d_ff, "dropout": args.dropout,
        "rope_theta": args.rope_theta,
    }


# --------------------------------------------------------------------------- #
# Distributed (DDP) helpers
# --------------------------------------------------------------------------- #
def setup_distributed():
    """Detects whether we were launched via torchrun (it sets RANK/WORLD_SIZE/
    LOCAL_RANK env vars) and, if so, initializes the process group and points this
    process's DEVICE at its own single GPU. If those env vars aren't present - i.e.
    someone just ran `python train.py ...` - this is a no-op and everything behaves
    exactly like a single-process run always has.

    Must be called once at the very top of main(), before any model/optimizer/
    dataloader is built, since prepare_model_for_device / make_dataloader /
    reduce_sum all read the globals this function sets."""
    global DEVICE, RANK, WORLD_SIZE, LOCAL_RANK, IS_DISTRIBUTED

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    IS_DISTRIBUTED = WORLD_SIZE > 1

    if not IS_DISTRIBUTED:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    # NCCL needs real GPUs; fall back to gloo so `torchrun --nproc_per_node=N` still
    # works (slowly, over CPU) on a machine with no CUDA, e.g. for a local smoke test.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
        DEVICE = torch.device("cuda", LOCAL_RANK)
    else:
        DEVICE = torch.device("cpu")

    if RANK == 0:
        print(f"Distributed training: world_size={WORLD_SIZE}, backend={backend}")


def cleanup_distributed():
    """Tears down the process group. No-op outside DDP."""
    if IS_DISTRIBUTED:
        dist.destroy_process_group()


def is_main_process():
    """True on rank 0, or always True outside DDP. Use this to gate prints and
    checkpoint saves so N processes don't race on the same stdout / file."""
    return RANK == 0


def reduce_sum(tensor):
    """All-reduces (SUM) a scalar tensor across every DDP process in place. Every
    train/val loop below accumulates loss/token/correct counts as running sums on
    DEVICE specifically so this can fold in the other ranks' shards with one
    collective call right before the final division - the same trick that already
    avoided a per-batch .item() sync now also gives a *global* metric instead of
    just this process's local shard. No-op (returns tensor unchanged) outside DDP."""
    if IS_DISTRIBUTED:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def raw_model(model):
    """Returns the underlying module whether or not `model` is DDP-wrapped. Use this
    whenever you need to reach model-specific attributes/methods that DDP doesn't
    proxy, and whenever saving a checkpoint (checkpoints always store the unwrapped
    state dict - see save_checkpoint)."""
    return model.module if isinstance(model, DDP) else model


def load_model_state(model, state_dict):
    """Loads a checkpoint's (always-unwrapped) state dict into `model`, whether or
    not `model` is currently DDP-wrapped."""
    raw_model(model).load_state_dict(state_dict)


def prepare_model_for_device(model):
    """Moves model to this process's DEVICE. Under torchrun (world_size > 1, see
    setup_distributed), wraps it in DistributedDataParallel bound to this process's
    single GPU: each process owns exactly one GPU and one copy of the model, and DDP
    keeps their gradients in sync with an automatic all-reduce during backward()
    (unlike the old single-process gather/scatter nn.DataParallel used to do, which
    couldn't scale past one machine and was slower even on one machine with several
    GPUs). Outside torchrun this just moves the model to DEVICE and returns it
    unwrapped - identical to a plain single-GPU or CPU run. Checkpoints always store
    the *unwrapped* state dict (see save_checkpoint / load_model_state), so a
    checkpoint produced by an 8-process DDP run loads back fine into a single-process
    run and vice versa."""
    model = model.to(DEVICE)
    if IS_DISTRIBUTED:
        device_ids = [LOCAL_RANK] if DEVICE.type == "cuda" else None
        output_device = device_ids[0] if device_ids else None
        model = DDP(model, device_ids=device_ids, output_device=output_device)
        if is_main_process():
            print(f"Distributed: wrapped model in DDP across {WORLD_SIZE} processes "
                  f"(backend={dist.get_backend()})")
    elif DEVICE.type == "cuda":
        print(f"Found 1 GPU -> training on {DEVICE} (single process, no DDP). "
              f"Use `torchrun --nproc_per_node=N train.py ...` for multi-GPU.")
    else:
        print("No GPU found -> training on CPU")
    return model


# --------------------------------------------------------------------------- #
# Throughput helpers: these three things matter far more than which GPU you
# have for a model this small, and are the usual reason "faster" hardware
# doesn't actually train faster - see the long comment in prepare_model_for_device
# usage sites and the CLI --num-workers/--amp help text for the full story.
# --------------------------------------------------------------------------- #
def dataloader_kwargs(args):
    """With num_workers=0 (the DataLoader default), every batch's collate_fn
    (tokenizing/padding, all pure Python) runs in the main process in between
    training steps, so the GPU sits completely idle waiting for the CPU to
    build the next batch - a bottleneck whose cost has nothing to do with
    which GPU you bought, which is why a GTX 1660 Super and a T4 can post the
    same epoch time. num_workers>0 prefetches batches in background
    processes while the GPU is busy with the current one; pin_memory copies
    the finished batch into page-locked host memory so the .to(DEVICE,
    non_blocking=True) calls below can actually overlap with compute instead
    of blocking on a synchronous copy."""
    return dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def make_dataloader(dataset, batch_size, shuffle, collate_fn, args):
    """Builds a DataLoader that works identically in single-process and torchrun/DDP
    runs. Under DDP each rank gets a DistributedSampler slice of the dataset - so N
    processes each train on 1/N of the data per epoch instead of all N redundantly
    training on the whole thing - and the sampler itself does the shuffling, so
    `shuffle` only gets passed straight to the DataLoader when there's no sampler
    (DataLoader raises if you pass both `shuffle=True` and a `sampler`). Returns
    (loader, sampler); sampler is None outside DDP. When it isn't None, call
    sampler.set_epoch(epoch) once before iterating each training epoch (see the
    training loops below) - without that call every rank would reshuffle to the
    exact same order every epoch instead of a fresh one.

    Note for validation: DistributedSampler pads the tail so every rank gets an
    equal number of batches, which can very slightly double-count a handful of
    samples when the dataset size isn't evenly divisible by world_size. Negligible
    for picking a best checkpoint, but worth knowing about."""
    sampler = None
    if IS_DISTRIBUTED:
        sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=shuffle)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler, collate_fn=collate_fn, **dataloader_kwargs(args),
    )
    return loader, sampler


def autocast_ctx(args):
    """Mixed-precision context (fp16 autocast), enabled only when --amp is
    passed and a GPU is present. This is what actually engages a T4's Tensor
    Cores for the matmuls inside attention/FFN - without it, training runs in
    plain fp32 and Tensor Cores never get used, which is a second reason a
    "much more capable" GPU can post the same wall-clock time as an older
    one. (A GTX 1660 Super has no Tensor Cores at all, so --amp helps it far
    less - expect the gap between the two GPUs to actually show up once this
    is on.)"""
    enabled = args.amp and DEVICE.type == "cuda"
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=DEVICE.type, dtype=dtype, enabled=enabled)


def make_grad_scaler(args):
    return torch.amp.GradScaler(DEVICE.type,enabled=args.amp and DEVICE.type == "cuda")


def backward_step(loss, model, optimizer, scaler, grad_clip):
    """One optimizer update, with or without AMP's loss scaling. scaler is a
    no-op passthrough when --amp wasn't passed (see make_grad_scaler)."""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


# --------------------------------------------------------------------------- #
# Noam learning-rate schedule (shared across all modes)
# --------------------------------------------------------------------------- #
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


# --------------------------------------------------------------------------- #
# Within-epoch progress reporting
# --------------------------------------------------------------------------- #
class ProgressTracker:
    """Prints a single, live-updating progress line *during* an epoch (step count,
    metrics, throughput, ETA for the rest of the epoch) instead of the old
    behavior where stdout stayed silent until the whole epoch had finished.

    Printing is throttled to once every `min_interval` seconds (default 2s) rather
    than every batch, for two reasons: it keeps a fast GPU from flooding the
    terminal with thousands of lines, and it means any tensor metrics you pass in
    (e.g. a running loss) are only pulled to the CPU with .item() - which forces a
    GPU sync - at those throttled print times, not on every single step. That's a
    much lower sync frequency than "once per batch" while still being far more
    responsive than "once per epoch".

    Usage:
        progress = ProgressTracker(len(loader), epoch, "decoder/train") if is_main_process() else None
        for batch in loader:
            ...
            if progress is not None:
                progress.update(loss=running_loss_tensor, lr=current_lr)
        if progress is not None:
            progress.finish()
    """
    def __init__(self, total_steps, epoch, label, min_interval=2.0):
        self.total_steps = max(total_steps, 1)
        self.epoch = epoch
        self.label = label
        self.min_interval = min_interval
        self.start_time = time.time()
        self.last_print_time = 0.0  # 0 forces the very first step to print right away
        self.step = 0

    def update(self, **metrics):
        self.step += 1
        now = time.time()
        is_last_step = self.step >= self.total_steps
        if (now - self.last_print_time) < self.min_interval and not is_last_step:
            return  # not time to print yet - just keep accumulating
        self.last_print_time = now

        elapsed = now - self.start_time
        rate = self.step / elapsed if elapsed > 0 else 0.0
        eta_seconds = (self.total_steps - self.step) / rate if rate > 0 else 0.0
        pct = 100 * self.step / self.total_steps

        parts = []
        for name, value in metrics.items():
            value = value.item() if torch.is_tensor(value) else value
            parts.append(f"{name} {value:.4f}")
        metrics_str = (" | " + " | ".join(parts)) if parts else ""

        line = (f"[{self.label}] epoch {self.epoch:02d} | step {self.step}/{self.total_steps} "
                f"({pct:5.1f}%){metrics_str} | elapsed {self._fmt(elapsed)} | "
                f"eta {self._fmt(eta_seconds)}   ")
        # \r overwrites the previous progress line in place instead of scrolling
        print(f"\r{line}", end="", flush=True)

    def finish(self):
        # move to a fresh line so whatever prints next (e.g. the epoch summary)
        # doesn't get appended to the end of the progress line
        print()

    @staticmethod
    def _fmt(seconds):
        seconds = max(seconds, 0)
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def save_checkpoint(ckpt_dir, name, model, optimizer, scheduler, config, epoch, metric):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "model_state": raw_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_step": scheduler.step_num if scheduler is not None else None,
        "config": config,
        "epoch": epoch,
        "metric": metric,
    }, os.path.join(ckpt_dir, name))


def load_resume(resume_path):
    """Loads a checkpoint dict to resume from, or returns None if no path was given."""
    if resume_path is None:
        return None
    print(f"Resuming from {resume_path} ...")
    return torch.load(resume_path, map_location=DEVICE)


# =========================================================================== #
# MODE 1: encoder-decoder (translation)
# =========================================================================== #
class EncDecDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_encdec(batch, pad_idx):
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)

    src_batch = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_batch[i, :len(s)] = s
        tgt_batch[i, :len(t)] = t
    return src_batch, tgt_batch


def train_encdec(args):
    data_dir = os.path.join(args.data_root, "encdec")
    ckpt_dir = os.path.join(args.ckpt_root, "encdec")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    pad_idx = meta["pad_idx"]

    train_ds = EncDecDataset(os.path.join(data_dir, "train.pkl"))
    val_ds = EncDecDataset(os.path.join(data_dir, "val.pkl"))

    collate = lambda b: collate_encdec(b, pad_idx)
    train_loader, train_sampler = make_dataloader(train_ds, args.batch_size, True, collate, args)
    val_loader, _ = make_dataloader(val_ds, args.batch_size, False, collate, args)

    resume_ckpt = load_resume(args.resume)
    # if resuming, architecture must match the saved checkpoint - CLI arch flags are ignored
    arch = resume_ckpt["config"] if resume_ckpt else resolve_arch_args(args)

    model = Transformer(
        src_vocab_size=meta["src_vocab_size"],
        tgt_vocab_size=meta["tgt_vocab_size"],
        d_model=arch["d_model"], num_layers=arch["num_layers"], num_heads=arch["num_heads"],
        num_kv_heads=arch["num_kv_heads"], d_ff=arch["d_ff"], max_len=meta["max_len"] + 10,
        dropout=arch["dropout"], pad_idx=pad_idx, rope_theta=arch["rope_theta"],
    )
    model = prepare_model_for_device(model)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, arch["d_model"], args.warmup_steps)
    scaler = make_grad_scaler(args)

    config = {
        "mode": "encdec", "d_model": arch["d_model"], "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"], "num_kv_heads": arch["num_kv_heads"], "d_ff": arch["d_ff"],
        "dropout": arch["dropout"], "rope_theta": arch["rope_theta"],
        "src_vocab_size": meta["src_vocab_size"], "tgt_vocab_size": meta["tgt_vocab_size"],
        "max_len": meta["max_len"] + 10, "pad_idx": pad_idx,
    }

    start_epoch = 1
    best_val = float("inf")
    if resume_ckpt is not None:
        load_model_state(model, resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.step_num = resume_ckpt.get("scheduler_step", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt["metric"]
        if is_main_process():
            print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # re-shuffles each rank's shard differently per epoch

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_tokens_sum = torch.zeros((), device=DEVICE)
        progress = ProgressTracker(len(train_loader), epoch, "encdec/train", min_interval=args.progress_interval) if is_main_process() else None
        for src, tgt in train_loader:
            src = src.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            with autocast_ctx(args):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            lr = scheduler.step()
            backward_step(loss, model, optimizer, scaler, args.grad_clip)

            # accumulate on-device; only synced to CPU at throttled progress prints
            # (see ProgressTracker) and once more at the end of the epoch below -
            # never on every single step
            n_tok = (tgt_out != pad_idx).sum()
            train_loss_sum += loss.detach() * n_tok
            train_tokens_sum += n_tok
            if progress is not None:
                progress.update(loss=train_loss_sum / train_tokens_sum.clamp(min=1), lr=lr)
        if progress is not None:
            progress.finish()
        reduce_sum(train_loss_sum)
        reduce_sum(train_tokens_sum)
        train_loss = (train_loss_sum / train_tokens_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_tokens_sum = torch.zeros((), device=DEVICE)
        val_progress = ProgressTracker(len(val_loader), epoch, "encdec/val", min_interval=args.progress_interval) if is_main_process() else None
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(DEVICE, non_blocking=True)
                tgt = tgt.to(DEVICE, non_blocking=True)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                with autocast_ctx(args):
                    logits = model(src, tgt_in)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                n_tok = (tgt_out != pad_idx).sum()
                val_loss_sum += loss.detach() * n_tok
                val_tokens_sum += n_tok
                if val_progress is not None:
                    val_progress.update(loss=val_loss_sum / val_tokens_sum.clamp(min=1))
        if val_progress is not None:
            val_progress.finish()
        reduce_sum(val_loss_sum)
        reduce_sum(val_tokens_sum)
        val_loss = (val_loss_sum / val_tokens_sum.clamp(min=1)).item()

        if is_main_process():
            print(f"[encdec] epoch {epoch:02d} | train_loss {train_loss:.4f} "
                  f"(ppl {math.exp(min(train_loss, 20)):.2f}) | val_loss {val_loss:.4f} "
                  f"(ppl {math.exp(min(val_loss, 20)):.2f}) | {time.time()-start:.1f}s")

            save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
                print(f"  -> new best (val_loss={val_loss:.4f})")
        if IS_DISTRIBUTED:
            dist.barrier()  # keep ranks together while rank 0 writes the checkpoint files


# =========================================================================== #
# MODE 2: decoder-only (causal LM)
# =========================================================================== #
class LMDataset(Dataset):
    """Works for both 'stream' chunks (fixed block_size+1) and 'lines' (variable length)."""
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def collate_lm(batch, pad_idx):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


def train_decoder(args):
    data_dir = os.path.join(args.data_root, "decoder")
    ckpt_dir = os.path.join(args.ckpt_root, "decoder")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    pad_idx = meta["pad_idx"]
    max_len = meta.get("block_size", meta.get("max_len", 256)) + 8

    train_ds = LMDataset(os.path.join(data_dir, "train.pkl"))
    val_ds = LMDataset(os.path.join(data_dir, "val.pkl"))

    collate = lambda b: collate_lm(b, pad_idx)
    train_loader, train_sampler = make_dataloader(train_ds, args.batch_size, True, collate, args)
    val_loader, _ = make_dataloader(val_ds, args.batch_size, False, collate, args)

    resume_ckpt = load_resume(args.resume)
    arch = resume_ckpt["config"] if resume_ckpt else resolve_arch_args(args)

    model = TransformerDecoderOnly(
        vocab_size=meta["vocab_size"], d_model=arch["d_model"], num_layers=arch["num_layers"],
        num_heads=arch["num_heads"], num_kv_heads=arch["num_kv_heads"], d_ff=arch["d_ff"],
        max_len=max_len, dropout=arch["dropout"], pad_idx=pad_idx, rope_theta=arch["rope_theta"],
    )
    model = prepare_model_for_device(model)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, arch["d_model"], args.warmup_steps)
    scaler = make_grad_scaler(args)

    config = {
        "mode": "decoder", "d_model": arch["d_model"], "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"], "num_kv_heads": arch["num_kv_heads"], "d_ff": arch["d_ff"],
        "dropout": arch["dropout"], "rope_theta": arch["rope_theta"],
        "vocab_size": meta["vocab_size"], "max_len": max_len, "pad_idx": pad_idx,
    }

    start_epoch = 1
    best_val = float("inf")
    if resume_ckpt is not None:
        load_model_state(model, resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.step_num = resume_ckpt.get("scheduler_step", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt["metric"]
        if is_main_process():
            print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_tokens_sum = torch.zeros((), device=DEVICE)
        progress = ProgressTracker(len(train_loader), epoch, "decoder/train", min_interval=args.progress_interval) if is_main_process() else None
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]  # next-token prediction

            with autocast_ctx(args):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            lr = scheduler.step()
            backward_step(loss, model, optimizer, scaler, args.grad_clip)

            n_tok = (y != pad_idx).sum()
            train_loss_sum += loss.detach() * n_tok
            train_tokens_sum += n_tok
            if progress is not None:
                progress.update(loss=train_loss_sum / train_tokens_sum.clamp(min=1), lr=lr)
        if progress is not None:
            progress.finish()
        reduce_sum(train_loss_sum)
        reduce_sum(train_tokens_sum)
        train_loss = (train_loss_sum / train_tokens_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_tokens_sum = torch.zeros((), device=DEVICE)
        val_progress = ProgressTracker(len(val_loader), epoch, "decoder/val", min_interval=args.progress_interval) if is_main_process() else None
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE, non_blocking=True)
                x, y = batch[:, :-1], batch[:, 1:]
                with autocast_ctx(args):
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                n_tok = (y != pad_idx).sum()
                val_loss_sum += loss.detach() * n_tok
                val_tokens_sum += n_tok
                if val_progress is not None:
                    val_progress.update(loss=val_loss_sum / val_tokens_sum.clamp(min=1))
        if val_progress is not None:
            val_progress.finish()
        reduce_sum(val_loss_sum)
        reduce_sum(val_tokens_sum)
        val_loss = (val_loss_sum / val_tokens_sum.clamp(min=1)).item()

        if is_main_process():
            print(f"[decoder] epoch {epoch:02d} | train_loss {train_loss:.4f} "
                  f"(ppl {math.exp(min(train_loss, 20)):.2f}) | val_loss {val_loss:.4f} "
                  f"(ppl {math.exp(min(val_loss, 20)):.2f}) | {time.time()-start:.1f}s")

            save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
                print(f"  -> new best (val_loss={val_loss:.4f})")
        if IS_DISTRIBUTED:
            dist.barrier()


# =========================================================================== #
# MODE 3a: encoder-only, classify
# =========================================================================== #
class ClassifyDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long), label


def collate_classify(batch, pad_idx):
    tokens, labels = zip(*batch)
    max_len = max(len(t) for t in tokens)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded[i, :len(t)] = t
    return padded, torch.tensor(labels, dtype=torch.long)


def train_encoder_classify(args):
    data_dir = os.path.join(args.data_root, "encoder_classify")
    ckpt_dir = os.path.join(args.ckpt_root, "encoder_classify")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    pad_idx = meta["pad_idx"]

    train_ds = ClassifyDataset(os.path.join(data_dir, "train.pkl"))
    val_ds = ClassifyDataset(os.path.join(data_dir, "val.pkl"))

    collate = lambda b: collate_classify(b, pad_idx)
    train_loader, train_sampler = make_dataloader(train_ds, args.batch_size, True, collate, args)
    val_loader, _ = make_dataloader(val_ds, args.batch_size, False, collate, args)

    resume_ckpt = load_resume(args.resume)
    arch = resume_ckpt["config"] if resume_ckpt else {**resolve_arch_args(args), "pooling": args.pooling}

    model = TransformerEncoderOnly(
        vocab_size=meta["vocab_size"], d_model=arch["d_model"], num_layers=arch["num_layers"],
        num_heads=arch["num_heads"], num_kv_heads=arch["num_kv_heads"], d_ff=arch["d_ff"],
        max_len=meta["max_len"] + 10, dropout=arch["dropout"], pad_idx=pad_idx,
        num_classes=meta["num_classes"], pooling=arch["pooling"], rope_theta=arch["rope_theta"],
        # classify-only training never calls task="mlm", so don't build mlm_head at all -
        # an unused head's params would never get a gradient under DDP, which is exactly
        # what DistributedDataParallel's "unused parameters" error/hang is complaining
        # about (see the comment on TransformerEncoderOnly in model.py).
        use_mlm_head=False,
    )
    model = prepare_model_for_device(model)

    criterion = nn.CrossEntropyLoss()
    # NOTE: classification does NOT use the shared Noam schedule. Noam was tuned for
    # huge-step MT/LM training (paper default: d_model=512, warmup=4000 -> peak lr ~7e-4).
    # On a small d_model / small dataset / short warmup like this one, Noam's peak lr
    # (~ d_model**-0.5 * warmup_steps**-0.5) comes out several times higher than that and
    # is hit within the first few epochs, which is what was blowing the classifier up to
    # the ln(2) "predicts everything as 50/50" collapse. A flat AdamW lr is the standard
    # recipe for fine-tuning/training small Transformer classifiers instead.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    scaler = make_grad_scaler(args)

    config = {
        "mode": "encoder_classify", "d_model": arch["d_model"], "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"], "num_kv_heads": arch["num_kv_heads"], "d_ff": arch["d_ff"],
        "dropout": arch["dropout"], "rope_theta": arch["rope_theta"],
        "vocab_size": meta["vocab_size"], "max_len": meta["max_len"] + 10, "pad_idx": pad_idx,
        "num_classes": meta["num_classes"], "pooling": arch["pooling"],
        "label_names": meta.get("label_names"),
        "use_mlm_head": False,
    }

    start_epoch = 1
    best_val_acc = 0.0
    if resume_ckpt is not None:
        load_model_state(model, resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_acc = resume_ckpt["metric"]
        if is_main_process():
            print(f"Resumed at epoch {start_epoch} (previous val_acc={best_val_acc:.3f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_correct_sum = torch.zeros((), device=DEVICE)
        train_total_sum = torch.zeros((), device=DEVICE)
        progress = ProgressTracker(len(train_loader), epoch, "encoder_classify/train", min_interval=args.progress_interval) if is_main_process() else None
        for tokens, labels in train_loader:
            tokens = tokens.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast_ctx(args):
                logits = model(tokens, task="classify")
                loss = criterion(logits, labels)

            backward_step(loss, model, optimizer, scaler, args.grad_clip)

            bs = tokens.size(0)  # plain python int, shape metadata only - no sync
            train_loss_sum += loss.detach() * bs
            train_correct_sum += (logits.argmax(-1) == labels).sum()
            train_total_sum += bs
            if progress is not None:
                progress.update(loss=train_loss_sum / train_total_sum.clamp(min=1),
                                 acc=train_correct_sum / train_total_sum.clamp(min=1))
        if progress is not None:
            progress.finish()
        reduce_sum(train_loss_sum)
        reduce_sum(train_correct_sum)
        reduce_sum(train_total_sum)
        train_loss = (train_loss_sum / train_total_sum.clamp(min=1)).item()
        train_acc = (train_correct_sum / train_total_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_correct_sum = torch.zeros((), device=DEVICE)
        val_total_sum = torch.zeros((), device=DEVICE)
        val_progress = ProgressTracker(len(val_loader), epoch, "encoder_classify/val", min_interval=args.progress_interval) if is_main_process() else None
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens = tokens.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast_ctx(args):
                    logits = model(tokens, task="classify")
                    loss = criterion(logits, labels)
                bs = tokens.size(0)
                val_loss_sum += loss.detach() * bs
                val_correct_sum += (logits.argmax(-1) == labels).sum()
                val_total_sum += bs
                if val_progress is not None:
                    val_progress.update(loss=val_loss_sum / val_total_sum.clamp(min=1),
                                         acc=val_correct_sum / val_total_sum.clamp(min=1))
        if val_progress is not None:
            val_progress.finish()
        reduce_sum(val_loss_sum)
        reduce_sum(val_correct_sum)
        reduce_sum(val_total_sum)
        val_loss = (val_loss_sum / val_total_sum.clamp(min=1)).item()
        val_acc = (val_correct_sum / val_total_sum.clamp(min=1)).item()

        if is_main_process():
            print(f"[encoder/classify] epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | lr {optimizer.param_groups[0]['lr']:.6f} "
                  f"| {time.time()-start:.1f}s")

            save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_acc)
                print(f"  -> new best (val_acc={val_acc:.3f})")
        if IS_DISTRIBUTED:
            dist.barrier()


# =========================================================================== #
# MODE 3c: encoder-only, regression
# =========================================================================== #
class RegressionDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, target = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long), target


def collate_regression(batch, pad_idx):
    tokens, targets = zip(*batch)
    max_len = max(len(t) for t in tokens)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded[i, :len(t)] = t
    return padded, torch.tensor(targets, dtype=torch.float32)


def train_encoder_regression(args):
    data_dir = os.path.join(args.data_root, "encoder_regression")
    ckpt_dir = os.path.join(args.ckpt_root, "encoder_regression")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    pad_idx = meta["pad_idx"]
    num_targets = meta.get("num_targets", 1)

    train_ds = RegressionDataset(os.path.join(data_dir, "train.pkl"))
    val_ds = RegressionDataset(os.path.join(data_dir, "val.pkl"))

    collate = lambda b: collate_regression(b, pad_idx)
    train_loader, train_sampler = make_dataloader(train_ds, args.batch_size, True, collate, args)
    val_loader, _ = make_dataloader(val_ds, args.batch_size, False, collate, args)

    resume_ckpt = load_resume(args.resume)
    arch = resume_ckpt["config"] if resume_ckpt else {**resolve_arch_args(args), "pooling": args.pooling}

    model = TransformerRegression(
        vocab_size=meta["vocab_size"], d_model=arch["d_model"], num_layers=arch["num_layers"],
        num_heads=arch["num_heads"], num_kv_heads=arch["num_kv_heads"], d_ff=arch["d_ff"],
        max_len=meta["max_len"] + 10, dropout=arch["dropout"], pad_idx=pad_idx,
        num_targets=num_targets, pooling=arch["pooling"], rope_theta=arch["rope_theta"],
    )
    model = prepare_model_for_device(model)

    criterion = nn.MSELoss()
    # Same reasoning as encoder --task classify: a small d_model / small dataset / short
    # warmup makes Noam's peak lr high enough to blow this small head up (here: collapsing
    # to always predicting the target mean). Flat AdamW is the standard, stable choice.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    scaler = make_grad_scaler(args)

    config = {
        "mode": "encoder_regression", "d_model": arch["d_model"], "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"], "num_kv_heads": arch["num_kv_heads"], "d_ff": arch["d_ff"],
        "dropout": arch["dropout"], "rope_theta": arch["rope_theta"],
        "vocab_size": meta["vocab_size"], "max_len": meta["max_len"] + 10, "pad_idx": pad_idx,
        "num_targets": num_targets, "pooling": arch["pooling"],
        # needed by inference.py to map standardized predictions back to the original scale
        "target_mean": meta.get("target_mean", 0.0), "target_std": meta.get("target_std", 1.0),
    }

    start_epoch = 1
    best_val_loss = float("inf")
    if resume_ckpt is not None:
        load_model_state(model, resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["metric"]
        if is_main_process():
            print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_abs_err_sum = torch.zeros((), device=DEVICE)
        train_total_sum = torch.zeros((), device=DEVICE)
        progress = ProgressTracker(len(train_loader), epoch, "encoder_regression/train", min_interval=args.progress_interval) if is_main_process() else None
        for tokens, targets in train_loader:
            tokens = tokens.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with autocast_ctx(args):
                preds = model(tokens, task="regress").squeeze(-1)  # (batch,) - assumes num_targets == 1
                loss = criterion(preds, targets)

            backward_step(loss, model, optimizer, scaler, args.grad_clip)

            bs = tokens.size(0)
            train_loss_sum += loss.detach() * bs
            train_abs_err_sum += (preds.detach() - targets).abs().sum()
            train_total_sum += bs
            if progress is not None:
                progress.update(mse=train_loss_sum / train_total_sum.clamp(min=1),
                                 mae=train_abs_err_sum / train_total_sum.clamp(min=1))
        if progress is not None:
            progress.finish()
        reduce_sum(train_loss_sum)
        reduce_sum(train_abs_err_sum)
        reduce_sum(train_total_sum)
        train_loss = (train_loss_sum / train_total_sum.clamp(min=1)).item()
        train_mae = (train_abs_err_sum / train_total_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_abs_err_sum = torch.zeros((), device=DEVICE)
        val_total_sum = torch.zeros((), device=DEVICE)
        val_progress = ProgressTracker(len(val_loader), epoch, "encoder_regression/val", min_interval=args.progress_interval) if is_main_process() else None
        with torch.no_grad():
            for tokens, targets in val_loader:
                tokens = tokens.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                with autocast_ctx(args):
                    preds = model(tokens, task="regress").squeeze(-1)
                    loss = criterion(preds, targets)
                bs = tokens.size(0)
                val_loss_sum += loss.detach() * bs
                val_abs_err_sum += (preds.detach() - targets).abs().sum()
                val_total_sum += bs
                if val_progress is not None:
                    val_progress.update(mse=val_loss_sum / val_total_sum.clamp(min=1),
                                         mae=val_abs_err_sum / val_total_sum.clamp(min=1))
        if val_progress is not None:
            val_progress.finish()
        reduce_sum(val_loss_sum)
        reduce_sum(val_abs_err_sum)
        reduce_sum(val_total_sum)
        val_loss = (val_loss_sum / val_total_sum.clamp(min=1)).item()
        val_mae = (val_abs_err_sum / val_total_sum.clamp(min=1)).item()

        # MAE reported in standardized units (same scale the model was trained on) -
        # multiply by meta["target_std"] to convert back to the original label units.
        if is_main_process():
            print(f"[encoder/regression] epoch {epoch:02d} | train_loss(MSE) {train_loss:.4f} "
                  f"mae {train_mae:.4f} | val_loss(MSE) {val_loss:.4f} mae {val_mae:.4f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.6f} | {time.time()-start:.1f}s")

            save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
                print(f"  -> new best (val_loss={val_loss:.4f})")
        if IS_DISTRIBUTED:
            dist.barrier()


# =========================================================================== #
# MODE 3b: encoder-only, mlm
# =========================================================================== #
class MLMDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def mask_tokens(batch, mask_idx, vocab_size, pad_idx, mlm_prob=0.15):
    """BERT-style dynamic masking: 80% -> [MASK], 10% random token, 10% unchanged.
    Returns (input_ids, labels) where labels is -100 (ignore) at non-masked positions."""
    device = batch.device
    labels = batch.clone()
    prob_matrix = torch.full(batch.shape, mlm_prob, device=device)
    prob_matrix.masked_fill_(batch == pad_idx, 0.0)  # never mask padding
    masked_positions = torch.bernoulli(prob_matrix).bool()

    labels[~masked_positions] = -100  # only compute loss on masked positions

    # 80% of the time, replace with [MASK]
    replace_mask = torch.bernoulli(torch.full(batch.shape, 0.8, device=device)).bool() & masked_positions
    batch = batch.clone()
    batch[replace_mask] = mask_idx

    # 10% of the time, replace with a random token
    random_mask = (
        torch.bernoulli(torch.full(batch.shape, 0.5, device=device)).bool() & masked_positions & ~replace_mask
    )
    random_tokens = torch.randint(0, vocab_size, batch.shape, dtype=torch.long, device=device)
    batch[random_mask] = random_tokens[random_mask]

    # remaining 10%: left unchanged
    return batch, labels


def collate_mlm(batch, pad_idx):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


def train_encoder_mlm(args):
    data_dir = os.path.join(args.data_root, "encoder_mlm")
    ckpt_dir = os.path.join(args.ckpt_root, "encoder_mlm")

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    pad_idx, mask_idx = meta["pad_idx"], meta["mask_idx"]
    max_len = meta["block_size"] + 8

    train_ds = MLMDataset(os.path.join(data_dir, "train.pkl"))
    val_ds = MLMDataset(os.path.join(data_dir, "val.pkl"))

    collate = lambda b: collate_mlm(b, pad_idx)
    train_loader, train_sampler = make_dataloader(train_ds, args.batch_size, True, collate, args)
    val_loader, _ = make_dataloader(val_ds, args.batch_size, False, collate, args)

    resume_ckpt = load_resume(args.resume)
    arch = resume_ckpt["config"] if resume_ckpt else resolve_arch_args(args)

    model = TransformerEncoderOnly(
        vocab_size=meta["vocab_size"], d_model=arch["d_model"], num_layers=arch["num_layers"],
        num_heads=arch["num_heads"], num_kv_heads=arch["num_kv_heads"], d_ff=arch["d_ff"],
        max_len=max_len, dropout=arch["dropout"], pad_idx=pad_idx, rope_theta=arch["rope_theta"],
    )
    model = prepare_model_for_device(model)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, arch["d_model"], args.warmup_steps)
    scaler = make_grad_scaler(args)

    config = {
        "mode": "encoder_mlm", "d_model": arch["d_model"], "num_layers": arch["num_layers"],
        "num_heads": arch["num_heads"], "num_kv_heads": arch["num_kv_heads"], "d_ff": arch["d_ff"],
        "dropout": arch["dropout"], "rope_theta": arch["rope_theta"],
        "vocab_size": meta["vocab_size"], "max_len": max_len, "pad_idx": pad_idx,
    }

    start_epoch = 1
    best_val = float("inf")
    if resume_ckpt is not None:
        load_model_state(model, resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.step_num = resume_ckpt.get("scheduler_step", 0)
        start_epoch = resume_ckpt["epoch"] + 1
        best_val = resume_ckpt["metric"]
        if is_main_process():
            print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_masked_sum = torch.zeros((), device=DEVICE)
        progress = ProgressTracker(len(train_loader), epoch, "encoder_mlm/train", min_interval=args.progress_interval) if is_main_process() else None
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            inputs, labels = mask_tokens(batch, mask_idx, meta["vocab_size"], pad_idx)

            with autocast_ctx(args):
                logits = model(inputs, task="mlm")
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            lr = scheduler.step()
            backward_step(loss, model, optimizer, scaler, args.grad_clip)


            n_masked = (labels != -100).sum()
            train_loss_sum += loss.detach() * n_masked
            train_masked_sum += n_masked
            if progress is not None:
                progress.update(loss=train_loss_sum / train_masked_sum.clamp(min=1), lr=lr)
        if progress is not None:
            progress.finish()
        reduce_sum(train_loss_sum)
        reduce_sum(train_masked_sum)
        train_loss = (train_loss_sum / train_masked_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_masked_sum = torch.zeros((), device=DEVICE)
        val_progress = ProgressTracker(len(val_loader), epoch, "encoder_mlm/val", min_interval=args.progress_interval) if is_main_process() else None
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE, non_blocking=True)
                inputs, labels = mask_tokens(batch, mask_idx, meta["vocab_size"], pad_idx)
                with autocast_ctx(args):
                    logits = model(inputs, task="mlm")
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                n_masked = (labels != -100).sum()
                val_loss_sum += loss.detach() * n_masked
                val_masked_sum += n_masked
                if val_progress is not None:
                    val_progress.update(loss=val_loss_sum / val_masked_sum.clamp(min=1))
        if val_progress is not None:
            val_progress.finish()
        reduce_sum(val_loss_sum)
        reduce_sum(val_masked_sum)
        val_loss = (val_loss_sum / val_masked_sum.clamp(min=1)).item()

        if is_main_process():
            print(f"[encoder/mlm] epoch {epoch:02d} | train_loss {train_loss:.4f} "
                  f"| val_loss {val_loss:.4f} | {time.time()-start:.1f}s")

            save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
                print(f"  -> new best (val_loss={val_loss:.4f})")
        if IS_DISTRIBUTED:
            dist.barrier()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["encdec", "decoder", "encoder"])
    p.add_argument("--task", default="classify", choices=["classify", "mlm", "regression"],
                   help="only used when --mode encoder")

    p.add_argument("--data-root", default="data", dest="data_root")
    p.add_argument("--ckpt-root", default="checkpoints", dest="ckpt_root")
    p.add_argument("--resume", default=None,
                   help="path to a checkpoint to resume from, e.g. checkpoints/decoder/last.pt "
                        "(architecture flags below are ignored when resuming - the checkpoint's "
                        "own config is used instead)")

    p.add_argument("--d-model", type=int, default=256, dest="d_model")
    p.add_argument("--num-layers", type=int, default=4, dest="num_layers")
    p.add_argument("--num-heads", type=int, default=8, dest="num_heads")
    p.add_argument("--num-kv-heads", type=int, default=None, dest="num_kv_heads",
                   help="Grouped-Query-Attention: number of key/value heads (must divide "
                        "--num-heads). Defaults to a ~4:1 query:kv ratio if omitted; pass "
                        "the same value as --num-heads to get plain multi-head attention.")
    p.add_argument("--d-ff", type=int, default=1024, dest="d_ff")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rope-theta", type=float, default=10000.0, dest="rope_theta",
                   help="base frequency for the Rotary Position Embedding table")

    p.add_argument("--batch-size", type=int, default=64, dest="batch_size",
                   help="per-process batch size. Under torchrun with N processes, the "
                        "effective global batch size is batch_size * N, since each process "
                        "pulls its own batch from its own DistributedSampler shard.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4, dest="num_workers",
                   help="DataLoader background worker processes. With 0, batch "
                        "collation runs on the main process between training steps and "
                        "the GPU sits idle waiting for it - usually the real reason a "
                        "faster GPU doesn't train faster on a small model like this one.")
    p.add_argument("--amp", action="store_true",
                   help="Enable automatic mixed precision (fp16 autocast + gradient "
                        "scaling). This is what actually engages Tensor Cores on "
                        "Turing/Ampere+ GPUs (e.g. T4, A100); without it those cores sit "
                        "unused and training runs in plain fp32. A GTX 1660 Super has no "
                        "Tensor Cores, so it benefits much less from this flag.")
    p.add_argument("--warmup-steps", type=int, default=4000, dest="warmup_steps",
                   help="only used by encdec / decoder / encoder-mlm (Noam schedule)")
    p.add_argument("--label-smoothing", type=float, default=0.1, dest="label_smoothing")
    p.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip")
    p.add_argument("--pooling", default="mean", choices=["mean", "cls"],
                   help="only used when --mode encoder --task classify/regression")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="flat AdamW learning rate, only used when --mode encoder --task classify/regression")
    p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay",
                   help="AdamW weight decay, only used when --mode encoder --task classify/regression")
    p.add_argument("--progress-interval", type=float, default=2.0, dest="progress_interval",
                   help="minimum seconds between within-epoch progress line updates (default 2.0). "
                        "Lower it for more frequent updates, raise it to print less often.")

    return p


def main():
    args = build_arg_parser().parse_args()
    setup_distributed()
    try:
        if args.mode == "encdec":
            train_encdec(args)
        elif args.mode == "decoder":
            train_decoder(args)
        elif args.mode == "encoder" and args.task == "classify":
            train_encoder_classify(args)
        elif args.mode == "encoder" and args.task == "mlm":
            train_encoder_mlm(args)
        elif args.mode == "encoder" and args.task == "regression":
            train_encoder_regression(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()