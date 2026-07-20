"""
train.py
--------
Trains whichever architecture in model.py matches your data, produced by
data_cleaning.py. Pick a mode:

    python train.py --mode encdec                    # Transformer (encoder-decoder)
    python train.py --mode decoder                   # TransformerDecoderOnly (GPT-style)
    python train.py --mode encoder --task classify    # TransformerEncoderOnly (classification)
    python train.py --mode encoder --task mlm         # TransformerEncoderOnly (masked-LM pretrain)

Each mode reads from ./data/<matching_subfolder>/ (created by data_cleaning.py)
and writes checkpoints to ./checkpoints/<mode_name>/{last,best}.pt.

Shared training recipe (matches the original paper) for encdec / decoder / encoder-mlm:
  - Adam (betas=0.9, 0.98, eps=1e-9) + Noam warmup/decay LR schedule
  - label smoothing
  - gradient clipping

encoder --task classify is the exception: it uses a flat AdamW (--lr, --weight-decay)
instead of Noam. Noam's peak lr scales as d_model**-0.5 * warmup_steps**-0.5, and with a
small d_model / small dataset / short warmup (as classification typically uses) that peak
is reached within the first few epochs and is high enough to blow the model up into a
degenerate "predicts everything as 50/50" state (train_loss stuck at ln(2)). AdamW with a
small flat lr is the standard recipe for training/fine-tuning small Transformer
classifiers and avoids that failure mode.
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
from torch.utils.data import Dataset, DataLoader

from model import Transformer, TransformerDecoderOnly, TransformerEncoderOnly, default_num_kv_heads

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Multi-GPU helpers (nn.DataParallel)
# --------------------------------------------------------------------------- #
def prepare_model_for_device(model):
    """Moves model to DEVICE and, if more than one CUDA GPU is visible, wraps it in
    nn.DataParallel so a single process fans batches out across all of them. On CPU
    or a single-GPU machine this just moves the model over and returns it unwrapped.
    Checkpoints always store the *unwrapped* state dict (see save_checkpoint /
    load_model_state), so switching between 1-GPU and multi-GPU machines across runs
    is transparent."""
    model = model.to(DEVICE)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        print(f"Found {n_gpus} GPUs -> wrapping model in nn.DataParallel")
        model = nn.DataParallel(model)
    elif n_gpus == 1:
        print("Found 1 GPU -> training on a single GPU (no DataParallel)")
    else:
        print("No GPU found -> training on CPU")
    return model


def raw_model(model):
    """Returns the underlying module whether or not `model` is DataParallel-wrapped.
    Use this whenever you need to reach model-specific attributes/methods that
    DataParallel doesn't proxy."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_model_state(model, state_dict):
    """Loads a checkpoint's (always-unwrapped) state dict into `model`, whether or
    not `model` is currently DataParallel-wrapped."""
    raw_model(model).load_state_dict(state_dict)


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
    loader_kwargs = dataloader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **loader_kwargs)

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
        print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_tokens_sum = torch.zeros((), device=DEVICE)
        for src, tgt in train_loader:
            src = src.to(DEVICE, non_blocking=True)
            tgt = tgt.to(DEVICE, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            with autocast_ctx(args):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            backward_step(loss, model, optimizer, scaler, args.grad_clip)
            scheduler.step()

            # accumulate on-device and only sync to CPU once, after the epoch -
            # a per-batch .item() here would force a GPU sync every single step
            n_tok = (tgt_out != pad_idx).sum()
            train_loss_sum += loss.detach() * n_tok
            train_tokens_sum += n_tok
        train_loss = (train_loss_sum / train_tokens_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_tokens_sum = torch.zeros((), device=DEVICE)
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
        val_loss = (val_loss_sum / val_tokens_sum.clamp(min=1)).item()

        print(f"[encdec] epoch {epoch:02d} | train_loss {train_loss:.4f} "
              f"(ppl {math.exp(min(train_loss, 20)):.2f}) | val_loss {val_loss:.4f} "
              f"(ppl {math.exp(min(val_loss, 20)):.2f}) | {time.time()-start:.1f}s")

        save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
            print(f"  -> new best (val_loss={val_loss:.4f})")


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
    loader_kwargs = dataloader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **loader_kwargs)

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
        print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_tokens_sum = torch.zeros((), device=DEVICE)
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]  # next-token prediction

            with autocast_ctx(args):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            backward_step(loss, model, optimizer, scaler, args.grad_clip)
            scheduler.step()

            n_tok = (y != pad_idx).sum()
            train_loss_sum += loss.detach() * n_tok
            train_tokens_sum += n_tok
        train_loss = (train_loss_sum / train_tokens_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_tokens_sum = torch.zeros((), device=DEVICE)
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
        val_loss = (val_loss_sum / val_tokens_sum.clamp(min=1)).item()

        print(f"[decoder] epoch {epoch:02d} | train_loss {train_loss:.4f} "
              f"(ppl {math.exp(min(train_loss, 20)):.2f}) | val_loss {val_loss:.4f} "
              f"(ppl {math.exp(min(val_loss, 20)):.2f}) | {time.time()-start:.1f}s")

        save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
            print(f"  -> new best (val_loss={val_loss:.4f})")


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
    loader_kwargs = dataloader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **loader_kwargs)

    resume_ckpt = load_resume(args.resume)
    arch = resume_ckpt["config"] if resume_ckpt else {**resolve_arch_args(args), "pooling": args.pooling}

    model = TransformerEncoderOnly(
        vocab_size=meta["vocab_size"], d_model=arch["d_model"], num_layers=arch["num_layers"],
        num_heads=arch["num_heads"], num_kv_heads=arch["num_kv_heads"], d_ff=arch["d_ff"],
        max_len=meta["max_len"] + 10, dropout=arch["dropout"], pad_idx=pad_idx,
        num_classes=meta["num_classes"], pooling=arch["pooling"], rope_theta=arch["rope_theta"],
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
    }

    start_epoch = 1
    best_val_acc = 0.0
    if resume_ckpt is not None:
        load_model_state(model, resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_acc = resume_ckpt["metric"]
        print(f"Resumed at epoch {start_epoch} (previous val_acc={best_val_acc:.3f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_correct_sum = torch.zeros((), device=DEVICE)
        train_total = 0
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
            train_total += bs
        train_loss = (train_loss_sum / max(train_total, 1)).item()
        train_acc = (train_correct_sum / max(train_total, 1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_correct_sum = torch.zeros((), device=DEVICE)
        val_total = 0
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
                val_total += bs
        val_loss = (val_loss_sum / max(val_total, 1)).item()
        val_acc = (val_correct_sum / max(val_total, 1)).item()

        print(f"[encoder/classify] epoch {epoch:02d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | lr {optimizer.param_groups[0]['lr']:.6f} "
              f"| {time.time()-start:.1f}s")

        save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_acc)
            print(f"  -> new best (val_acc={val_acc:.3f})")


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
    loader_kwargs = dataloader_kwargs(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, **loader_kwargs)

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
        print(f"Resumed at epoch {start_epoch} (previous val_loss={best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_masked_sum = torch.zeros((), device=DEVICE)
        for batch in train_loader:
            batch = batch.to(DEVICE, non_blocking=True)
            inputs, labels = mask_tokens(batch, mask_idx, meta["vocab_size"], pad_idx)

            with autocast_ctx(args):
                logits = model(inputs, task="mlm")
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            backward_step(loss, model, optimizer, scaler, args.grad_clip)
            scheduler.step()

            n_masked = (labels != -100).sum()
            train_loss_sum += loss.detach() * n_masked
            train_masked_sum += n_masked
        train_loss = (train_loss_sum / train_masked_sum.clamp(min=1)).item()

        model.eval()
        val_loss_sum = torch.zeros((), device=DEVICE)
        val_masked_sum = torch.zeros((), device=DEVICE)
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
        val_loss = (val_loss_sum / val_masked_sum.clamp(min=1)).item()

        print(f"[encoder/mlm] epoch {epoch:02d} | train_loss {train_loss:.4f} "
              f"| val_loss {val_loss:.4f} | {time.time()-start:.1f}s")

        save_checkpoint(ckpt_dir, "last.pt", model, optimizer, scheduler, config, epoch, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir, "best.pt", model, optimizer, scheduler, config, epoch, val_loss)
            print(f"  -> new best (val_loss={val_loss:.4f})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["encdec", "decoder", "encoder"])
    p.add_argument("--task", default="classify", choices=["classify", "mlm"],
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

    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
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
                   help="only used when --mode encoder --task classify")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="flat AdamW learning rate, only used when --mode encoder --task classify")
    p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay",
                   help="AdamW weight decay, only used when --mode encoder --task classify")

    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "encdec":
        train_encdec(args)
    elif args.mode == "decoder":
        train_decoder(args)
    elif args.mode == "encoder" and args.task == "classify":
        train_encoder_classify(args)
    elif args.mode == "encoder" and args.task == "mlm":
        train_encoder_mlm(args)


if __name__ == "__main__":
    main()