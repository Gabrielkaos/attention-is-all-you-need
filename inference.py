"""
inference.py
------------
Loads a checkpoint from train.py + the matching tokenizer(s) from
data_cleaning.py, and lets you play with whichever model you trained.

    python inference.py --mode encdec                              # translate, REPL
    python inference.py --mode encdec --beam 5                     # beam search
    python inference.py --mode decoder                              # generate text, REPL
    python inference.py --mode decoder --text "once upon a"        # one-shot
    python inference.py --mode encoder_classify                     # classify, REPL
    python inference.py --mode encoder_mlm                          # fill-in-the-blank, REPL
                                                                      (use ___ as the blank)

Every mode reuses clean_text + the BPE/char tokenizer classes from
data_cleaning.py, so preprocessing at inference time matches training exactly.

Note on encoder_mlm with different tokenizers: a "___" blank always becomes
exactly one masked token, but what that token *means* depends on --tokenizer
in data_cleaning.py - with bpe it's usually a whole word or word-piece, with
char it's a single character. Char-mode fill-in-the-blank is less useful for
guessing a whole missing word (use several consecutive "___" to mask more
characters at once).
"""

import os
import re
import json
import argparse

import torch

from model import Transformer, TransformerDecoderOnly, TransformerEncoderOnly
from data_cleaning import clean_text, load_tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CKPT = {
    "encdec": os.path.join("checkpoints", "encdec", "best.pt"),
    "decoder": os.path.join("checkpoints", "decoder", "best.pt"),
    "encoder_classify": os.path.join("checkpoints", "encoder_classify", "best.pt"),
    "encoder_mlm": os.path.join("checkpoints", "encoder_mlm", "best.pt"),
}
DEFAULT_DATA_DIR = {
    "encdec": os.path.join("data", "encdec"),
    "decoder": os.path.join("data", "decoder"),
    "encoder_classify": os.path.join("data", "encoder_classify"),
    "encoder_mlm": os.path.join("data", "encoder_mlm"),
}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def load_checkpoint(ckpt_path, expected_mode):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ckpt["config"]
    if cfg.get("mode") != expected_mode:
        print(f"warning: checkpoint was trained as mode='{cfg.get('mode')}', "
              f"but you asked for '{expected_mode}' - loading anyway.")
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, metric={ckpt['metric']:.4f})")
    return ckpt, cfg


# =========================================================================== #
# MODE: encdec (translation)
# =========================================================================== #
def build_encdec_model(cfg):
    return Transformer(
        src_vocab_size=cfg["src_vocab_size"], tgt_vocab_size=cfg["tgt_vocab_size"],
        d_model=cfg["d_model"], num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
        num_kv_heads=cfg.get("num_kv_heads"), d_ff=cfg["d_ff"], max_len=cfg["max_len"],
        dropout=cfg["dropout"], pad_idx=cfg["pad_idx"], rope_theta=cfg.get("rope_theta", 10000.0),
    ).to(DEVICE)


@torch.no_grad()
def greedy_translate(model, src_ids, sos_idx, eos_idx, max_len=100):
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    enc_out, src_mask = model.encode(src)
    tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len):
        logits = model.decode(tgt, enc_out, src_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == eos_idx:
            break
    return tgt.squeeze(0).tolist()


@torch.no_grad()
def beam_translate(model, src_ids, sos_idx, eos_idx, beam_size=5, max_len=100, length_penalty=0.7):
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    enc_out, src_mask = model.encode(src)

    beams = [([sos_idx], 0.0, False)]
    for _ in range(max_len):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, finished))
                continue
            tgt = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
            logits = model.decode(tgt, enc_out, src_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            top_lp, top_idx = log_probs.topk(beam_size)
            for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                candidates.append((tokens + [idx], score + lp, idx == eos_idx))

        candidates.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True)
        beams = candidates[:beam_size]
        if all(b[2] for b in beams):
            break

    return beams[0][0]


def run_encdec(args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR["encdec"]
    ckpt_path = args.ckpt or DEFAULT_CKPT["encdec"]

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    src_tokenizer = load_tokenizer(os.path.join(data_dir, "src_tokenizer.json"))
    tgt_tokenizer = load_tokenizer(os.path.join(data_dir, "tgt_tokenizer.json"))

    ckpt, cfg = load_checkpoint(ckpt_path, "encdec")
    model = build_encdec_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sos_idx, eos_idx = meta["sos_idx"], meta["eos_idx"]

    def translate_once(sentence):
        src_ids = src_tokenizer.encode(clean_text(sentence))
        gen_max_len = cfg["max_len"] - 1  # leave room for the token itself within positional capacity
        if args.beam > 0:
            out_ids = beam_translate(model, src_ids, sos_idx, eos_idx, beam_size=args.beam, max_len=gen_max_len)
        else:
            out_ids = greedy_translate(model, src_ids, sos_idx, eos_idx, max_len=gen_max_len)
        return tgt_tokenizer.decode(out_ids)

    if args.text:
        print(translate_once(args.text))
        return

    mode_desc = f"beam search (k={args.beam})" if args.beam > 0 else "greedy"
    print(f"\nTranslate {meta['src_lang']} -> {meta['tgt_lang']} ({mode_desc}). Type 'quit' to exit.\n")
    while True:
        sentence = input(f"[{meta['src_lang']}] > ").strip()
        if sentence.lower() in ("quit", "exit"):
            break
        if not sentence:
            continue
        print(f"[{meta['tgt_lang']}] > {translate_once(sentence)}\n")


# =========================================================================== #
# MODE: decoder (GPT-style generation)
# =========================================================================== #
def build_decoder_model(cfg):
    return TransformerDecoderOnly(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], num_kv_heads=cfg.get("num_kv_heads"), d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"], pad_idx=cfg["pad_idx"],
        rope_theta=cfg.get("rope_theta", 10000.0),
    ).to(DEVICE)


def run_decoder(args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR["decoder"]
    ckpt_path = args.ckpt or DEFAULT_CKPT["decoder"]

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(os.path.join(data_dir, "tokenizer.json"))

    ckpt, cfg = load_checkpoint(ckpt_path, "decoder")
    model = build_decoder_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eos_idx = meta["eos_idx"]

    def generate_once(prompt):
        import time
        prompt_ids = tokenizer.encode(clean_text(prompt))
        if not prompt_ids:
            prompt_ids = [meta["sos_idx"]]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        t0 = time.perf_counter()
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_k=args.top_k, eos_idx=eos_idx,
                              use_cache=not args.no_kv_cache)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_new = out.size(1) - idx.size(1)
        mode = "no-cache" if args.no_kv_cache else "kv-cache"
        print(f"  [{mode}: {n_new} new tokens in {elapsed:.3f}s = {n_new / max(elapsed, 1e-9):.1f} tok/s]")
        return tokenizer.decode(out.squeeze(0).tolist())

    if args.text:
        print(generate_once(args.text))
        return

    print(f"\nGenerate text (temperature={args.temperature}, top_k={args.top_k}). Type 'quit' to exit.\n")
    while True:
        prompt = input("prompt > ").strip()
        if prompt.lower() in ("quit", "exit"):
            break
        print(generate_once(prompt), "\n")


# =========================================================================== #
# MODE: encoder_classify
# =========================================================================== #
def build_classify_model(cfg):
    return TransformerEncoderOnly(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], num_kv_heads=cfg.get("num_kv_heads"), d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"], pad_idx=cfg["pad_idx"],
        num_classes=cfg["num_classes"], pooling=cfg.get("pooling", "mean"),
        rope_theta=cfg.get("rope_theta", 10000.0),
    ).to(DEVICE)


def run_encoder_classify(args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR["encoder_classify"]
    ckpt_path = args.ckpt or DEFAULT_CKPT["encoder_classify"]

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(os.path.join(data_dir, "tokenizer.json"))

    ckpt, cfg = load_checkpoint(ckpt_path, "encoder_classify")
    model = build_classify_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    label_names = cfg.get("label_names") or [str(i) for i in range(cfg["num_classes"])]
    unk_idx = meta["unk_idx"]
    sos_idx = meta["sos_idx"]

    @torch.no_grad()
    def classify_once(sentence):
        ids = tokenizer.encode(clean_text(sentence))
        if not ids:
            ids = [unk_idx]
        ids = [sos_idx] + ids
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        logits = model(x, task="classify")
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = probs.argmax().item()
        return label_names[pred], probs[pred].item(), probs

    if args.text:
        label, conf, _ = classify_once(args.text)
        print(f"{label}  (confidence {conf:.3f})")
        return

    print(f"\nClassify text into: {label_names}. Type 'quit' to exit.\n")
    while True:
        sentence = input("text > ").strip()
        if sentence.lower() in ("quit", "exit"):
            break
        if not sentence:
            continue
        label, conf, probs = classify_once(sentence)
        print(f"  -> {label}  (confidence {conf:.3f})")
        print(f"     full distribution: " +
              ", ".join(f"{n}={p:.3f}" for n, p in zip(label_names, probs.tolist())) + "\n")


# =========================================================================== #
# MODE: encoder_mlm (fill-in-the-blank)
# =========================================================================== #
def build_mlm_model(cfg):
    return TransformerEncoderOnly(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], num_kv_heads=cfg.get("num_kv_heads"), d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"], pad_idx=cfg["pad_idx"],
        rope_theta=cfg.get("rope_theta", 10000.0),
    ).to(DEVICE)


_BLANK_RE = re.compile(r"___|<mask>|\[mask\]", re.IGNORECASE)


def run_encoder_mlm(args):
    data_dir = args.data_dir or DEFAULT_DATA_DIR["encoder_mlm"]
    ckpt_path = args.ckpt or DEFAULT_CKPT["encoder_mlm"]

    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(os.path.join(data_dir, "tokenizer.json"))

    ckpt, cfg = load_checkpoint(ckpt_path, "encoder_mlm")
    model = build_mlm_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mask_idx = meta["mask_idx"]

    @torch.no_grad()
    def fill_once(sentence, top_k=5):
        # split on the blank marker(s) so each "___" becomes exactly one
        # <mask> token id, regardless of how the surrounding text tokenizes -
        # this also means one blank can only ever be filled by ONE vocab
        # entry. If the true word got BPE-split into multiple pieces (e.g.
        # "birthday" -> "birth" + "day"), a single blank can only produce one
        # of those pieces, never the whole word - use consecutive blanks
        # ("___ ___") to give it room for a multi-piece word.
        parts = _BLANK_RE.split(sentence)
        if len(parts) == 1:
            print("  (no blank found - use ___ where you want a prediction)")
            return

        working_ids, mask_positions = [], []
        for i, part in enumerate(parts):
            if part:
                working_ids.extend(tokenizer.encode(clean_text(part)))
            if i < len(parts) - 1:
                mask_positions.append(len(working_ids))
                working_ids.append(mask_idx)

        # Iteratively resolve the single most-confident blank first, then
        # substitute its predicted token back into the input before
        # predicting the rest. A plain one-shot pass predicts every blank
        # independently, only ever looking at the original masked context -
        # so for "Happy ___ ___ to you!" it never gets to use its own guess
        # for one blank while filling the other. Resolving one at a time lets
        # e.g. "day" (predicted first) inform the "birth"/"day" -> "birthday"
        # guess for the neighboring blank, once it's substituted back in.
        remaining = list(mask_positions)
        resolved = {}
        while remaining:
            x = torch.tensor([working_ids], dtype=torch.long, device=DEVICE)
            probs = torch.softmax(model(x, task="mlm")[0], dim=-1)  # (seq_len, vocab)

            best_pos, best_conf, best_id, best_topk = None, -1.0, None, None
            for pos in remaining:
                top_vals, top_idx = probs[pos].topk(top_k)
                conf = top_vals[0].item()
                if conf > best_conf:
                    best_pos, best_conf = pos, conf
                    best_id = top_idx[0].item()
                    best_topk = [(tokenizer.itos[i], v)
                                 for i, v in zip(top_idx.tolist(), top_vals.tolist())]

            working_ids[best_pos] = best_id
            resolved[best_pos] = best_topk
            remaining.remove(best_pos)

        for pos in mask_positions:
            preds = resolved[pos]
            print(f"  blank at position {pos}: " +
                  ", ".join(f"{repr(w)} ({p:.3f})" for w, p in preds))

        print(f"  filled in: {tokenizer.decode(working_ids)!r}")

    if args.text:
        fill_once(args.text, top_k=args.top_k or 5)
        return

    print("\nFill-in-the-blank. Use ___ where you want a prediction. Type 'quit' to exit.")
    print("(with --tokenizer char in data_cleaning.py, each ___ predicts a single character,")
    print(" not a whole word - use several ___ in a row for a longer guess.)\n")
    while True:
        sentence = input("text > ").strip()
        if sentence.lower() in ("quit", "exit"):
            break
        if not sentence:
            continue
        fill_once(sentence, top_k=args.top_k or 5)
        print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["encdec", "decoder", "encoder_classify", "encoder_mlm"])
    p.add_argument("--ckpt", default=None, help="defaults to checkpoints/<mode>/best.pt")
    p.add_argument("--data-dir", default=None, dest="data_dir",
                   help="defaults to data/<mode>/")
    p.add_argument("--text", default=None, help="run once on this input and exit (no REPL)")

    # encdec
    p.add_argument("--beam", type=int, default=0, help="beam size for encdec, 0 = greedy")

    # decoder
    p.add_argument("--max-new-tokens", type=int, default=60, dest="max_new_tokens")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, dest="top_k")
    p.add_argument("--no-kv-cache", action="store_true", dest="no_kv_cache",
                   help="disable the KV-cache and recompute the full forward pass every "
                        "step instead (slower - mainly useful for A/B timing the cache itself)")

    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "encdec":
        run_encdec(args)
    elif args.mode == "decoder":
        run_decoder(args)
    elif args.mode == "encoder_classify":
        run_encoder_classify(args)
    elif args.mode == "encoder_mlm":
        run_encoder_mlm(args)


if __name__ == "__main__":
    main()