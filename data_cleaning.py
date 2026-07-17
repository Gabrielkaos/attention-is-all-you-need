"""
data_cleaning.py
-----------------
Prepares data for whichever model architecture in model.py you're training.
Three modes, matching train.py's three modes:

  --mode encdec
      Translation pairs (src/tgt) for the full encoder-decoder Transformer.
      Default source: Helsinki-NLP/opus_books (en-fr) from Hugging Face.

  --mode decoder
      Plain text corpus for the GPT-style TransformerDecoderOnly - next-token
      prediction. Two sub-modes via --unit:
        stream : one long corpus chunked into fixed-length blocks
                 (e.g. Tiny Shakespeare - default, auto-downloaded)
        lines  : one example per line, independently padded
                 (e.g. a names.txt / poems.txt file you supply with --source)

  --mode encoder --task classify
      Labeled classification data for TransformerEncoderOnly.
      Default source: HF `ag_news` (4-class news topic classification).

  --mode encoder --task mlm
      Unlabeled text corpus for masked-language-model pretraining of
      TransformerEncoderOnly (same chunking as decoder/stream - masking
      itself happens dynamically in train.py, not here).

Every mode writes to its own subfolder under ./data/<mode_name>/ with
train.pkl / val.pkl / test.pkl, vocab json(s), and a meta.json describing
everything train.py needs to reconstruct the right model.

Examples:
    python data_cleaning.py --mode encdec
    python data_cleaning.py --mode decoder --unit stream            # Tiny Shakespeare
    python data_cleaning.py --mode decoder --unit lines --source names.txt
    python data_cleaning.py --mode encoder --task classify
    python data_cleaning.py --mode encoder --task mlm
"""

import os
import re
import json
import pickle
import random
import argparse
import urllib.request
from collections import Counter

from tqdm import tqdm
from datasets import load_dataset

# --------------------------------------------------------------------------- #
# Shared config / specials
# --------------------------------------------------------------------------- #
MIN_FREQ = 2
VAL_SPLIT = 0.02
TEST_SPLIT = 0.02

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>", "<mask>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, MASK_IDX = 0, 1, 2, 3, 4

random.seed(42)

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


# --------------------------------------------------------------------------- #
# Text cleaning / tokenizing (shared by every mode so preprocessing is
# identical between data_cleaning.py and inference.py)
# --------------------------------------------------------------------------- #
_WHITESPACE_RE = re.compile(r"\s+")
_KEEP_CHARS_RE = re.compile(r"[^a-zA-ZÀ-ÿ0-9.,!?'\-\s]")


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = _KEEP_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str):
    text = re.sub(r"([.,!?'\-])", r" \1 ", text)
    return [tok for tok in text.split() if tok]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    def __init__(self, counter: Counter, min_freq: int, specials):
        self.itos = list(specials)
        for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(tok, UNK_IDX) for tok in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i < len(self.itos)]

    def __len__(self):
        return len(self.itos)


def build_vocab(token_lists, min_freq):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    return Vocab(counter, min_freq, SPECIALS)


def split_train_val_test(data):
    random.shuffle(data)
    n = len(data)
    n_val = max(1, int(n * VAL_SPLIT))
    n_test = max(1, int(n * TEST_SPLIT))
    val = data[:n_val]
    test = data[n_val:n_val + n_test]
    train = data[n_val + n_test:]
    return train, val, test


def save_split(out_dir, train, val, test):
    os.makedirs(out_dir, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(split, f)


def save_vocab(out_dir, name, vocab: Vocab):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(vocab.itos, f, ensure_ascii=False, indent=2)


def save_meta(out_dir, meta: dict):
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
# Mode 1: encoder-decoder (translation pairs)
# --------------------------------------------------------------------------- #
def prepare_encdec(args):
    out_dir = os.path.join(args.out_root, "encdec")

    print(f"Loading {args.dataset} ({args.lang_pair}) from Hugging Face ...")
    raw = load_dataset(args.dataset, args.lang_pair)["train"]

    print("Cleaning + tokenizing ...")
    src_sentences, tgt_sentences = [], []
    for example in tqdm(raw):
        pair = example["translation"]
        src_toks = tokenize(clean_text(pair[args.src_lang]))
        tgt_toks = tokenize(clean_text(pair[args.tgt_lang]))

        if not (1 <= len(src_toks) <= args.max_len):
            continue
        if not (1 <= len(tgt_toks) <= args.max_len):
            continue

        src_sentences.append(src_toks)
        tgt_sentences.append(tgt_toks)

    print(f"Kept {len(src_sentences)} sentence pairs after filtering.")

    print("Building vocabularies ...")
    src_vocab = build_vocab(src_sentences, MIN_FREQ)
    tgt_vocab = build_vocab(tgt_sentences, MIN_FREQ)
    print(f"src vocab size: {len(src_vocab)} | tgt vocab size: {len(tgt_vocab)}")

    data = []
    for s_toks, t_toks in zip(src_sentences, tgt_sentences):
        s_ids = src_vocab.encode(s_toks)
        t_ids = [SOS_IDX] + tgt_vocab.encode(t_toks) + [EOS_IDX]
        data.append((s_ids, t_ids))

    train, val, test = split_train_val_test(data)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    save_vocab(out_dir, "src_vocab.json", src_vocab)
    save_vocab(out_dir, "tgt_vocab.json", tgt_vocab)
    save_meta(out_dir, {
        "mode": "encdec",
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "max_len": args.max_len,
    })
    print(f"Done. Saved to ./{out_dir}/")


# --------------------------------------------------------------------------- #
# Mode 2: decoder-only (causal LM text corpus)
# --------------------------------------------------------------------------- #
def _download_tiny_shakespeare(dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {dest_path} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest_path)
    return dest_path


def prepare_decoder_stream(args):
    """One long corpus -> chunked into fixed-length blocks for next-token prediction."""
    out_dir = os.path.join(args.out_root, "decoder")

    source_path = args.source
    if source_path is None:
        source_path = _download_tiny_shakespeare(os.path.join("data", "raw", "tinyshakespeare.txt"))

    print(f"Reading corpus from {source_path} ...")
    with open(source_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    print("Cleaning + tokenizing ...")
    all_tokens = []
    for line in tqdm(raw_lines):
        toks = tokenize(clean_text(line))
        if toks:
            all_tokens.extend(toks)
            all_tokens.append("<eos>")  # keep line boundaries as a signal

    print(f"Corpus length: {len(all_tokens)} tokens")

    print("Building vocabulary ...")
    vocab = build_vocab([all_tokens], MIN_FREQ)
    print(f"vocab size: {len(vocab)}")

    ids = vocab.encode(all_tokens)

    block = args.block_size + 1  # +1 so we can shift by one for (input, target)
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]
    print(f"Built {len(chunks)} chunks of block_size={args.block_size}")

    train, val, test = split_train_val_test(chunks)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    save_vocab(out_dir, "vocab.json", vocab)
    save_meta(out_dir, {
        "mode": "decoder", "unit": "stream",
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(vocab),
        "block_size": args.block_size,
    })
    print(f"Done. Saved to ./{out_dir}/")


def prepare_decoder_lines(args):
    """One example per line (names, poems, jokes, ...) -> independently padded sequences."""
    if args.source is None:
        raise ValueError("--unit lines requires --source path/to/file.txt (one example per line)")

    out_dir = os.path.join(args.out_root, "decoder")

    print(f"Reading lines from {args.source} ...")
    with open(args.source, "r", encoding="utf-8") as f:
        raw_lines = [l for l in f.read().splitlines() if l.strip()]

    print("Cleaning + tokenizing ...")
    tokenized = []
    for line in tqdm(raw_lines):
        toks = tokenize(clean_text(line))
        if 1 <= len(toks) <= args.max_len:
            tokenized.append(toks)

    print(f"Kept {len(tokenized)} lines after filtering.")

    print("Building vocabulary ...")
    vocab = build_vocab(tokenized, MIN_FREQ)
    print(f"vocab size: {len(vocab)}")

    data = [[SOS_IDX] + vocab.encode(toks) + [EOS_IDX] for toks in tokenized]

    train, val, test = split_train_val_test(data)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    save_vocab(out_dir, "vocab.json", vocab)
    save_meta(out_dir, {
        "mode": "decoder", "unit": "lines",
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(vocab),
        "max_len": args.max_len + 2,  # + <sos>/<eos>
    })
    print(f"Done. Saved to ./{out_dir}/")


# --------------------------------------------------------------------------- #
# Mode 3a: encoder-only, classify
# --------------------------------------------------------------------------- #
def prepare_encoder_classify(args):
    out_dir = os.path.join(args.out_root, "encoder_classify")

    print(f"Loading {args.dataset} from Hugging Face ...")
    raw = load_dataset(args.dataset)
    train_raw = raw["train"]
    test_raw = raw["test"] if "test" in raw else None

    text_field = args.text_field
    label_field = args.label_field

    label_names = train_raw.features[label_field].names if hasattr(
        train_raw.features[label_field], "names") else None

    def process(split):
        tokenized, labels = [], []
        for ex in tqdm(split):
            toks = tokenize(clean_text(ex[text_field]))
            if 1 <= len(toks) <= args.max_len:
                tokenized.append(toks)
                labels.append(ex[label_field])
        return tokenized, labels

    print("Cleaning + tokenizing train split ...")
    train_toks, train_labels = process(train_raw)

    print("Building vocabulary ...")
    vocab = build_vocab(train_toks, MIN_FREQ)
    print(f"vocab size: {len(vocab)} | num_classes: {len(set(train_labels))}")

    data = [(vocab.encode(toks), label) for toks, label in zip(train_toks, train_labels)]
    train, val, test_from_train = split_train_val_test(data)

    if test_raw is not None:
        print("Cleaning + tokenizing test split ...")
        test_toks, test_labels = process(test_raw)
        test = [(vocab.encode(toks), label) for toks, label in zip(test_toks, test_labels)]
    else:
        test = test_from_train

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    save_vocab(out_dir, "vocab.json", vocab)
    save_meta(out_dir, {
        "mode": "encoder", "task": "classify",
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(vocab),
        "num_classes": len(label_names) if label_names else len(set(train_labels)),
        "label_names": label_names,
        "max_len": args.max_len,
    })
    print(f"Done. Saved to ./{out_dir}/")


# --------------------------------------------------------------------------- #
# Mode 3b: encoder-only, mlm (masked language model pretraining)
# --------------------------------------------------------------------------- #
def prepare_encoder_mlm(args):
    """Same chunking as decoder/stream - masking itself is dynamic, done in train.py."""
    out_dir = os.path.join(args.out_root, "encoder_mlm")

    source_path = args.source
    if source_path is None:
        source_path = _download_tiny_shakespeare(os.path.join("data", "raw", "tinyshakespeare.txt"))

    print(f"Reading corpus from {source_path} ...")
    with open(source_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    print("Cleaning + tokenizing ...")
    all_tokens = []
    for line in tqdm(raw_lines):
        toks = tokenize(clean_text(line))
        if toks:
            all_tokens.extend(toks)
            all_tokens.append("<eos>")

    print("Building vocabulary ...")
    vocab = build_vocab([all_tokens], MIN_FREQ)
    print(f"vocab size: {len(vocab)}")

    ids = vocab.encode(all_tokens)
    block = args.block_size
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]
    print(f"Built {len(chunks)} chunks of block_size={block}")

    train, val, test = split_train_val_test(chunks)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    save_vocab(out_dir, "vocab.json", vocab)
    save_meta(out_dir, {
        "mode": "encoder", "task": "mlm",
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(vocab),
        "block_size": block,
    })
    print(f"Done. Saved to ./{out_dir}/")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["encdec", "decoder", "encoder"])
    p.add_argument("--task", default="classify", choices=["classify", "mlm"],
                   help="only used when --mode encoder")
    p.add_argument("--unit", default="stream", choices=["stream", "lines"],
                   help="only used when --mode decoder")
    p.add_argument("--out-root", default="data")
    p.add_argument("--max-len", type=int, default=100, dest="max_len")
    p.add_argument("--block-size", type=int, default=64, dest="block_size",
                   help="sequence length for decoder/stream and encoder/mlm chunking")

    # encdec-specific
    p.add_argument("--dataset", default=None,
                   help="HF dataset name (defaults depend on mode/task)")
    p.add_argument("--lang-pair", default="en-fr", dest="lang_pair")
    p.add_argument("--src-lang", default="en", dest="src_lang")
    p.add_argument("--tgt-lang", default="fr", dest="tgt_lang")

    # decoder/lines and mlm/stream source override
    p.add_argument("--source", default=None,
                   help="local text file path (decoder/lines) or corpus override (decoder/stream, encoder/mlm)")

    # encoder/classify-specific
    p.add_argument("--text-field", default="text", dest="text_field")
    p.add_argument("--label-field", default="label", dest="label_field")

    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode == "encdec":
        if args.dataset is None:
            args.dataset = "Helsinki-NLP/opus_books"
        prepare_encdec(args)

    elif args.mode == "decoder":
        if args.unit == "stream":
            prepare_decoder_stream(args)
        else:
            prepare_decoder_lines(args)

    elif args.mode == "encoder":
        if args.task == "classify":
            if args.dataset is None:
                args.dataset = "stanfordnlp/imdb"
            prepare_encoder_classify(args)
        else:
            prepare_encoder_mlm(args)


if __name__ == "__main__":
    main()
