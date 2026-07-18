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

Tokenization: --tokenizer {bpe,char}, default bpe.
  bpe  : a from-scratch Byte-Pair-Encoding tokenizer (Sennrich et al. style) -
         learns subword merges from your training corpus. --vocab-size controls
         how many merges it learns (bigger = more whole-word tokens, fewer
         tokens per sentence, but a bigger embedding table).
  char : every individual character is its own token. No training needed,
         vocab size is just however many unique characters appear in the
         corpus. Simpler, longer sequences, easier to reason about.

Text is NOT lowercased and is not restricted to a fixed character whitelist -
both tokenizers handle arbitrary characters/case natively, so cleaning here
is limited to trimming and whitespace normalization.

Every mode writes to its own subfolder under ./data/<mode_name>/ with
train.pkl / val.pkl / test.pkl, tokenizer json(s), and a meta.json describing
everything train.py needs to reconstruct the right model.

Examples:
    python data_cleaning.py --mode encdec
    python data_cleaning.py --mode decoder --unit stream                       # Tiny Shakespeare, BPE
    python data_cleaning.py --mode decoder --unit stream --tokenizer char      # same corpus, char-level
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
# Text cleaning - deliberately light-touch: both tokenizers below handle any
# character/case natively, so we only trim stray whitespace. No lowercasing,
# no character whitelist.
# --------------------------------------------------------------------------- #
_WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


# Pre-tokenizer used only internally by BPE, to stop merges from ever crossing
# a word/punctuation/whitespace boundary. \w and \s are unicode-aware in
# Python's re module, so this handles accented letters etc. too.
_PRETOKEN_RE = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Character-level tokenizer: every character is its own token. No training
# step beyond scanning the corpus once for the set of characters that appear.
# --------------------------------------------------------------------------- #
class CharTokenizer:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    @classmethod
    def train(cls, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        itos = list(SPECIALS) + sorted(chars)
        return cls(itos)

    def encode(self, text: str):
        return [self.stoi.get(ch, UNK_IDX) for ch in text]

    def decode(self, ids):
        return "".join(
            self.itos[i] for i in ids
            if i < len(self.itos) and self.itos[i] not in SPECIALS
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["itos"])

    def __len__(self):
        return len(self.itos)


# --------------------------------------------------------------------------- #
# BPE tokenizer, trained from scratch on your corpus (classic Sennrich et al.
# algorithm): start from individual characters, repeatedly merge the most
# frequent adjacent pair, until the target vocab size is reached. Merges never
# cross a pre-token boundary (word / punctuation / whitespace-run), so it
# won't glue together things like "the" + " cat" into one nonsense unit.
# --------------------------------------------------------------------------- #
class BPETokenizer:
    def __init__(self, itos, merges):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.merges = [tuple(m) for m in merges]  # in the order they were learned
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}

    # ------------------------- training ------------------------- #
    @classmethod
    def train(cls, texts, vocab_size=3000):
        word_freq = Counter()
        for t in texts:
            word_freq.update(_PRETOKEN_RE.findall(t))

        word_splits = {w: list(w) for w in word_freq}

        base_vocab = set()
        for symbols in word_splits.values():
            base_vocab.update(symbols)

        num_merges_target = max(0, vocab_size - len(SPECIALS) - len(base_vocab))

        pair_counts, pair_to_words = cls._count_pairs(word_splits, word_freq)
        merges = []

        for _ in tqdm(range(num_merges_target), desc="learning BPE merges"):
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            merges.append(best_pair)

            affected_words = list(pair_to_words.get(best_pair, ()))
            for w in affected_words:
                symbols = word_splits[w]
                freq = word_freq[w]

                for i in range(len(symbols) - 1):
                    p = (symbols[i], symbols[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    if p in pair_to_words:
                        pair_to_words[p].discard(w)

                new_symbols = cls._merge_symbols(symbols, best_pair)
                word_splits[w] = new_symbols

                for i in range(len(new_symbols) - 1):
                    p = (new_symbols[i], new_symbols[i + 1])
                    pair_counts[p] += freq
                    pair_to_words.setdefault(p, set()).add(w)

        merged_vocab = set(base_vocab)
        for a, b in merges:
            merged_vocab.add(a + b)

        itos = list(SPECIALS) + sorted(merged_vocab)
        return cls(itos, merges)

    @staticmethod
    def _count_pairs(word_splits, word_freq):
        pair_counts = Counter()
        pair_to_words = {}
        for w, symbols in word_splits.items():
            freq = word_freq[w]
            for i in range(len(symbols) - 1):
                p = (symbols[i], symbols[i + 1])
                pair_counts[p] += freq
                pair_to_words.setdefault(p, set()).add(w)
        return pair_counts, pair_to_words

    @staticmethod
    def _merge_symbols(symbols, pair):
        a, b = pair
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    # ------------------------- encode / decode ------------------------- #
    def _bpe_word(self, chunk):
        symbols = list(chunk)
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.merge_rank[p], p) for p in pairs if p in self.merge_rank]
            if not ranked:
                break
            _, best_pair = min(ranked)
            symbols = self._merge_symbols(symbols, best_pair)
        return symbols

    def encode(self, text: str):
        ids = []
        for chunk in _PRETOKEN_RE.findall(text):
            for sym in self._bpe_word(chunk):
                ids.append(self.stoi.get(sym, UNK_IDX))
        return ids

    def decode(self, ids):
        return "".join(
            self.itos[i] for i in ids
            if i < len(self.itos) and self.itos[i] not in SPECIALS
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe", "itos": self.itos, "merges": self.merges},
                       f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["itos"], d["merges"])

    def __len__(self):
        return len(self.itos)


def train_tokenizer(texts, kind: str, vocab_size: int):
    if kind == "char":
        return CharTokenizer.train(texts)
    return BPETokenizer.train(texts, vocab_size=vocab_size)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return CharTokenizer.load(path) if d["type"] == "char" else BPETokenizer.load(path)


# --------------------------------------------------------------------------- #
# Split / save helpers
# --------------------------------------------------------------------------- #
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


def save_meta(out_dir, meta: dict):
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# --------------------------------------------------------------------------- #
# Mode 1: encoder-decoder (translation pairs)
# --------------------------------------------------------------------------- #
def prepare_encdec(args):
    out_dir = os.path.join(args.out_root, "encdec")

    if args.src_file and args.tgt_file:
        print(f"Loading local parallel files: {args.src_file} / {args.tgt_file} ...")
        with open(args.src_file, "r", encoding="utf-8") as f:
            src_lines = f.read().splitlines()
        with open(args.tgt_file, "r", encoding="utf-8") as f:
            tgt_lines = f.read().splitlines()
        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"--src-file has {len(src_lines)} lines but --tgt-file has {len(tgt_lines)} - "
                "they must be aligned line-by-line (line N of one is the translation of line N of the other)."
            )
        src_texts = [clean_text(l) for l in src_lines]
        tgt_texts = [clean_text(l) for l in tgt_lines]

    elif args.parallel_file:
        print(f"Loading local TSV: {args.parallel_file} ...")
        src_texts, tgt_texts = [], []
        with open(args.parallel_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue  # skip malformed rows rather than crashing on one bad line
                src, tgt = parts
                src_texts.append(clean_text(src))
                tgt_texts.append(clean_text(tgt))

    else:
        print(f"Loading {args.dataset} ({args.lang_pair}) from Hugging Face ...")
        raw = load_dataset(args.dataset, args.lang_pair)["train"]
        print("Cleaning text ...")
        src_texts, tgt_texts = [], []
        for example in tqdm(raw):
            pair = example["translation"]
            src_texts.append(clean_text(pair[args.src_lang]))
            tgt_texts.append(clean_text(pair[args.tgt_lang]))

    print(f"Training {args.tokenizer} tokenizers (src + tgt) ...")
    src_tokenizer = train_tokenizer(src_texts, args.tokenizer, args.vocab_size)
    tgt_tokenizer = train_tokenizer(tgt_texts, args.tokenizer, args.vocab_size)
    print(f"src vocab size: {len(src_tokenizer)} | tgt vocab size: {len(tgt_tokenizer)}")

    print("Encoding + filtering by length ...")
    data = []
    for src_text, tgt_text in zip(src_texts, tgt_texts):
        s_ids = src_tokenizer.encode(src_text)
        t_ids = tgt_tokenizer.encode(tgt_text)

        if not (1 <= len(s_ids) <= args.max_len):
            continue
        if not (1 <= len(t_ids) <= args.max_len):
            continue

        data.append((s_ids, [SOS_IDX] + t_ids + [EOS_IDX]))

    print(f"Kept {len(data)} sentence pairs after filtering.")

    train, val, test = split_train_val_test(data)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    src_tokenizer.save(os.path.join(out_dir, "src_tokenizer.json"))
    tgt_tokenizer.save(os.path.join(out_dir, "tgt_tokenizer.json"))
    save_meta(out_dir, {
        "mode": "encdec",
        "tokenizer": args.tokenizer,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "src_vocab_size": len(src_tokenizer),
        "tgt_vocab_size": len(tgt_tokenizer),
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

    lines = [clean_text(line) for line in raw_lines if clean_text(line)]

    print(f"Training {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(lines, args.tokenizer, args.vocab_size)
    print(f"vocab size: {len(tokenizer)}")

    print("Encoding corpus ...")
    ids = []
    for line in tqdm(lines):
        ids.extend(tokenizer.encode(line))
        ids.append(EOS_IDX)  # keep line boundaries as a signal

    print(f"Corpus length: {len(ids)} tokens")

    block = args.block_size + 1  # +1 so we can shift by one for (input, target)
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]
    print(f"Built {len(chunks)} chunks of block_size={args.block_size}")

    train, val, test = split_train_val_test(chunks)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "decoder", "unit": "stream", "tokenizer": args.tokenizer,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(tokenizer),
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

    lines = [clean_text(line) for line in raw_lines]

    print(f"Training {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(lines, args.tokenizer, args.vocab_size)
    print(f"vocab size: {len(tokenizer)}")

    print("Encoding + filtering by length ...")
    data = []
    for line in lines:
        ids = tokenizer.encode(line)
        if 1 <= len(ids) <= args.max_len:
            data.append([SOS_IDX] + ids + [EOS_IDX])

    print(f"Kept {len(data)} lines after filtering.")

    train, val, test = split_train_val_test(data)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "decoder", "unit": "lines", "tokenizer": args.tokenizer,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(tokenizer),
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

    def clean_split(split):
        texts, labels = [], []
        for ex in tqdm(split):
            texts.append(clean_text(ex[text_field]))
            labels.append(ex[label_field])
        return texts, labels

    print("Cleaning train split ...")
    train_texts, train_labels = clean_split(train_raw)

    print(f"Training {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(train_texts, args.tokenizer, args.vocab_size)
    print(f"vocab size: {len(tokenizer)} | num_classes: {len(set(train_labels))}")

    def encode_and_filter(texts, labels):
        data = []
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text)
            if 1 <= len(ids) <= args.max_len:
                data.append((ids, label))
        return data

    data = encode_and_filter(train_texts, train_labels)
    train, val, test_from_train = split_train_val_test(data)

    if test_raw is not None:
        print("Cleaning + encoding test split ...")
        test_texts, test_labels = clean_split(test_raw)
        test = encode_and_filter(test_texts, test_labels)
    else:
        test = test_from_train

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "encoder", "task": "classify", "tokenizer": args.tokenizer,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(tokenizer),
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

    lines = [clean_text(line) for line in raw_lines if clean_text(line)]

    print(f"Training {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(lines, args.tokenizer, args.vocab_size)
    print(f"vocab size: {len(tokenizer)}")

    print("Encoding corpus ...")
    ids = []
    for line in tqdm(lines):
        ids.extend(tokenizer.encode(line))
        ids.append(EOS_IDX)

    block = args.block_size
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block, block)]
    print(f"Built {len(chunks)} chunks of block_size={block}")

    train, val, test = split_train_val_test(chunks)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "encoder", "task": "mlm", "tokenizer": args.tokenizer,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(tokenizer),
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

    p.add_argument("--tokenizer", default="bpe", choices=["bpe", "char"],
                   help="bpe (default): learn subword merges from the corpus. "
                        "char: every character is its own token, no training needed.")
    p.add_argument("--vocab-size", type=int, default=3000, dest="vocab_size",
                   help="target vocab size for --tokenizer bpe (ignored for char)")

    # encdec-specific
    p.add_argument("--dataset", default=None,
                   help="HF dataset name (defaults depend on mode/task)")
    p.add_argument("--lang-pair", default="en-fr", dest="lang_pair")
    p.add_argument("--src-lang", default="en", dest="src_lang")
    p.add_argument("--tgt-lang", default="fr", dest="tgt_lang")
    p.add_argument("--src-file", default=None, dest="src_file",
                   help="encdec: local text file, one source sentence per line - "
                        "use with --tgt-file instead of --dataset")
    p.add_argument("--tgt-file", default=None, dest="tgt_file",
                   help="encdec: local text file, one target sentence per line, "
                        "aligned line-by-line with --src-file")
    p.add_argument("--parallel-file", default=None, dest="parallel_file",
                   help="encdec: single local TSV file, each line 'source<TAB>target' - "
                        "alternative to --src-file/--tgt-file")

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
                args.dataset = "ag_news"
            prepare_encoder_classify(args)
        else:
            prepare_encoder_mlm(args)


if __name__ == "__main__":
    main()