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
                 (e.g. Tiny Shakespeare - default, auto-downloaded; or a
                 Hugging Face dataset via --dataset, rows concatenated
                 together before chunking)
        lines  : one example per line, independently padded
                 (e.g. a names.txt / poems.txt file you supply with --source,
                 or a Hugging Face dataset via --dataset, one row per line -
                 --max-len filters out rows whose encoded length falls
                 outside [1, max_len])

  --mode encoder --task classify
      Labeled classification data for TransformerEncoderOnly.
      Default source: HF `ag_news` (4-class news topic classification).

  --mode encoder --task mlm
      Unlabeled text corpus for masked-language-model pretraining of
      TransformerEncoderOnly (same chunking as decoder/stream - masking
      itself happens dynamically in train.py, not here).

  --mode encoder --task regression
      Text paired with a numeric (float) target for TransformerRegression -
      e.g. predicting a star rating from review text.
      Default source: HF `yelp_review_full` (text -> 0-4 star rating, used
      here as a numeric regression target rather than a class label).
      Alternatively, pass --source (and optionally --val-source/--test-source)
      pointing at local TSV files of 'text<TAB>label' lines - the format a
      dataset-specific prep script (e.g. build_fluorescence_dataset.py) writes,
      once it has flattened that dataset's own fields into a single text string.
      Targets are standardized (zero mean, unit std, computed on the train
      split only) before saving; the mean/std are stored in meta.json so
      train.py/inference.py can convert predictions back to the original scale.

Tokenization: --tokenizer {tiktoken,char}, default tiktoken.
  tiktoken : uses a pretrained OpenAI tiktoken encoding (--tiktoken-encoding,
             default "cl100k_base") as a fixed subword vocabulary - nothing is
             learned/trained. We scan your corpus once with that encoding and
             keep only the token ids that actually show up (e.g. your corpus
             might only ever produce 30k of cl100k_base's ~100k possible
             tokens) - the unused ~70k rows are simply never added to the
             embedding table. Because of this, --vocab-size does NOT apply to
             this tokenizer: the vocab size is whatever comes out of that scan,
             not a target you pick. Any token encountered later (val/test, or
             inference) that wasn't seen during the training-corpus scan maps
             to <unk>, same as an unseen character would for --tokenizer char.
  char     : every individual character is its own token. No training needed,
             vocab size is just however many unique characters appear in the
             corpus. Simpler, longer sequences, easier to reason about.

Text is NOT lowercased and is not restricted to a fixed character whitelist -
both tokenizers handle arbitrary characters/case natively, so cleaning here
is limited to trimming and whitespace normalization.

Every mode writes to its own subfolder under ./data/<mode_name>/ with
train.pkl / val.pkl / test.pkl, tokenizer json(s), and a meta.json describing
everything train.py needs to reconstruct the right model.

Requires the `tiktoken` package (`pip install tiktoken`). The first time a
given --tiktoken-encoding is used on a machine, tiktoken downloads and caches
it locally, so that first run needs internet access.

Examples:
    python data_cleaning.py --mode encdec
    python data_cleaning.py --mode decoder --unit stream                       # Tiny Shakespeare, tiktoken
    python data_cleaning.py --mode decoder --unit stream --tokenizer char      # same corpus, char-level
    python data_cleaning.py --mode decoder --unit lines --source names.txt
    python data_cleaning.py --mode decoder --unit lines --dataset ag_news --text-field text --max-len 64
    python data_cleaning.py --mode decoder --unit stream --dataset wikitext --text-field text
    python data_cleaning.py --mode encoder --task classify
    python data_cleaning.py --mode encoder --task mlm
    python data_cleaning.py --mode encoder --task regression
"""

import os
import re
import json
import pickle
import random
import argparse
import urllib.request

import tiktoken
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


# --------------------------------------------------------------------------- #
# Row-limiting helpers - let you cap how many rows get pulled in before the
# (often slow) cleaning/tokenizing steps, via --data-rows. Subsampling is
# random (seeded off the global random.seed(42) above) rather than a plain
# head-slice, since many datasets are sorted (e.g. by label) and a head-slice
# would silently bias what you keep.
# --------------------------------------------------------------------------- #
def limit_rows(items, n, desc="rows"):
    """Randomly subsample a plain list down to at most `n` entries. No-op if
    n is None or the list already has <= n entries."""
    if n is None or len(items) <= n:
        return items
    print(f"Subsampling {desc}: {len(items)} -> {n}")
    return random.sample(items, n)


def limit_paired_rows(n, *lists, desc="rows"):
    """Same as limit_rows, but applied jointly across several equal-length
    lists (e.g. src_texts/tgt_texts, or texts/labels) so pairing is preserved."""
    length = len(lists[0])
    if n is None or length <= n:
        return lists
    print(f"Subsampling {desc}: {length} -> {n}")
    idx = random.sample(range(length), n)
    return tuple([lst[i] for i in idx] for lst in lists)


def limit_hf_dataset(ds, n, desc="rows"):
    """Same idea for a Hugging Face Dataset object - shuffle + select is the
    efficient way to subsample before mapping/cleaning over every row."""
    if n is None or len(ds) <= n:
        return ds
    print(f"Subsampling {desc}: {len(ds)} -> {n}")
    return ds.shuffle(seed=42).select(range(n))


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
# Tiktoken-backed tokenizer: no training/merge-learning at all - the subword
# vocabulary comes from a pretrained OpenAI tiktoken encoding (e.g.
# "cl100k_base", the GPT-3.5/4 encoding). The only thing "trained" here is
# which of that encoding's ~100k token ids actually appear in your corpus;
# ids that never show up are simply excluded from the embedding table instead
# of being carried around unused. So if your corpus only ever produces, say,
# 30k distinct tiktoken ids, your model's vocab size is ~30k (+ specials),
# not 100k. --vocab-size is ignored by this tokenizer for that reason - there's
# no merge count to target, only "whatever ids the corpus actually uses".
# --------------------------------------------------------------------------- #
class TiktokenTokenizer:
    def __init__(self, encoding_name, used_ids):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        # itos[i] for i >= len(SPECIALS) is the underlying tiktoken token id
        # that local id `i` refers to.
        self.itos = list(SPECIALS) + list(used_ids)
        # tiktoken_id -> local id, only for ids we actually kept.
        self.tok_to_local = {
            tok_id: len(SPECIALS) + i for i, tok_id in enumerate(used_ids)
        }

    @classmethod
    def train(cls, texts, encoding_name="cl100k_base"):
        enc = tiktoken.get_encoding(encoding_name)
        used = set()
        for t in tqdm(texts, desc=f"scanning corpus with tiktoken ({encoding_name})"):
            used.update(enc.encode(t, disallowed_special=()))
        used_ids = sorted(used)
        return cls(encoding_name, used_ids)

    def encode(self, text: str):
        # disallowed_special=() tells tiktoken to treat every one of its own
        # reserved special strings (e.g. "<|endoftext|>") as ordinary text if
        # it happens to appear in the corpus, instead of raising or special-
        # casing it - we only ever want *our* SPECIALS (<pad>/<sos>/<eos>/
        # <unk>/<mask>), never tiktoken's.
        raw_ids = self.enc.encode(text, disallowed_special=())
        # A raw id that never showed up during the training-corpus scan (only
        # possible on val/test/inference text) maps to <unk>, same as an
        # unseen character would for CharTokenizer.
        return [self.tok_to_local.get(i, UNK_IDX) for i in raw_ids]

    def decode(self, ids):
        raw_ids = [
            self.itos[i] for i in ids
            if i < len(self.itos) and i >= len(SPECIALS)
        ]
        return self.enc.decode(raw_ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "type": "tiktoken",
                "encoding_name": self.encoding_name,
                "used_ids": self.itos[len(SPECIALS):],
            }, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["encoding_name"], d["used_ids"])

    def __len__(self):
        return len(self.itos)


def train_tokenizer(texts, kind: str, encoding_name: str = "cl100k_base"):
    if kind == "char":
        return CharTokenizer.train(texts)
    return TiktokenTokenizer.train(texts, encoding_name=encoding_name)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return CharTokenizer.load(path) if d["type"] == "char" else TiktokenTokenizer.load(path)


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
        src_lines, tgt_lines = limit_paired_rows(args.data_rows, src_lines, tgt_lines, desc="sentence pairs")
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
        src_texts, tgt_texts = limit_paired_rows(args.data_rows, src_texts, tgt_texts, desc="sentence pairs")

    else:
        print(f"Loading {args.dataset} ({args.lang_pair}) from Hugging Face ...")
        raw = load_dataset(args.dataset, args.lang_pair)["train"]
        raw = limit_hf_dataset(raw, args.data_rows, desc="sentence pairs")
        print("Cleaning text ...")
        src_texts, tgt_texts = [], []
        for example in tqdm(raw):
            pair = example["translation"]
            src_texts.append(clean_text(pair[args.src_lang]))
            tgt_texts.append(clean_text(pair[args.tgt_lang]))

    print(f"Building {args.tokenizer} tokenizers (src + tgt) ...")
    src_tokenizer = train_tokenizer(src_texts, args.tokenizer, args.tiktoken_encoding)
    tgt_tokenizer = train_tokenizer(tgt_texts, args.tokenizer, args.tiktoken_encoding)
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


def _load_hf_text_rows(dataset_name, text_field, split="train", max_rows=None):
    """Load a Hugging Face dataset and pull out one cleaned string per row
    from `text_field`. Shared by decoder/stream and decoder/lines so both
    can source their corpus from the Hub instead of (or in addition to) a
    local file."""
    print(f"Loading {dataset_name} from Hugging Face (split={split}) ...")
    raw = load_dataset(dataset_name)
    if split not in raw:
        raise ValueError(
            f"Split '{split}' not found in {dataset_name} - available splits: {list(raw.keys())}"
        )
    rows = raw[split]
    rows = limit_hf_dataset(rows, max_rows, desc="rows")
    print("Cleaning text ...")
    texts = [clean_text(ex[text_field]) for ex in tqdm(rows)]
    return [t for t in texts if t]  # drop rows that clean down to empty strings


def prepare_decoder_stream(args):
    """One long corpus -> chunked into fixed-length blocks for next-token prediction.

    Source priority: --source (local file) > --dataset (Hugging Face, rows
    concatenated together) > Tiny Shakespeare (auto-downloaded default).
    """
    out_dir = os.path.join(args.out_root, "decoder")

    source_desc = None
    if args.source is not None:
        print(f"Reading corpus from {args.source} ...")
        with open(args.source, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
        lines = [clean_text(line) for line in raw_lines if clean_text(line)]
        lines = limit_rows(lines, args.data_rows, desc="lines")
        source_desc = args.source
    elif args.dataset is not None:
        lines = _load_hf_text_rows(args.dataset, args.text_field, max_rows=args.data_rows)
        source_desc = f"hf:{args.dataset}"
    else:
        source_path = _download_tiny_shakespeare(os.path.join("data", "raw", "tinyshakespeare.txt"))
        print(f"Reading corpus from {source_path} ...")
        with open(source_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
        lines = [clean_text(line) for line in raw_lines if clean_text(line)]
        lines = limit_rows(lines, args.data_rows, desc="lines")
        source_desc = source_path

    print(f"Building {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(lines, args.tokenizer, args.tiktoken_encoding)
    print(f"vocab size: {len(tokenizer)}")

    print("Encoding corpus ...")
    ids = []
    for line in tqdm(lines):
        ids.extend(tokenizer.encode(line))
        ids.append(EOS_IDX)  # keep line/row boundaries as a signal

    print(f"Corpus length: {len(ids)} tokens")

    block = args.block_size + 1  # +1 so we can shift by one for (input, target)
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block + 1, block)]
    print(f"Built {len(chunks)} chunks of block_size={args.block_size}")

    train, val, test = split_train_val_test(chunks)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "decoder", "unit": "stream", "tokenizer": args.tokenizer,
        "source": source_desc,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(tokenizer),
        "block_size": args.block_size,
    })
    print(f"Done. Saved to ./{out_dir}/")


def prepare_decoder_lines(args):
    """One example per line (names, poems, jokes, ...) -> independently padded
    sequences. Source priority: --source (local .txt, one example per line) >
    --dataset (Hugging Face, one row per example, `--text-field` selects the
    column). Either way, rows are dropped unless their encoded length falls in
    [1, --max-len] (checked below), mirroring how the other modes filter by
    length.
    """
    if args.source is None and args.dataset is None:
        raise ValueError(
            "--unit lines requires either --source path/to/file.txt (one example per line) "
            "or --dataset <hf_dataset_name> (with --text-field, default 'text')"
        )

    out_dir = os.path.join(args.out_root, "decoder")

    if args.source is not None:
        print(f"Reading lines from {args.source} ...")
        with open(args.source, "r", encoding="utf-8") as f:
            raw_lines = [l for l in f.read().splitlines() if l.strip()]
        lines = [clean_text(line) for line in raw_lines]
        lines = limit_rows(lines, args.data_rows, desc="lines")
        source_desc = args.source
    else:
        lines = _load_hf_text_rows(args.dataset, args.text_field, max_rows=args.data_rows)
        source_desc = f"hf:{args.dataset}"

    print(f"Building {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(lines, args.tokenizer, args.tiktoken_encoding)
    print(f"vocab size: {len(tokenizer)}")

    print(f"Encoding + filtering by length (max_len={args.max_len}) ...")
    data = []
    for line in lines:
        ids = tokenizer.encode(line)
        if 1 <= len(ids) <= args.max_len:
            data.append([SOS_IDX] + ids + [EOS_IDX])

    print(f"Kept {len(data)} / {len(lines)} rows after length filtering.")

    train, val, test = split_train_val_test(data)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "decoder", "unit": "lines", "tokenizer": args.tokenizer,
        "source": source_desc,
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

    train_raw = limit_hf_dataset(train_raw, args.data_rows, desc="train rows")
    if test_raw is not None:
        test_raw = limit_hf_dataset(test_raw, args.data_rows, desc="test rows")

    text_field = args.text_field
    label_field = args.label_field

    feature = train_raw.features[label_field]
    hf_label_names = feature.names if hasattr(feature, "names") else None

    def clean_split(split):
        texts, labels = [], []
        for ex in tqdm(split):
            texts.append(clean_text(ex[text_field]))
            labels.append(ex[label_field])
        return texts, labels

    print("Cleaning train split ...")
    train_texts, train_labels_raw = clean_split(train_raw)

    # Build a label -> integer id mapping. If the dataset uses HF's ClassLabel
    # type, its values are already integer ids and .names gives us the string
    # for each id, in order. If labels are plain strings (or anything else
    # not pre-encoded, e.g. "legitimate"/"phishing"), build our own mapping
    # from the sorted set of values actually seen in the train split, and
    # convert every label through it below - a torch tensor can't hold
    # strings, so this conversion is required either way.
    if hf_label_names is not None:
        label_names = hf_label_names
        label_to_id = {name: i for i, name in enumerate(label_names)}
        def to_id(raw_label):
            return raw_label if isinstance(raw_label, int) else label_to_id[raw_label]
    else:
        label_names = sorted({str(l) for l in train_labels_raw})
        label_to_id = {name: i for i, name in enumerate(label_names)}
        def to_id(raw_label):
            return label_to_id[str(raw_label)]

    print(f"classes ({len(label_names)}): {label_names}")

    print(f"Building {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(train_texts, args.tokenizer, args.tiktoken_encoding)
    print(f"vocab size: {len(tokenizer)}")

    def encode_and_filter(texts, labels_raw):
        data = []
        skipped_unseen = 0
        for text, raw_label in zip(texts, labels_raw):
            if not isinstance(raw_label, int) and str(raw_label) not in label_to_id:
                skipped_unseen += 1  # label not seen in train split - can't assign an id
                continue
            ids = tokenizer.encode(text)
            if 1 <= len(ids) <= args.max_len:
                data.append(([SOS_IDX] + ids, to_id(raw_label)))
        if skipped_unseen:
            print(f"  (skipped {skipped_unseen} examples with labels not seen in the train split)")
        return data

    data = encode_and_filter(train_texts, train_labels_raw)
    train, val, test_from_train = split_train_val_test(data)

    if test_raw is not None:
        print("Cleaning + encoding test split ...")
        test_texts, test_labels_raw = clean_split(test_raw)
        test = encode_and_filter(test_texts, test_labels_raw)
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
        "num_classes": len(label_names),
        "label_names": label_names,
        "max_len": args.max_len + 1,
    })
    print(f"Done. Saved to ./{out_dir}/")


# --------------------------------------------------------------------------- #
# Mode 3c: encoder-only, regression
# --------------------------------------------------------------------------- #
def load_regression_tsv(path):
    """Local file format for encoder/regression: one 'text<TAB>label' example per line.
    This is what a dataset-specific prep script (e.g. build_fluorescence_dataset.py)
    writes out - this function only knows about generic text/label pairs, not any
    particular dataset's own column names, which is exactly why that flattening step
    has to happen upstream, in the prep script."""
    texts, targets = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"  (skipping malformed line {line_num} in {path}: expected 1 tab, got {len(parts) - 1})")
                continue
            text, label = parts
            texts.append(clean_text(text))
            targets.append(float(label))
    return texts, targets


def prepare_encoder_regression(args):
    out_dir = os.path.join(args.out_root, "encoder_regression")

    test_texts, test_targets_raw = None, None  # populated below if a test split is available

    if args.source:
        print(f"Loading local training TSV: {args.source} ...")
        train_texts, train_targets_raw = load_regression_tsv(args.source)
        train_texts, train_targets_raw = limit_paired_rows(
            args.data_rows, train_texts, train_targets_raw, desc="train rows")
        if args.test_source:
            print(f"Loading local test TSV: {args.test_source} ...")
            test_texts, test_targets_raw = load_regression_tsv(args.test_source)
            test_texts, test_targets_raw = limit_paired_rows(
                args.data_rows, test_texts, test_targets_raw, desc="test rows")
    else:
        print(f"Loading {args.dataset} from Hugging Face ...")
        raw = load_dataset(args.dataset)
        train_raw = raw["train"]
        hf_test_raw = raw["test"] if "test" in raw else None

        train_raw = limit_hf_dataset(train_raw, args.data_rows, desc="train rows")
        if hf_test_raw is not None:
            hf_test_raw = limit_hf_dataset(hf_test_raw, args.data_rows, desc="test rows")

        text_field = args.text_field
        label_field = args.label_field

        def clean_split(split):
            texts, targets = [], []
            for ex in tqdm(split):
                texts.append(clean_text(ex[text_field]))
                # Cast straight to float - works whether label_field is already a float
                # score (e.g. a similarity score) or an int class id being repurposed as
                # a numeric target (e.g. yelp_review_full's 0-4 star rating).
                targets.append(float(ex[label_field]))
            return texts, targets

        print("Cleaning train split ...")
        train_texts, train_targets_raw = clean_split(train_raw)
        if hf_test_raw is not None:
            print("Cleaning + encoding test split ...")
            test_texts, test_targets_raw = clean_split(hf_test_raw)

    # Standardize targets (zero mean, unit std) using train-split statistics only,
    # to avoid leaking val/test information into the normalization. This is the
    # same reason encoder --task classify avoids Noam in train.py: a small model
    # training on raw, unnormalized targets (e.g. always in the hundreds) can make
    # MSE loss/gradients large enough to destabilize training. mean/std are saved
    # in meta.json so train.py's checkpoint (and inference.py) can convert
    # standardized predictions back to the original label scale.
    target_mean = sum(train_targets_raw) / len(train_targets_raw)
    variance = sum((t - target_mean) ** 2 for t in train_targets_raw) / len(train_targets_raw)
    target_std = max(variance ** 0.5, 1e-6)  # floor to avoid a divide-by-zero on a constant target
    print(f"target stats (train split): mean={target_mean:.4f} std={target_std:.4f}")

    print(f"Building {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(train_texts, args.tokenizer, args.tiktoken_encoding)
    print(f"vocab size: {len(tokenizer)}")

    def encode_and_filter(texts, targets_raw):
        data = []
        for text, target in zip(texts, targets_raw):
            ids = tokenizer.encode(text)
            if 1 <= len(ids) <= args.max_len:
                standardized = (target - target_mean) / target_std
                data.append(([SOS_IDX] + ids, standardized))
        return data

    data = encode_and_filter(train_texts, train_targets_raw)
    train, val, test_from_train = split_train_val_test(data)

    # A pre-defined val split (e.g. --val-source) overrides the carve-out above -
    # useful whenever the source dataset's own split matters (like fluorescence's
    # extrapolation-style train/valid/test, rather than an i.i.d. random split).
    if args.val_source:
        print(f"Loading local val TSV: {args.val_source} ...")
        val_texts, val_targets_raw = load_regression_tsv(args.val_source)
        val_texts, val_targets_raw = limit_paired_rows(
            args.data_rows, val_texts, val_targets_raw, desc="val rows")
        val = encode_and_filter(val_texts, val_targets_raw)

    if test_texts is not None:
        test = encode_and_filter(test_texts, test_targets_raw)
    else:
        test = test_from_train

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    save_split(out_dir, train, val, test)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    save_meta(out_dir, {
        "mode": "encoder", "task": "regression", "tokenizer": args.tokenizer,
        "pad_idx": PAD_IDX, "sos_idx": SOS_IDX, "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX, "mask_idx": MASK_IDX,
        "vocab_size": len(tokenizer),
        "num_targets": 1,
        "target_mean": target_mean,
        "target_std": target_std,
        "max_len": args.max_len + 1,
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
    lines = limit_rows(lines, args.data_rows, desc="lines")

    print(f"Building {args.tokenizer} tokenizer ...")
    tokenizer = train_tokenizer(lines, args.tokenizer, args.tiktoken_encoding)
    print(f"vocab size: {len(tokenizer)}")

    print("Encoding corpus ...")
    ids = []
    for line in tqdm(lines):
        ids.extend(tokenizer.encode(line))
        ids.append(EOS_IDX)

    block = args.block_size
    chunks = [ids[i:i + block] for i in range(0, len(ids) - block + 1, block)]
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
    p.add_argument("--task", default="classify", choices=["classify", "mlm", "regression"],
                   help="only used when --mode encoder")
    p.add_argument("--unit", default="stream", choices=["stream", "lines"],
                   help="only used when --mode decoder")
    p.add_argument("--out-root", default="data")
    p.add_argument("--max-len", type=int, default=100, dest="max_len",
                   help="max encoded length per row/example - used to filter rows for "
                        "encdec, encoder (classify/regression), and decoder/lines "
                        "(including decoder/lines when sourced from --dataset). Not used "
                        "by decoder/stream or encoder/mlm, which chunk by --block-size instead.")
    p.add_argument("--block-size", type=int, default=64, dest="block_size",
                   help="sequence length for decoder/stream and encoder/mlm chunking")
    p.add_argument("--data-rows", type=int, default=None, dest="data_rows",
                   help="cap on how many rows to pull in before cleaning/tokenizing, e.g. "
                        "--data-rows 100000 to avoid pulling in all 2M+ rows of a big HF "
                        "dataset. Applies per split (train/val/test each capped separately) "
                        "and, for encdec/regression, jointly across paired lists (src/tgt, "
                        "text/label) so pairing stays intact. Subsampling is random (seeded) "
                        "rather than just taking the first N rows, since many datasets are "
                        "sorted (e.g. by label) and a head-slice would bias what you keep. "
                        "Default: None (use everything).")

    p.add_argument("--tokenizer", default="tiktoken", choices=["tiktoken", "char"],
                   help="tiktoken (default): use a pretrained OpenAI tiktoken encoding "
                        "(see --tiktoken-encoding) as a fixed subword vocab, but only keep "
                        "the token ids that actually appear in your corpus - nothing is "
                        "learned/merged, so --vocab-size does not apply to it. "
                        "char: every character is its own token, no training needed.")
    p.add_argument("--tiktoken-encoding", default="cl100k_base", dest="tiktoken_encoding",
                   help="which pretrained tiktoken encoding to draw tokens from when "
                        "--tokenizer tiktoken is used, e.g. cl100k_base (GPT-3.5/4) or "
                        "o200k_base (GPT-4o). Ignored for --tokenizer char.")
    p.add_argument("--vocab-size", type=int, default=3000, dest="vocab_size",
                   help="[unused - kept only for backwards-compatible CLI calls] vocab "
                        "size is no longer a target: --tokenizer tiktoken keeps exactly "
                        "the token ids seen in your corpus, and --tokenizer char keeps "
                        "exactly the characters seen in your corpus.")

    # encdec-specific
    p.add_argument("--dataset", default=None,
                   help="HF dataset name (defaults depend on mode/task). For --mode decoder, "
                        "there is no default - pass this to pull the corpus from Hugging Face "
                        "instead of a local file/URL.")
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

    # decoder/lines and mlm/stream source override, also encoder/regression's local data
    p.add_argument("--source", default=None,
                   help="local text file path (decoder/lines, decoder/stream), corpus "
                        "override (decoder/stream, encoder/mlm), or a local TSV of "
                        "'text<TAB>label' lines (encoder/regression) - overrides --dataset")
    p.add_argument("--val-source", default=None, dest="val_source",
                   help="encoder regression only: local TSV of 'text<TAB>label' lines for a "
                        "pre-defined validation split; if omitted, val is carved out of the "
                        "training data the normal way")
    p.add_argument("--test-source", default=None, dest="test_source",
                   help="encoder regression only: local TSV of 'text<TAB>label' lines for a "
                        "pre-defined test split; if omitted, falls back to the HF dataset's "
                        "own test split (if --source wasn't used and one exists), then to a "
                        "carve-out from the training data")

    # encoder/classify, encoder/regression, and decoder (when sourced from --dataset)
    p.add_argument("--text-field", default="text", dest="text_field",
                   help="HF dataset column holding the text - used by encoder/classify, "
                        "encoder/regression, and decoder/{stream,lines} when --dataset is given")
    p.add_argument("--label-field", default="label", dest="label_field",
                   help="classify: class label field. regression: numeric target field "
                        "(cast to float, e.g. a rating or score)")

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
                args.dataset = "fancyzhx/ag_news"
            prepare_encoder_classify(args)
        elif args.task == "regression":
            if args.dataset is None and args.source is None:
                args.dataset = "yelp_review_full"
            prepare_encoder_regression(args)
        else:
            prepare_encoder_mlm(args)


if __name__ == "__main__":
    main()