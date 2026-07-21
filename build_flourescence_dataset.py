"""
build_fluorescence_dataset.py
------------------------------
Downloads proteinea/fluorescence from Hugging Face - the TAPE "fluorescence
landscape" benchmark (Sarkisyan et al. 2016: avGFP mutants, log-fluorescence
intensity as the target) - and writes it out as local TSV files that
data_cleaning.py's `--mode encoder --task regression` can consume directly via
--source/--val-source/--test-source.

Why this script exists at all: data_cleaning.py's encoder/regression prep only
understands one generic text field + one numeric label field (see --text-field/
--label-field). This task's real input is *two* things - a protein sequence and
how many mutations it carries relative to wild-type avGFP - so this script folds
both into a single text string before handing it off:

    "m<N> <SEQUENCE>"   e.g.   "m3 MEHVIDNFDNIDKCLKCGKPIKVVKLKYIKKKIENIPNSHLINFKYC..."

Both of data_cleaning.py's tokenizers handle this natively: char-level treats
"m", each digit, the space, and every amino-acid letter as its own token; bpe
will happily learn "m0"/"m1"/"m2"/... as short subword units, same as any other
frequent short token, right alongside amino-acid n-grams it discovers in the
sequences themselves.

Column names are read defensively (a few candidates are tried for each field)
since different mirrors/re-uploads of this dataset use slightly different
names (e.g. "primary" vs "seq" for the sequence, "log_fluorescence" vs "label"
for the target) - see *_FIELD_CANDIDATES below. If proteinea/fluorescence's
actual columns don't match any candidate, this script will raise a clear
KeyError naming exactly what it found instead, rather than silently guessing.

Usage:
    python build_fluorescence_dataset.py
    # writes data/raw/fluorescence/{train,val,test}.tsv (whichever splits exist)

    python data_cleaning.py --mode encoder --task regression \\
        --source data/raw/fluorescence/train.tsv \\
        --val-source data/raw/fluorescence/val.tsv \\
        --test-source data/raw/fluorescence/test.tsv

    python train.py --mode encoder --task regression --epochs 10
    python inference.py --mode encoder_regression --text "m2 MEHVIDNFD..."
"""

import os
import argparse

from datasets import load_dataset

# Different mirrors of this dataset name columns slightly differently - try each
# candidate in order and use whichever is actually present, rather than hard-coding
# one name and breaking the moment a mirror renames a column.
SEQUENCE_FIELD_CANDIDATES = ["primary", "seq", "sequence"]
LABEL_FIELD_CANDIDATES = ["log_fluorescence", "label", "fluorescence"]
MUTATION_FIELD_CANDIDATES = ["num_mutations", "num_mutation", "n_mutations"]

# TAPE's own split names are train/valid/test; some mirrors use "validation"
# instead of "valid" - both are checked. Left side is the name this script
# writes to disk (matching train.py's --val-source / --test-source flags).
SPLIT_NAME_CANDIDATES = {
    "train": ["train"],
    "val": ["valid", "validation", "val"],
    "test": ["test"],
}


def _first_present(example, candidates, what):
    for name in candidates:
        if name in example:
            return name
    raise KeyError(
        f"Couldn't find a {what} field - tried {candidates}, but this example "
        f"only has these columns: {list(example.keys())}. Pass the real column "
        f"name(s) by editing *_FIELD_CANDIDATES at the top of this script."
    )


def _scalar(value):
    """Some mirrors store the label as a length-1 list (e.g. log_fluorescence:
    [3.6]) - matching the original TAPE json format - others store a plain
    float. Handle both."""
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def convert_split(split, out_path):
    if len(split) == 0:
        return 0

    first = split[0]
    seq_field = _first_present(first, SEQUENCE_FIELD_CANDIDATES, "protein sequence")
    label_field = _first_present(first, LABEL_FIELD_CANDIDATES, "fluorescence label")
    mut_field = next((f for f in MUTATION_FIELD_CANDIDATES if f in first), None)
    if mut_field is None:
        print("  (no mutation-count field found - writing sequence only, no 'm<N>' prefix)")

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in split:
            sequence = ex[seq_field].strip()
            if not sequence:
                continue
            label = _scalar(ex[label_field])
            if mut_field is not None:
                text = f"m{int(ex[mut_field])} {sequence}"
            else:
                text = sequence
            # text<TAB>label, one example per line - the format load_regression_tsv()
            # in data_cleaning.py expects for --source/--val-source/--test-source.
            f.write(f"{text}\t{label}\n")
            n_written += 1
    return n_written


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="proteinea/fluorescence")
    p.add_argument("--out-dir", default=os.path.join("data", "raw", "fluorescence"), dest="out_dir")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.dataset} from Hugging Face ...")
    try:
        raw = load_dataset(args.dataset, trust_remote_code=True)
    except TypeError:
        # older `datasets` versions don't accept trust_remote_code at all
        raw = load_dataset(args.dataset)

    print(f"Available splits: {list(raw.keys())}")

    written = {}
    for out_name, candidates in SPLIT_NAME_CANDIDATES.items():
        split_key = next((c for c in candidates if c in raw), None)
        if split_key is None:
            print(f"  (no {out_name} split found - tried {candidates}, skipping)")
            continue
        out_path = os.path.join(args.out_dir, f"{out_name}.tsv")
        n = convert_split(raw[split_key], out_path)
        written[out_name] = out_path
        print(f"  wrote {n} examples from split '{split_key}' -> {out_path}")

    print("\nDone. Now run e.g.:")
    cmd = ["python data_cleaning.py --mode encoder --task regression",
           f"    --source {written['train']}"]
    if "val" in written:
        cmd.append(f"    --val-source {written['val']}")
    if "test" in written:
        cmd.append(f"    --test-source {written['test']}")
    print(" \\\n".join(cmd))
    print("python train.py --mode encoder --task regression --epochs 10")


if __name__ == "__main__":
    main()