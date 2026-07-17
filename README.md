# Transformer From Scratch — Usage Guide

Four files, one shared architecture toolbox (`model.py`), three ways to use it:

| Mode | Model class | What it's for |
|---|---|---|
| `encdec` | `Transformer` | Translation (the classic encoder-decoder diagram) |
| `decoder` | `TransformerDecoderOnly` | GPT-style text generation (poems, names, Shakespeare, etc.) |
| `encoder_classify` | `TransformerEncoderOnly` | BERT-style text classification (sentiment, topic, etc.) |
| `encoder_mlm` | `TransformerEncoderOnly` | BERT-style masked-word pretraining / fill-in-the-blank |

Every mode follows the same three-step pipeline:

```
data_cleaning.py  --mode <mode>   -->   train.py --mode <mode>   -->   inference.py --mode <mode>
   (build data/vocab)                  (train + checkpoint)              (play with it)
```

Recommended first project on a GTX 1660 Super: **`decoder` mode on Tiny Shakespeare** — it auto-downloads, needs no extra setup, and trains a visible result in minutes.

---

## 0. Setup

```bash
pip install torch datasets tqdm
```

(`torch` needs a CUDA build if you want GPU training — check [pytorch.org](https://pytorch.org) for the right install command for your driver version.)

All three scripts are run from the folder containing `model.py`, `data_cleaning.py`, `train.py`, `inference.py`.

---

## 1. `decoder` mode — text generation (start here)

### Step 1: Clean the data

**Option A — Tiny Shakespeare (auto-downloaded, zero setup):**
```bash
python data_cleaning.py --mode decoder --unit stream --block-size 64
```
This downloads the corpus, tokenizes it, builds a vocabulary, and chunks it into fixed-length blocks of 64 tokens for next-token prediction. Output goes to `data/decoder/`.

**Option B — your own list (names, poems, jokes — one per line):**
```bash
python data_cleaning.py --mode decoder --unit lines --source names.txt --max-len 20
```
Each line becomes its own training example, wrapped in `<sos>`/`<eos>` and padded at train time. Good for short, independent items rather than one long flowing corpus.

Key flags:
- `--block-size` — sequence length per training chunk (`stream` unit only). Bigger = more context, more VRAM. 32–128 is reasonable for a 1660 Super.
- `--max-len` — max tokens per line (`lines` unit only).

### Step 2: Train

```bash
python train.py --mode decoder \
    --d-model 256 --num-layers 4 --num-heads 8 --d-ff 1024 \
    --batch-size 32 --epochs 20
```

Watch `train_loss` / `val_loss` (and perplexity) drop each epoch. Checkpoints save to `checkpoints/decoder/last.pt` and `checkpoints/decoder/best.pt` (best = lowest val loss so far).

**Sizing for 6GB VRAM (GTX 1660 Super):** `d_model=256, num_layers=4, d_ff=1024, batch_size=32` with `block_size=64` comfortably fits with room to spare. If you hit an out-of-memory error, lower `--batch-size` first, then `--d-model`.

### Step 3: Generate

```bash
# one-shot
python inference.py --mode decoder --text "once upon a time" --max-new-tokens 60 --temperature 0.8 --top-k 20

# interactive
python inference.py --mode decoder
```

- `--temperature` — lower (e.g. 0.5) = safer/more repetitive, higher (e.g. 1.2) = more chaotic/creative.
- `--top-k` — only sample from the k most likely next tokens; smaller k = more coherent, larger k = more variety.

---

## 2. `encoder_classify` mode — text classification

### Step 1: Clean the data

Default pulls `ag_news` (4-class news topic classification) from Hugging Face:
```bash
python data_cleaning.py --mode encoder --task classify
```

Point it at any other Hugging Face text-classification dataset:
```bash
python data_cleaning.py --mode encoder --task classify \
    --dataset dair-ai/emotion --text-field text --label-field label
```
`--text-field` / `--label-field` tell it which columns in the dataset hold the sentence and the label. Output goes to `data/encoder_classify/`, including a `label_names` list saved in `meta.json`.

### Step 2: Train

```bash
python train.py --mode encoder --task classify \
    --d-model 256 --num-layers 4 --num-heads 8 --d-ff 1024 \
    --batch-size 32 --epochs 10 --pooling mean
```

This mode also tracks accuracy alongside loss. `--pooling mean` averages token representations for classification; `--pooling cls` uses only the first token's representation (only makes sense if your data actually has a leading `[CLS]`-style token — `mean` is the safer default here).

### Step 3: Classify

```bash
python inference.py --mode encoder_classify --text "this movie was fantastic"
# or interactive:
python inference.py --mode encoder_classify
```
Prints the predicted label, its confidence, and the full probability distribution across classes.

---

## 3. `encoder_mlm` mode — masked-word pretraining

Same idea as `decoder`/`stream`, but the model learns to fill in randomly masked words instead of predicting the next one — this is what BERT-style pretraining looks like.

### Step 1: Clean the data

```bash
python data_cleaning.py --mode encoder --task mlm --block-size 64
```
Reuses the same auto-downloaded Tiny Shakespeare corpus by default (or pass `--source your_corpus.txt`).

### Step 2: Train

```bash
python train.py --mode encoder --task mlm \
    --d-model 256 --num-layers 4 --num-heads 8 --d-ff 1024 \
    --batch-size 32 --epochs 15
```
Masking (80% → `<mask>`, 10% → random token, 10% unchanged, on ~15% of tokens) happens fresh every batch inside `train.py` — you don't need to pre-mask anything in `data_cleaning.py`.

### Step 3: Fill in the blank

```bash
python inference.py --mode encoder_mlm --text "romeo , where art ___"
# or interactive:
python inference.py --mode encoder_mlm
```
Use `___` (three underscores) anywhere you want a prediction. It'll print the top-k candidate words for that spot.

---

## 4. `encdec` mode — translation

### Step 1: Clean the data

Default pulls `Helsinki-NLP/opus_books` (English → French):
```bash
python data_cleaning.py --mode encdec
```
Swap languages or dataset:
```bash
python data_cleaning.py --mode encdec --lang-pair en-de --src-lang en --tgt-lang de
```

### Step 2: Train

```bash
python train.py --mode encdec \
    --d-model 256 --num-layers 4 --num-heads 8 --d-ff 1024 \
    --batch-size 32 --epochs 20
```
This is the heaviest of the four (two embedding tables, encoder + decoder + cross-attention) — on a 1660 Super, keep `d_model`/`num_layers` modest and expect training to take noticeably longer per epoch than the other three modes.

### Step 3: Translate

```bash
python inference.py --mode encdec --text "hello, how are you"          # greedy
python inference.py --mode encdec --text "hello, how are you" --beam 5  # beam search
# or interactive:
python inference.py --mode encdec
```
Beam search (`--beam 5` or so) usually gives noticeably better translations than greedy decoding, at the cost of more compute per sentence.

---

## Folder layout after running everything

```
data/
  decoder/              vocab.json, {train,val,test}.pkl, meta.json
  encoder_classify/      vocab.json, {train,val,test}.pkl, meta.json
  encoder_mlm/            vocab.json, {train,val,test}.pkl, meta.json
  encdec/                src_vocab.json, tgt_vocab.json, {train,val,test}.pkl, meta.json
  raw/                    downloaded raw corpora (e.g. tinyshakespeare.txt)

checkpoints/
  decoder/{last,best}.pt
  encoder_classify/{last,best}.pt
  encoder_mlm/{last,best}.pt
  encdec/{last,best}.pt
```

`inference.py` defaults to reading `data/<mode>/` and `checkpoints/<mode>/best.pt` automatically — you only need `--data-dir`/`--ckpt` if you moved things around or want to compare a non-"best" checkpoint (e.g. `--ckpt checkpoints/decoder/last.pt`).

---

## Troubleshooting

- **CUDA out of memory** → lower `--batch-size` first, then `--d-model`/`--d-ff`/`--num-layers`, or `--block-size`/`--max-len` for shorter sequences.
- **Loss is `nan`** → lower the learning rate warmup isn't the issue (Noam schedule handles that) — more likely `--grad-clip` is too high, or `--batch-size` is 1 and a stray all-padding batch snuck through. Try `--grad-clip 0.5`.
- **Generated/translated text is empty or all `<eos>`** → normal in early epochs (undertrained model). Keep training, or lower `--temperature` for `decoder` mode to reduce noise while it's still learning.
- **`ModuleNotFoundError: datasets`** → `pip install datasets tqdm`.
- **Slow Hugging Face download** → the first `data_cleaning.py` run for `encdec` or `encoder_classify` downloads the dataset once and caches it locally; subsequent runs are fast.
