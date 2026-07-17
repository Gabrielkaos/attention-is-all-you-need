"""
model.py
--------
A from-scratch implementation of the original Transformer
(Vaswani et al., "Attention Is All You Need") that mirrors the
classic encoder/decoder diagram:

    Inputs -> Input Embedding -> + Positional Encoding -> [Encoder x N] ---\
                                                                            \
    Outputs(shifted right) -> Output Embedding -> + Positional Encoding -> [Decoder x N] -> Linear -> Softmax

Each Encoder block:  Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm
Each Decoder block:  Masked Multi-Head Attention -> Add&Norm
                      -> Multi-Head (cross) Attention -> Add&Norm
                      -> Feed Forward -> Add&Norm

No external attention/transformer libraries are used - everything (attention,
positional encoding, encoder/decoder layers) is implemented manually with
plain torch.nn so you can see exactly what every box in the diagram does.
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Positional Encoding (the sinusoidal "swirl" symbol in the diagram)
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# --------------------------------------------------------------------------- #
# Multi-Head Attention (the orange "Multi-Head Attention" boxes)
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)

        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask: broadcastable to (batch, 1, 1/seq_len, seq_len); True/1 = keep, 0 = mask out
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.w_o(out)


# --------------------------------------------------------------------------- #
# Position-wise Feed Forward (the blue "Feed Forward" boxes)
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Add & Norm helper (residual connection + LayerNorm, applied post-sublayer
# exactly as in the original paper / diagram)
# --------------------------------------------------------------------------- #
class AddNorm(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))


# --------------------------------------------------------------------------- #
# One Encoder block: Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.add_norm1(x, attn_out)
        ff_out = self.feed_forward(x)
        x = self.add_norm2(x, ff_out)
        return x


# --------------------------------------------------------------------------- #
# One Decoder block: Masked MHA -> Add&Norm -> (cross) MHA -> Add&Norm
#                     -> Feed Forward -> Add&Norm
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # Masked multi-head self-attention over decoder inputs
        self_attn_out = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.add_norm1(x, self_attn_out)

        # Multi-head cross-attention: Q from decoder, K/V from encoder output
        cross_attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.add_norm2(x, cross_attn_out)

        ff_out = self.feed_forward(x)
        x = self.add_norm3(x, ff_out)
        return x


# --------------------------------------------------------------------------- #
# Encoder stack: Input Embedding + Positional Encoding -> N x EncoderLayer
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# --------------------------------------------------------------------------- #
# Decoder stack: Output Embedding + Positional Encoding -> N x DecoderLayer
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# --------------------------------------------------------------------------- #
# Full Transformer: Encoder + Decoder + Linear + Softmax head
# --------------------------------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads,
                                d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads,
                                d_ff, max_len, dropout)

        # "Linear" box in the diagram. Softmax is applied only at inference
        # time (see inference.py) - during training we feed raw logits
        # straight into nn.CrossEntropyLoss, which applies log-softmax itself.
        self.linear = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------- mask helpers ------------------------- #
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # (batch, 1, 1, src_len) -> True where token is NOT padding
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        batch, tgt_len = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (b,1,1,t)
        subsequent_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool)
        )  # (t, t) lower-triangular = "look only at previous tokens"
        return pad_mask & subsequent_mask  # broadcasts to (b, 1, t, t)

    # ------------------------------ forward ------------------------------ #
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        logits = self.linear(dec_out)  # (batch, tgt_len, tgt_vocab_size)
        return logits

    # convenience used by inference.py for step-by-step decoding
    def encode(self, src):
        src_mask = self.make_src_mask(src)
        return self.encoder(src, src_mask), src_mask

    def decode(self, tgt, enc_out, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.linear(dec_out)


# --------------------------------------------------------------------------- #
# Decoder-only stack (GPT-style): embedding + pos encoding -> N x
# (masked self-attention + Add&Norm + Feed Forward + Add&Norm).
# This is structurally identical to an EncoderLayer stack - the only
# difference from the BERT-style encoder below is which mask you feed it
# (causal vs. padding-only) - so we reuse EncoderLayer directly instead of
# duplicating it.
# --------------------------------------------------------------------------- #
class GPTStack(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)
        for layer in self.layers:
            h = layer(h, mask)
        return h


class TransformerDecoderOnly(nn.Module):
    """
    GPT-style causal language model: no encoder, no cross-attention.
    Just: Input Embedding -> + Positional Encoding -> N x (Masked
    Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm) -> Linear.

    Good fit for creative text generation (Shakespeare, names, poetry, etc.)
    - trains lighter than the full encoder-decoder Transformer since there's
    no cross-attention and only one embedding table.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_len: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.stack = GPTStack(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (b,1,1,t)
        subsequent_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        )  # (t, t)
        return pad_mask & subsequent_mask  # broadcasts to (b, 1, t, t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.make_causal_mask(x)
        hidden = self.stack(x, mask)
        return self.lm_head(hidden)  # (batch, seq_len, vocab_size)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None, eos_idx: int = None):
        """Autoregressive sampling loop, one token at a time. Stops early if the
        sequence would exceed this model's positional-encoding capacity (max_len),
        even if eos_idx is never predicted."""
        self.eval()
        room = max(0, self.max_len - idx.size(1))
        steps = min(max_new_tokens, room)
        for _ in range(steps):
            logits = self.forward(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            if eos_idx is not None and (next_token == eos_idx).all():
                break
        return idx


# --------------------------------------------------------------------------- #
# Encoder-only stack (BERT-style): embedding + pos encoding -> N x
# (Multi-Head Attention -> Add&Norm -> Feed Forward -> Add&Norm), bidirectional
# (no causal mask - every position can attend to every other position).
# Useful for classification (e.g. your sentiment project), masked-language-
# modeling pretraining, or just producing contextual embeddings.
# --------------------------------------------------------------------------- #
class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        num_classes: int = None,
        pooling: str = "mean",  # "mean" or "cls" (assumes token 0 of each sequence is a [CLS]-like token)
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.pooling = pooling
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

        # optional heads - leave num_classes=None to just get contextual embeddings back
        self.classifier = nn.Linear(d_model, num_classes) if num_classes else None
        self.mlm_head = nn.Linear(d_model, vocab_size)  # handy for masked-LM pretraining

    def make_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (b,1,1,seq_len)

    def forward(self, x: torch.Tensor, task: str = "classify"):
        mask = self.make_pad_mask(x)
        hidden = self.encoder(x, mask)  # (batch, seq_len, d_model)

        if task == "embed":
            return hidden

        if task == "mlm":
            return self.mlm_head(hidden)  # (batch, seq_len, vocab_size)

        # task == "classify"
        if self.classifier is None:
            raise ValueError("num_classes was not set - construct with num_classes=N to classify.")

        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:  # mean-pool over real (non-pad) tokens
            pad_mask_2d = (x != self.pad_idx).unsqueeze(-1).float()  # (b, seq_len, 1)
            summed = (hidden * pad_mask_2d).sum(dim=1)
            counts = pad_mask_2d.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts

        return self.classifier(pooled)  # (batch, num_classes)


if __name__ == "__main__":
    # quick sanity check - encoder-decoder (translation-style)
    model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000,
                         d_model=128, num_layers=2, num_heads=4, d_ff=512)
    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 12))
    out = model(src, tgt)
    print("Transformer (encoder-decoder) logits shape:", out.shape)  # (2, 12, 1000)

    # decoder-only (GPT-style, e.g. for creative text generation)
    gpt = TransformerDecoderOnly(vocab_size=1000, d_model=128, num_layers=2, num_heads=4, d_ff=512)
    x = torch.randint(1, 1000, (2, 16))
    logits = gpt(x)
    print("TransformerDecoderOnly (GPT-style) logits shape:", logits.shape)  # (2, 16, 1000)
    generated = gpt.generate(x[:, :4], max_new_tokens=10, top_k=20)
    print("generated sequence shape:", generated.shape)  # (2, 14)

    # encoder-only (BERT-style, e.g. for your sentiment classifier)
    bert = TransformerEncoderOnly(vocab_size=1000, d_model=128, num_layers=2,
                                   num_heads=4, d_ff=512, num_classes=3)
    cls_logits = bert(x, task="classify")
    print("TransformerEncoderOnly (BERT-style) classification logits shape:", cls_logits.shape)  # (2, 3)
