"""
model.py
--------
A from-scratch Transformer implementation that started as the classic
Vaswani et al. encoder/decoder diagram and has since been updated with the
architectural choices that current (2023+) open-weight models actually use:

    Inputs -> Input Embedding -> [Encoder x N] -------\
                                                         \
    Outputs(shifted right) -> Output Embedding -> [Decoder x N] -> Linear -> Softmax

Each Encoder block (Pre-LN):  RMSNorm -> Self-Attn(RoPE, GQA) -> +res
                               -> RMSNorm -> SwiGLU FFN -> +res
Each Decoder block (Pre-LN):  RMSNorm -> Masked Self-Attn(RoPE, GQA) -> +res
                               -> RMSNorm -> Cross-Attn(GQA) -> +res
                               -> RMSNorm -> SwiGLU FFN -> +res

What changed from the original paper and why:
  - No more sinusoidal PositionalEncoding added to the embeddings. Position is
    instead injected inside attention via Rotary Position Embeddings (RoPE),
    which rotate Q/K as a function of position so the attention score falls
    out as a function of *relative* position (Su et al., "RoFormer", 2021).
  - Grouped-Query Attention (GQA): K/V use fewer heads than Q (num_kv_heads <
    num_heads), each shared across a group of query heads. Shrinks the K/V
    projections (and, at inference time, the KV-cache) with little quality
    loss (Ainslie et al., "GQA", 2023). num_kv_heads == num_heads recovers
    plain multi-head attention; num_kv_heads == 1 is multi-query attention.
  - Attention itself is computed with torch.nn.functional.
    scaled_dot_product_attention instead of manual matmul+softmax. On CUDA
    this dispatches to a fused FlashAttention / memory-efficient kernel that
    never materializes the full (seq_len, seq_len) score matrix (Dao et al.,
    "FlashAttention", 2022) - same math, much less memory traffic.
  - Pre-LN instead of Post-LN: normalization moved *inside* each residual
    branch (norm -> sublayer -> add), instead of after it. This is what lets
    deep Transformers train stably without a delicate LR-warmup schedule.
  - RMSNorm instead of LayerNorm: normalizes by root-mean-square only, no
    mean-centering, no bias term - fewer statistics per token, cheaper, and
    is what most current models normalize with (Zhang & Sennrich, 2019).
  - SwiGLU instead of ReLU in the feed-forward block: a gated unit
    (down_proj(SiLU(gate_proj(x)) * up_proj(x))) that measurably improves
    quality per FLOP over a plain Linear->ReLU->Linear MLP (Shazeer, "GLU
    Variants Improve Transformer", 2020).
  - Linear layers inside attention/FFN drop their bias terms (bias=False),
    the standard pairing with RMSNorm in current architectures (LLaMA,
    Mistral, ...) - one less set of parameters that RMSNorm's rescaling
    makes redundant.

No external attention/transformer libraries are used - everything is still
plain torch.nn (plus the one call into F.scaled_dot_product_attention) so you
can see exactly what every piece does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# GQA helper: pick a sensible default number of KV heads for a given number
# of query heads, when the caller doesn't specify one explicitly.
# --------------------------------------------------------------------------- #
def default_num_kv_heads(num_heads: int) -> int:
    """Prefers a 4:1 query:kv ratio (a common real-world GQA setting), falling
    back to a coarser ratio (or 1, i.e. multi-query attention) if num_heads
    isn't evenly divisible by 4."""
    for cand in (num_heads // 4, num_heads // 2, num_heads, 1):
        if cand >= 1 and num_heads % cand == 0:
            return cand
    return num_heads  # unreachable (cand=1 always divides), kept for safety


# --------------------------------------------------------------------------- #
# Rotary Position Embedding (RoPE) - replaces the old sinusoidal
# PositionalEncoding class. Instead of adding a position vector to the
# embeddings once at the input, RoPE rotates each attention head's Q and K
# vectors by an angle proportional to token position, separately in every
# layer. The dot product of two rotated vectors depends only on their
# *relative* position, which is the property that made RoPE the default
# choice for LLaMA/Mistral/Qwen/etc.
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                # (max_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)          # (max_len, head_dim)
        # cached, not learned, and not part of the optimizer's state
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype, offset: int = 0):
        # offset lets a KV-cached decoding step ask for the rotation angles of
        # token positions [offset, offset+seq_len) instead of always [0, seq_len)
        # - e.g. generating the 51st token needs position 50's angle, not 0's,
        # even though only one new token is being processed this call.
        end = offset + seq_len
        assert end <= self.cos_cached.size(0), (
            f"position {end} exceeds RotaryEmbedding max_len "
            f"{self.cos_cached.size(0)} - construct the model with a larger max_len"
        )
        return (
            self.cos_cached[offset:end].to(device=device, dtype=dtype),
            self.sin_cached[offset:end].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (batch, num_heads, seq_len, head_dim); cos, sin: (seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, head_dim), broadcasts over batch/heads
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# --------------------------------------------------------------------------- #
# GQA helper: broadcast each KV head across its group of query heads so
# Q/K/V end up with matching head counts right before the attention call.
# --------------------------------------------------------------------------- #
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: (batch, num_kv_heads, seq_len, head_dim)
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


# --------------------------------------------------------------------------- #
# Attention: GQA + RoPE + fused/Flash scaled-dot-product-attention.
# Replaces the old MultiHeadAttention class. Used for encoder self-attention,
# decoder masked self-attention (both use_rope=True), and decoder cross-
# attention (use_rope=False - see DecoderLayer for why).
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = None,
                 dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.dropout_p = dropout

        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None, rope=None, kv_cache=None, use_cache=False):
        """
        kv_cache: optional (past_k, past_v), each (batch, num_kv_heads, past_len, d_k) -
            K/V for tokens already seen, cached at num_kv_heads width (before
            repeat_kv broadcasts them out to num_heads). None means "no history yet"
            (e.g. the first/prefill call).
        use_cache: if True, also return the updated (k, v) pair (past_k/v with this
            call's new K/V appended) so the caller can pass it back in next step.
            Costs nothing extra to compute - k/v below already contain exactly that -
            it's just whether we bother returning it.
        """
        batch, q_len, _ = query.shape

        q = self._split_heads(self.w_q(query), self.num_heads)     # (b, H,   q_len, d_k)
        k = self._split_heads(self.w_k(key), self.num_kv_heads)    # (b, Hkv, new_len, d_k)
        v = self._split_heads(self.w_v(value), self.num_kv_heads)  # (b, Hkv, new_len, d_k)

        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepend cached history (already projected + RoPE'd at its own
        # positions in an earlier call) so this call only ever pays for the
        # *new* token(s) - the whole point of the cache. Concatenating here,
        # before repeat_kv, is what lets GQA's smaller num_kv_heads shrink the
        # cache itself, not just the per-step projection cost.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)  # (b, Hkv, past_len + new_len, d_k)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v) if use_cache else None

        # GQA: broadcast each of the Hkv key/value heads across its group of
        # query heads so shapes match for scaled_dot_product_attention.
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Fused attention kernel - uses FlashAttention / memory-efficient
        # backends on CUDA instead of ever forming the full attention matrix.
        # mask: bool, broadcastable to (batch, num_heads, q_len, k_len);
        # True = attend, False = masked out (same convention as before).
        out = F.scaled_dot_product_attention(
            q, k_rep, v_rep, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0,
        )  # (b, H, q_len, d_k)

        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.num_heads * self.d_k)
        out = self.w_o(out)

        if use_cache:
            return out, present_kv
        return out


# --------------------------------------------------------------------------- #
# RMSNorm - replaces nn.LayerNorm. No mean-subtraction, no bias.
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


# --------------------------------------------------------------------------- #
# SwiGLU feed-forward - replaces the plain Linear->ReLU->Linear FeedForward.
# --------------------------------------------------------------------------- #
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# --------------------------------------------------------------------------- #
# One Encoder block (Pre-LN): RMSNorm -> Self-Attn -> +res -> RMSNorm ->
# SwiGLU FFN -> +res
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = Attention(d_model, num_heads, num_kv_heads, dropout, use_rope=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask, rope, kv_cache=None, use_cache=False):
        h = self.norm1(x)
        attn_out = self.self_attn(h, h, h, mask, rope, kv_cache=kv_cache, use_cache=use_cache)
        if use_cache:
            attn_out, present_kv = attn_out
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(h))

        if use_cache:
            return x, present_kv
        return x


# --------------------------------------------------------------------------- #
# One Decoder block (Pre-LN): RMSNorm -> Masked Self-Attn(RoPE) -> +res
#                              -> RMSNorm -> Cross-Attn -> +res
#                              -> RMSNorm -> SwiGLU FFN -> +res
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.masked_self_attn = Attention(d_model, num_heads, num_kv_heads, dropout, use_rope=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        # Cross-attention mixes decoder queries against encoder keys/values that live on
        # a different position axis (source tokens vs. target tokens), so there's no
        # single shared coordinate frame to rotate Q and K into the way RoPE needs.
        # We leave cross-attention un-rotated (use_rope=False); position still reaches
        # it indirectly, since the encoder output it reads from was itself built by
        # RoPE'd self-attention layers.
        self.cross_attn = Attention(d_model, num_heads, num_kv_heads, dropout, use_rope=False)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask, rope):
        h = self.norm1(x)
        x = x + self.dropout1(self.masked_self_attn(h, h, h, tgt_mask, rope))

        h = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(h, enc_out, enc_out, src_mask))

        h = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(h))
        return x


# --------------------------------------------------------------------------- #
# Encoder stack: Input Embedding -> N x EncoderLayer -> final RMSNorm
# (no more "+ Positional Encoding" here - RoPE is applied inside attention,
# per layer, via the shared rotary_emb table below)
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 num_kv_heads=None, max_len=5000, dropout=0.1, rope_theta=10000.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        d_k = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(d_k, max_len, rope_theta)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, num_kv_heads, dropout) for _ in range(num_layers)]
        )
        self.norm_f = RMSNorm(d_model)

    def forward(self, src, src_mask):
        x = self.dropout(self.embedding(src))
        rope = self.rotary_emb(src.size(1), x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, src_mask, rope)
        return self.norm_f(x)


# --------------------------------------------------------------------------- #
# Decoder stack: Output Embedding -> N x DecoderLayer -> final RMSNorm
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 num_kv_heads=None, max_len=5000, dropout=0.1, rope_theta=10000.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        d_k = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(d_k, max_len, rope_theta)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, num_kv_heads, dropout) for _ in range(num_layers)]
        )
        self.norm_f = RMSNorm(d_model)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.dropout(self.embedding(tgt))
        rope = self.rotary_emb(tgt.size(1), x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask, rope)
        return self.norm_f(x)


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
        num_kv_heads: int = None,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        num_kv_heads = num_kv_heads or default_num_kv_heads(num_heads)

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff,
                                num_kv_heads, max_len, dropout, rope_theta)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff,
                                num_kv_heads, max_len, dropout, rope_theta)

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
# Decoder-only stack (GPT-style): embedding -> N x (Pre-LN masked self-attn
# + Pre-LN SwiGLU FFN) -> final RMSNorm. Structurally identical to the
# Encoder stack (same EncoderLayer blocks) - the only difference from the
# BERT-style encoder below is which mask you feed it (causal vs. padding-
# only) - so we reuse EncoderLayer directly instead of duplicating it.
# --------------------------------------------------------------------------- #
class GPTStack(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 num_kv_heads=None, max_len=5000, dropout=0.1, rope_theta=10000.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        d_k = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(d_k, max_len, rope_theta)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, num_kv_heads, dropout) for _ in range(num_layers)]
        )
        self.norm_f = RMSNorm(d_model)

    def forward(self, x, mask, kv_cache=None, use_cache=False, position_offset=0):
        """
        kv_cache: optional list of per-layer (past_k, past_v) pairs (or None for a
            fresh/prefill call, i.e. no history yet).
        position_offset: absolute position of x[:, 0] in the full sequence so far -
            0 on the prefill call, then the running sequence length on every
            single-new-token step after that, so RoPE keeps rotating each new
            token by its true position instead of always starting over at 0.
        """
        h = self.dropout(self.embedding(x))
        rope = self.rotary_emb(x.size(1), h.device, h.dtype, offset=position_offset)

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            if use_cache:
                h, present_kv = layer(h, mask, rope, kv_cache=layer_kv, use_cache=True)
                new_cache.append(present_kv)
            else:
                h = layer(h, mask, rope)

        h = self.norm_f(h)
        if use_cache:
            return h, new_cache
        return h


class TransformerDecoderOnly(nn.Module):
    """
    GPT-style causal language model: no encoder, no cross-attention.
    Input Embedding -> N x (Pre-LN Masked Self-Attn[RoPE,GQA] -> +res ->
    Pre-LN SwiGLU FFN -> +res) -> final RMSNorm -> Linear.

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
        num_kv_heads: int = None,
        max_len: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        num_kv_heads = num_kv_heads or default_num_kv_heads(num_heads)
        self.stack = GPTStack(vocab_size, d_model, num_layers, num_heads, d_ff,
                               num_kv_heads, max_len, dropout, rope_theta)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, x: torch.Tensor, past_len: int = 0) -> torch.Tensor:
        """x here is only the *new* chunk being processed this call - the whole
        prompt on a prefill call, a single token on every cached step after
        that. past_len is how many already-cached positions precede it.
        Every new query position may attend to *all* past_len cached keys
        (they're all strictly earlier, so causality is automatically
        satisfied) plus the usual lower-triangular mask among the new
        positions themselves. past_len=0 makes this identical to the old
        tril-only mask."""
        batch, new_len = x.shape
        total_len = past_len + new_len
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (b,1,1,new_len)
        if past_len > 0:
            past_ok = torch.ones((batch, 1, 1, past_len), dtype=torch.bool, device=x.device)
            pad_mask = torch.cat([past_ok, pad_mask], dim=-1)  # (b,1,1,total_len)

        q_pos = torch.arange(past_len, total_len, device=x.device).unsqueeze(1)  # (new_len,1)
        k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)            # (1,total_len)
        causal = k_pos <= q_pos  # (new_len, total_len) - lower-triangular, shifted by past_len

        return pad_mask & causal  # broadcasts to (b, 1, new_len, total_len)

    def forward(self, x: torch.Tensor, kv_cache=None, use_cache: bool = False, past_len: int = 0):
        """
        x: the new token chunk to process (the full prompt on a prefill call,
            or just the newest token on every step after that).
        kv_cache / use_cache / past_len: see GPTStack.forward - plumbed straight
            through. Default (kv_cache=None, use_cache=False) is the original
            recompute-everything forward pass, unchanged, still used for training.
        """
        mask = self.make_causal_mask(x, past_len=past_len)
        if use_cache:
            hidden, new_cache = self.stack(x, mask, kv_cache=kv_cache, use_cache=True,
                                            position_offset=past_len)
            return self.lm_head(hidden), new_cache
        hidden = self.stack(x, mask)
        return self.lm_head(hidden)  # (batch, seq_len, vocab_size)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None, eos_idx: int = None, use_cache: bool = True):
        """Autoregressive sampling loop, one token at a time. Stops early if the
        sequence would exceed this model's positional capacity (max_len - which
        now bounds the RoPE table rather than an additive positional encoding),
        even if eos_idx is never predicted.

        use_cache=True (default): the prompt is run through the model once
        (the "prefill" step) to seed a KV-cache, then every following step
        only does a forward pass on the single newest token, reusing cached
        K/V for every earlier position via the GQA-shrunk cache built above.
        Cost per step drops from O(seq_len) to O(1), so total generation cost
        drops from O(seq_len^2) to O(seq_len) attention work.

        use_cache=False: the old behavior - recompute the full forward pass
        over the whole sequence-so-far at every step. Kept around mainly to
        sanity-check the cached path (see the __main__ block below) and as a
        fallback if you ever need it, since it's strictly simpler.
        """
        self.eval()
        room = max(0, self.max_len - idx.size(1))
        steps = min(max_new_tokens, room)

        def sample(logits_last_step):
            logits_last_step = logits_last_step[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits_last_step, top_k)
                logits_last_step[logits_last_step < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits_last_step, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        if not use_cache:
            for _ in range(steps):
                logits = self.forward(idx)
                next_token = sample(logits)
                idx = torch.cat([idx, next_token], dim=1)
                if eos_idx is not None and (next_token == eos_idx).all():
                    break
            return idx

        kv_cache = None
        past_len = 0
        cur = idx  # first call processes the whole prompt (prefill); after that, just the newest token
        for _ in range(steps):
            logits, kv_cache = self.forward(cur, kv_cache=kv_cache, use_cache=True, past_len=past_len)
            next_token = sample(logits)
            past_len += cur.size(1)
            idx = torch.cat([idx, next_token], dim=1)
            cur = next_token
            if eos_idx is not None and (next_token == eos_idx).all():
                break
        return idx


# --------------------------------------------------------------------------- #
# Encoder-only stack (BERT-style): embedding -> N x (Pre-LN self-attn + Pre-LN
# SwiGLU FFN) -> final RMSNorm, bidirectional (no causal mask - every position
# can attend to every other position). Useful for classification (e.g. your
# sentiment project), masked-language-modeling pretraining, or just producing
# contextual embeddings.
# --------------------------------------------------------------------------- #
class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_kv_heads: int = None,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        num_classes: int = None,
        pooling: str = "mean",  # "mean" or "cls" (assumes token 0 of each sequence is a [CLS]-like token)
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.pooling = pooling
        num_kv_heads = num_kv_heads or default_num_kv_heads(num_heads)
        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff,
                                num_kv_heads, max_len, dropout, rope_theta)

        # optional heads - leave num_classes=None to just get contextual embeddings back
        self.classifier = nn.Linear(d_model, num_classes) if num_classes else None
        self.mlm_head = nn.Linear(d_model, vocab_size)  # handy for masked-LM pretraining
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
    # quick sanity check - encoder-decoder (translation-style), GQA (4 heads, 2 kv heads)
    model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000,
                         d_model=128, num_layers=2, num_heads=4, num_kv_heads=2, d_ff=512)
    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 12))
    out = model(src, tgt)
    print("Transformer (encoder-decoder) logits shape:", out.shape)  # (2, 12, 1000)

    # decoder-only (GPT-style, e.g. for creative text generation)
    gpt = TransformerDecoderOnly(vocab_size=1000, d_model=128, num_layers=2, num_heads=4,
                                  num_kv_heads=2, d_ff=512)
    x = torch.randint(1, 1000, (2, 16))
    logits = gpt(x)
    print("TransformerDecoderOnly (GPT-style) logits shape:", logits.shape)  # (2, 16, 1000)
    generated = gpt.generate(x[:, :4], max_new_tokens=10, top_k=20)
    print("generated sequence shape:", generated.shape)  # (2, 14)

    # KV-cache sanity check: with dropout off and greedy decoding (top_k=1,
    # so there's only ever one nonzero-probability token to sample), the
    # cached and uncached paths have zero randomness and *must* produce the
    # exact same tokens. Any RoPE-offset or cache-concat mistake shows up
    # here as a hard mismatch rather than a subtle quality regression later.
    gpt_check = TransformerDecoderOnly(vocab_size=200, d_model=64, num_layers=3, num_heads=4,
                                        num_kv_heads=2, d_ff=256, max_len=64, dropout=0.0)
    gpt_check.eval()
    prompt = torch.randint(1, 200, (2, 5))
    cached_out = gpt_check.generate(prompt, max_new_tokens=15, top_k=1, use_cache=True)
    uncached_out = gpt_check.generate(prompt, max_new_tokens=15, top_k=1, use_cache=False)
    assert torch.equal(cached_out, uncached_out), "KV-cache path diverged from the uncached path!"
    print("KV-cache sanity check passed: cached and uncached generate() agree exactly.")

    # encoder-only (BERT-style, e.g. for your sentiment classifier)
    bert = TransformerEncoderOnly(vocab_size=1000, d_model=128, num_layers=2,
                                   num_heads=4, num_kv_heads=2, d_ff=512, num_classes=3)
    cls_logits = bert(x, task="classify")
    print("TransformerEncoderOnly (BERT-style) classification logits shape:", cls_logits.shape)  # (2, 3)