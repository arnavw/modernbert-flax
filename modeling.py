"""ModernBERT in Flax NNX

Dimension key:
  B = batch size
  T = sequence length
  D = hidden size
  F = intermediate size
  V = vocab size
  H = number of attention heads
  K = head dim
"""

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class Config:
  vocab_size: int = 50368
  hidden_size: int = 768
  num_hidden_layers: int = 22
  num_attention_heads: int = 12
  intermediate_size: int = 1152
  max_position_embeddings: int = 8192
  norm_eps: float = 1e-5
  norm_bias: bool = False
  pad_token_id: int = 50283
  global_attn_every_n_layers: int = 3
  local_attention: int = 128
  global_rope_theta: float = 160_000.0
  local_rope_theta: float = 10_000.0
  attention_bias: bool = False
  mlp_bias: bool = False
  decoder_bias: bool = True
  classifier_bias: bool = False
  embedding_dropout: float = 0.0
  attention_dropout: float = 0.0
  mlp_dropout: float = 0.0
  initializer_range: float = 0.02

  @property
  def head_dim(self) -> int:
    return self.hidden_size // self.num_attention_heads

  @property
  def layer_types(self) -> list[str]:
    return [
      "global" if (i % self.global_attn_every_n_layers == 0) else "sliding"
      for i in range(self.num_hidden_layers)
    ]

  @classmethod
  def from_hf(cls, d: dict) -> "Config":
    kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
    if "rope_parameters" in d:
      rp = d["rope_parameters"]
      if "full_attention" in rp:
        kw["global_rope_theta"] = float(rp["full_attention"].get("rope_theta", 160_000.0))
      if "sliding_attention" in rp:
        kw["local_rope_theta"] = float(rp["sliding_attention"].get("rope_theta", 10_000.0))
    return cls(**kw)


class MLP(nnx.Module):
  def __init__(self, cfg: Config, *, rngs):
    self.Wi = nnx.Linear(
      cfg.hidden_size,
      cfg.intermediate_size * 2,
      use_bias=cfg.mlp_bias,
      rngs=rngs,
    )
    self.Wo = nnx.Linear(
      cfg.intermediate_size,
      cfg.hidden_size,
      use_bias=cfg.mlp_bias,
      rngs=rngs,
    )

  def __call__(self, hidden_BTD: jax.Array) -> jax.Array:
    hidden_BTF = self.Wi(hidden_BTD)
    gated, gate = jnp.split(hidden_BTF, 2, axis=-1)
    return self.Wo(jax.nn.gelu(gated, approximate=False) * gate)


class Embeddings(nnx.Module):
  def __init__(self, cfg: Config, *, rngs):
    self.token_embeddings = nnx.Embed(
      num_embeddings=cfg.vocab_size,
      features=cfg.hidden_size,
      rngs=rngs,
    )
    self.norm = nnx.LayerNorm(
      cfg.hidden_size,
      epsilon=cfg.norm_eps,
      use_bias=cfg.norm_bias,
      rngs=rngs,
    )

  def __call__(self, token_ids_BT: jax.Array) -> jax.Array:
    return self.norm(self.token_embeddings(token_ids_BT))


def rotate_half(x: jax.Array) -> jax.Array:
  d = x.shape[-1] // 2
  return jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def apply_rotary_pos_emb(
  q: jax.Array,
  k: jax.Array,
  cos: jax.Array,
  sin: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  orig_dtype = q.dtype
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
  return q.astype(orig_dtype), k.astype(orig_dtype)


class Attention(nnx.Module):
  def __init__(self, cfg: Config, *, rngs):
    self.num_heads = cfg.num_attention_heads
    self.head_dim = cfg.head_dim
    self.Wqkv = nnx.Linear(
      cfg.hidden_size,
      3 * cfg.hidden_size,
      use_bias=cfg.attention_bias,
      rngs=rngs,
    )
    self.Wo = nnx.Linear(
      cfg.hidden_size,
      cfg.hidden_size,
      use_bias=cfg.attention_bias,
      rngs=rngs,
    )

  def __call__(
    self,
    hidden_BTD: jax.Array,
    cos_TK: jax.Array,
    sin_TK: jax.Array,
    mask: jax.Array | None = None,
    local_window_size: tuple[int, int] | None = None,
  ) -> jax.Array:
    B, T, _ = hidden_BTD.shape

    qkv = self.Wqkv(hidden_BTD).reshape(B, T, 3, self.num_heads, self.head_dim)
    q, k, v = jnp.unstack(qkv, axis=2)

    cos = cos_TK[None, :T, None, :]
    sin = sin_TK[None, :T, None, :]
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    h = jax.nn.dot_product_attention(
      q,
      k,
      v,
      mask=mask,
      is_causal=False,
      local_window_size=local_window_size,
    ).reshape(B, T, -1)
    return self.Wo(h)


class EncoderLayer(nnx.Module):
  def __init__(self, cfg: Config, layer_idx: int, *, rngs):
    self.layer_idx = layer_idx
    self.has_attn_norm = layer_idx != 0
    if self.has_attn_norm:
      self.attn_norm = nnx.LayerNorm(
        cfg.hidden_size,
        epsilon=cfg.norm_eps,
        use_bias=cfg.norm_bias,
        rngs=rngs,
      )
    self.attn = Attention(cfg, rngs=rngs)
    self.mlp_norm = nnx.LayerNorm(
      cfg.hidden_size,
      epsilon=cfg.norm_eps,
      use_bias=cfg.norm_bias,
      rngs=rngs,
    )
    self.mlp = MLP(cfg, rngs=rngs)

  def __call__(
    self,
    hidden_BTD: jax.Array,
    cos_TK: jax.Array,
    sin_TK: jax.Array,
    mask: jax.Array | None = None,
    local_window_size: tuple[int, int] | None = None,
  ) -> jax.Array:
    normed_BTD = self.attn_norm(hidden_BTD) if self.has_attn_norm else hidden_BTD
    hidden_BTD = hidden_BTD + self.attn(normed_BTD, cos_TK, sin_TK, mask, local_window_size)
    hidden_BTD = hidden_BTD + self.mlp(self.mlp_norm(hidden_BTD))
    return hidden_BTD


def build_rope_cache(seq_len: int, head_dim: int, theta: float) -> tuple[jax.Array, jax.Array]:
  inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
  pos = jnp.arange(seq_len, dtype=jnp.float32)
  freqs = jnp.outer(pos, inv_freq)
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  return jnp.cos(emb), jnp.sin(emb)


class ModernBert(nnx.Module):
  def __init__(self, cfg: Config, *, rngs):
    self.cfg = cfg
    self.embeddings = Embeddings(cfg, rngs=rngs)
    self.layers = nnx.List([EncoderLayer(cfg, i, rngs=rngs) for i in range(cfg.num_hidden_layers)])
    self.final_norm = nnx.LayerNorm(
      cfg.hidden_size,
      epsilon=cfg.norm_eps,
      use_bias=cfg.norm_bias,
      rngs=rngs,
    )
    self.global_cos, self.global_sin = build_rope_cache(
      cfg.max_position_embeddings, cfg.head_dim, cfg.global_rope_theta
    )
    self.local_cos, self.local_sin = build_rope_cache(
      cfg.max_position_embeddings, cfg.head_dim, cfg.local_rope_theta
    )

  def __call__(
    self,
    token_ids_BT: jax.Array,
    padding_mask_BT: jax.Array | None = None,
  ) -> jax.Array:
    hidden_BTD = self.embeddings(token_ids_BT)

    mask = padding_mask_BT[:, None, None, :] if padding_mask_BT is not None else None
    half_w = self.cfg.local_attention // 2

    layer_types = self.cfg.layer_types
    for i, layer in enumerate(self.layers):
      is_global = layer_types[i] == "global"
      cos_TK = self.global_cos if is_global else self.local_cos
      sin_TK = self.global_sin if is_global else self.local_sin
      window = None if is_global else (half_w, half_w)
      hidden_BTD = layer(hidden_BTD, cos_TK, sin_TK, mask=mask, local_window_size=window)

    return self.final_norm(hidden_BTD)


class MLMHead(nnx.Module):
  def __init__(self, cfg: Config, *, rngs):
    self.dense = nnx.Linear(
      cfg.hidden_size,
      cfg.hidden_size,
      use_bias=cfg.classifier_bias,
      rngs=rngs,
    )
    self.norm = nnx.LayerNorm(
      cfg.hidden_size,
      epsilon=cfg.norm_eps,
      use_bias=cfg.norm_bias,
      rngs=rngs,
    )

  def __call__(self, hidden_BTD: jax.Array) -> jax.Array:
    return self.norm(jax.nn.gelu(self.dense(hidden_BTD), approximate=False))


class ModernBertMLM(nnx.Module):
  def __init__(self, cfg: Config, *, rngs):
    self.cfg = cfg
    self.model = ModernBert(cfg, rngs=rngs)
    self.head = MLMHead(cfg, rngs=rngs)
    if cfg.decoder_bias:
      self.decoder_bias = nnx.Param(jnp.zeros(cfg.vocab_size))

  def __call__(
    self,
    token_ids_BT: jax.Array,
    padding_mask_BT: jax.Array | None = None,
  ) -> jax.Array:
    hidden_BTD = self.model(token_ids_BT, padding_mask_BT)
    hidden_BTD = self.head(hidden_BTD)
    logits_BTV = self.model.embeddings.token_embeddings.attend(hidden_BTD)
    if self.cfg.decoder_bias:
      logits_BTV = logits_BTV + self.decoder_bias[...]
    return logits_BTV
