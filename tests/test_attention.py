import jax.numpy as jnp
import numpy as np
import torch

from modeling import Config, build_rope_cache


def _get_pt_position_embeddings(pt_model, T, layer_type):
  """Get cos, sin from PyTorch model for a given layer type."""
  position_ids = torch.arange(T).unsqueeze(0)
  dummy = torch.zeros(1, T, 768)
  cos, sin = pt_model.model.rotary_emb(dummy, position_ids, layer_type=layer_type)
  return cos, sin


def test_global_attention_equivalence(pt_model, flax_model):
  """Verify a single global attention layer output matches."""
  rng = np.random.RandomState(42)
  B, T, D = 2, 64, 768
  hidden_BTD = rng.randn(B, T, D).astype(np.float32)

  pt_layer = pt_model.model.layers[0]
  flax_layer = flax_model.model.layers[0]

  cos_pt, sin_pt = _get_pt_position_embeddings(pt_model, T, "full_attention")
  with torch.no_grad():
    pt_out_BTD = pt_layer.attn(
      torch.tensor(hidden_BTD),
      position_embeddings=(cos_pt, sin_pt),
    )
    pt_out_BTD = pt_out_BTD[0].numpy()

  cfg = Config()
  cos_TK, sin_TK = build_rope_cache(T, cfg.head_dim, cfg.global_rope_theta)
  flax_out_BTD = np.array(flax_layer.attn(jnp.array(hidden_BTD), cos_TK, sin_TK, mask=None))

  np.testing.assert_allclose(flax_out_BTD, pt_out_BTD, atol=2e-5)


def test_sliding_attention_equivalence(pt_model, flax_model):
  """Verify a single sliding-window attention layer output matches."""
  rng = np.random.RandomState(42)
  B, T, D = 2, 64, 768
  hidden_BTD = rng.randn(B, T, D).astype(np.float32)

  pt_layer = pt_model.model.layers[1]
  flax_layer = flax_model.model.layers[1]

  cfg = Config()

  cos_pt, sin_pt = _get_pt_position_embeddings(pt_model, T, "sliding_attention")
  sliding_window = cfg.local_attention // 2
  row = torch.arange(T).unsqueeze(1)
  col = torch.arange(T).unsqueeze(0)
  pt_mask = torch.where(
    (row - col).abs() <= sliding_window,
    torch.tensor(0.0),
    torch.tensor(float("-inf")),
  )
  pt_mask = pt_mask.unsqueeze(0).unsqueeze(0)

  with torch.no_grad():
    pt_out_BTD = pt_layer.attn(
      torch.tensor(hidden_BTD),
      position_embeddings=(cos_pt, sin_pt),
      attention_mask=pt_mask,
    )
    pt_out_BTD = pt_out_BTD[0].numpy()

  half_w = cfg.local_attention // 2
  cos_TK, sin_TK = build_rope_cache(T, cfg.head_dim, cfg.local_rope_theta)
  flax_out_BTD = np.array(
    flax_layer.attn(jnp.array(hidden_BTD), cos_TK, sin_TK, local_window_size=(half_w, half_w))
  )

  np.testing.assert_allclose(flax_out_BTD, pt_out_BTD, atol=2e-5)
