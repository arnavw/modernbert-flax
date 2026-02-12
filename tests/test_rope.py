import jax.numpy as jnp
import numpy as np
import torch

from modeling import apply_rotary_pos_emb, build_rope_cache


def test_rope_cache_matches_pt(pt_model):
  """Verify cos/sin tables match the PyTorch rotary embedding."""

  seq_len = 128
  head_dim = 64

  for layer_type, theta in [("full_attention", 160_000.0), ("sliding_attention", 10_000.0)]:
    cos_TK, sin_TK = build_rope_cache(seq_len, head_dim, theta)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, 768)
    cos_pt, sin_pt = pt_model.model.rotary_emb(dummy, position_ids, layer_type=layer_type)
    cos_pt = cos_pt.squeeze(0).detach().numpy()
    sin_pt = sin_pt.squeeze(0).detach().numpy()

    np.testing.assert_allclose(
      np.array(cos_TK), cos_pt, atol=1e-6, err_msg=f"cos mismatch for {layer_type}"
    )
    np.testing.assert_allclose(
      np.array(sin_TK), sin_pt, atol=1e-6, err_msg=f"sin mismatch for {layer_type}"
    )


def test_apply_rotary_pos_emb_matches_pt(pt_model):
  """Verify apply_rotary_pos_emb matches PyTorch output."""

  rng = np.random.RandomState(123)
  B, H, T, K = 2, 12, 32, 64

  q_np_BHTK = rng.randn(B, H, T, K).astype(np.float32)
  k_np_BHTK = rng.randn(B, H, T, K).astype(np.float32)

  cos_TK, sin_TK = build_rope_cache(T, K, 160_000.0)
  cos_11TK = cos_TK[None, None, :, :]
  sin_11TK = sin_TK[None, None, :, :]
  q_flax_BHTK, k_flax_BHTK = apply_rotary_pos_emb(
    jnp.array(q_np_BHTK), jnp.array(k_np_BHTK), cos_11TK, sin_11TK
  )

  from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb as pt_apply

  position_ids = torch.arange(T).unsqueeze(0)
  dummy = torch.zeros(1, T, 768)
  cos_pt, sin_pt = pt_model.model.rotary_emb(dummy, position_ids, layer_type="full_attention")
  q_pt_BHTK, k_pt_BHTK = pt_apply(torch.tensor(q_np_BHTK), torch.tensor(k_np_BHTK), cos_pt, sin_pt)

  np.testing.assert_allclose(np.array(q_flax_BHTK), q_pt_BHTK.numpy(), atol=1e-6)
  np.testing.assert_allclose(np.array(k_flax_BHTK), k_pt_BHTK.numpy(), atol=1e-6)
