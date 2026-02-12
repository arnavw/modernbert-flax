import jax.numpy as jnp
import numpy as np
import pytest
import torch

from modeling import Config


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  """Cosine similarity between two flattened arrays."""
  a_flat = a.ravel().astype(np.float64)
  b_flat = b.ravel().astype(np.float64)
  return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)))


def _run_both_models(pt_model, flax_model, input_ids_BT):
  """Run both models on identical input, return (flax_logits_BTV, pt_logits_BTV) as np arrays."""
  with torch.no_grad():
    pt_out = pt_model(torch.tensor(input_ids_BT, dtype=torch.long))
    pt_logits_BTV = pt_out.logits.numpy()
  flax_logits_BTV = np.array(flax_model(jnp.array(input_ids_BT)))
  return flax_logits_BTV, pt_logits_BTV


def test_full_model_logits_equivalence(pt_model, flax_model, sample_input):
  """Ultimate gate test: same input -> same logits, verified by both atol and cosine similarity."""
  flax_logits_BTV, pt_logits_BTV = _run_both_models(pt_model, flax_model, sample_input)

  np.testing.assert_allclose(flax_logits_BTV, pt_logits_BTV, atol=1e-3)

  cos_sim = _cosine_similarity(flax_logits_BTV, pt_logits_BTV)
  assert cos_sim > 0.99999, f"Cosine similarity {cos_sim:.8f} below threshold 0.99999"


@pytest.mark.parametrize("seed", [0, 42, 123, 999, 65535])
def test_multi_seed_logits(pt_model, flax_model, seed):
  """Verify logits match across multiple random inputs, not just one lucky seed."""
  rng = np.random.RandomState(seed)
  input_ids_BT = rng.randint(0, 50368, size=(2, 128)).astype(np.int32)
  flax_logits_BTV, pt_logits_BTV = _run_both_models(pt_model, flax_model, input_ids_BT)

  max_diff = np.max(np.abs(flax_logits_BTV - pt_logits_BTV))
  mean_diff = np.mean(np.abs(flax_logits_BTV - pt_logits_BTV))
  cos_sim = _cosine_similarity(flax_logits_BTV, pt_logits_BTV)

  assert max_diff < 5e-3, f"seed={seed}: max_diff={max_diff:.6e} exceeds 5e-3"
  assert mean_diff < 5e-5, f"seed={seed}: mean_diff={mean_diff:.6e} exceeds 5e-5"
  assert cos_sim > 0.99999, f"seed={seed}: cos_sim={cos_sim:.8f} below 0.99999"


def test_with_padding_mask(pt_model, flax_model):
  """Verify equivalence with actual padded batches (different sequence lengths).

  Constructs a [2, 32] batch where:
    - Row 0: 32 real tokens
    - Row 1: 20 real tokens + 12 padding tokens
  """

  cfg = Config()
  rng = np.random.RandomState(77)

  T = 32
  input_ids_BT = rng.randint(0, 50368, size=(2, T)).astype(np.int32)
  input_ids_BT[1, 20:] = cfg.pad_token_id

  pt_attention_mask = torch.ones(2, T, dtype=torch.float32)
  pt_attention_mask[1, 20:] = 0.0

  with torch.no_grad():
    pt_out = pt_model(
      torch.tensor(input_ids_BT, dtype=torch.long),
      attention_mask=pt_attention_mask,
    )
    pt_logits_BTV = pt_out.logits.numpy()

  padding_mask_BT = np.ones((2, T), dtype=bool)
  padding_mask_BT[1, 20:] = False

  flax_logits_BTV = np.array(
    flax_model(
      jnp.array(input_ids_BT),
      padding_mask_BT=jnp.array(padding_mask_BT),
    )
  )

  np.testing.assert_allclose(
    flax_logits_BTV[0],
    pt_logits_BTV[0],
    atol=1e-3,
    err_msg="Padded batch: row 0 (no padding) mismatch",
  )
  np.testing.assert_allclose(
    flax_logits_BTV[1, :20],
    pt_logits_BTV[1, :20],
    atol=1e-3,
    err_msg="Padded batch: row 1 (real tokens) mismatch",
  )

  valid_flax = np.concatenate([flax_logits_BTV[0], flax_logits_BTV[1, :20]])
  valid_pt = np.concatenate([pt_logits_BTV[0], pt_logits_BTV[1, :20]])
  cos_sim = _cosine_similarity(valid_flax, valid_pt)
  assert cos_sim > 0.99999, f"Padded batch: cos_sim={cos_sim:.8f} below 0.99999"


def test_edge_single_token(pt_model, flax_model):
  """Verify equivalence with a single token (T=1)."""
  input_ids_BT = np.array([[1000]], dtype=np.int32)
  flax_logits_BTV, pt_logits_BTV = _run_both_models(pt_model, flax_model, input_ids_BT)

  np.testing.assert_allclose(flax_logits_BTV, pt_logits_BTV, atol=1e-3)
  cos_sim = _cosine_similarity(flax_logits_BTV, pt_logits_BTV)
  assert cos_sim > 0.99999, f"Single token: cos_sim={cos_sim:.8f}"
