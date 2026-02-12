import jax.numpy as jnp
import numpy as np
import pytest
import torch

REPO_ID = "answerdotai/ModernBERT-base"


@pytest.fixture(scope="session")
def pt_model():
  """Load the PyTorch ModernBERT-base model in float32."""
  from transformers import AutoModelForMaskedLM

  model = AutoModelForMaskedLM.from_pretrained(REPO_ID, torch_dtype=torch.float32)
  model.eval()
  return model


@pytest.fixture(scope="session")
def pt_config(pt_model):
  """Return the HF config dict."""
  return pt_model.config.to_dict()


@pytest.fixture(scope="session")
def flax_model():
  """Load the Flax ModernBERT-base model via from_pretrained."""
  from hub import from_pretrained

  return from_pretrained(REPO_ID, dtype=jnp.float32)


@pytest.fixture(scope="session")
def sample_input():
  """Deterministic random [2, 128] int32 token ids."""
  rng = np.random.RandomState(42)
  return rng.randint(0, 50368, size=(2, 128)).astype(np.int32)
