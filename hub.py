import json
import os
import tempfile
from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx import traversals
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.flax import load_file as load_flax
from safetensors.flax import save_file as save_flax
from safetensors.numpy import load_file as load_numpy

from modeling import Config, ModernBertMLM


def flatten_state(model: ModernBertMLM) -> dict[str, jnp.ndarray]:
  """Extract all Param values as a flat dict with dot-separated keys."""
  pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
  return {".".join(str(p) for p in k): v for k, v in traversals.flatten_mapping(pure).items()}


def load_state(model: ModernBertMLM, weights: dict[str, jnp.ndarray]) -> None:
  """Load a flat dot-keyed dict of arrays into model parameters."""
  nested = traversals.unflatten_mapping(weights, sep=".")
  state = nnx.state(model, nnx.Param)
  nnx.replace_by_pure_dict(state, nested)
  nnx.update(model, state)


def weight_map(cfg: Config):
  """Yield (flax_key, pt_key, transpose) for every weight."""
  yield "model.embeddings.tok_embeddings.embedding", "model.embeddings.tok_embeddings.weight", False
  yield "model.embeddings.norm.scale", "model.embeddings.norm.weight", False
  yield "model.embeddings.norm.bias", "model.embeddings.norm.bias", False

  for i in range(cfg.num_hidden_layers):
    p = f"model.layers.{i}"
    if i != 0:
      yield f"{p}.attn_norm.scale", f"{p}.attn_norm.weight", False
      yield f"{p}.attn_norm.bias", f"{p}.attn_norm.bias", False
    for n in ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"):
      yield f"{p}.{n}.kernel", f"{p}.{n}.weight", True
      yield f"{p}.{n}.bias", f"{p}.{n}.bias", False
    yield f"{p}.mlp_norm.scale", f"{p}.mlp_norm.weight", False
    yield f"{p}.mlp_norm.bias", f"{p}.mlp_norm.bias", False

  yield "model.final_norm.scale", "model.final_norm.weight", False
  yield "model.final_norm.bias", "model.final_norm.bias", False
  yield "head.dense.kernel", "head.dense.weight", True
  yield "head.dense.bias", "head.dense.bias", False
  yield "head.norm.scale", "head.norm.weight", False
  yield "head.norm.bias", "head.norm.bias", False


def convert_pt_to_flax(pt_state: dict[str, np.ndarray], cfg: Config) -> dict[str, np.ndarray]:
  """Convert PyTorch state dict to flat Flax weight dict."""
  flax = {}
  for fk, pk, T in weight_map(cfg):
    if pk in pt_state:
      flax[fk] = pt_state[pk].T if T else pt_state[pk]
  if "decoder.bias" in pt_state:
    flax["decoder_bias"] = pt_state["decoder.bias"]
  return flax


def convert_flax_to_pt(model: ModernBertMLM) -> dict[str, np.ndarray]:
  """Extract PyTorch-compatible state dict from Flax model."""
  flax = {k: np.array(v) for k, v in flatten_state(model).items()}
  pt = {}
  for fk, pk, T in weight_map(model.cfg):
    if fk in flax:
      pt[pk] = flax[fk].T if T else flax[fk]
  pt["decoder.weight"] = flax["model.embeddings.tok_embeddings.embedding"]
  if "decoder_bias" in flax:
    pt["decoder.bias"] = flax["decoder_bias"]
  return pt


def from_pretrained(
  repo_id: str,
  dtype: jnp.dtype = jnp.bfloat16,
) -> ModernBertMLM:
  config_path = hf_hub_download(repo_id, "config.json")
  with open(config_path) as f:
    hf_config = json.load(f)
  cfg = Config.from_hf(hf_config)

  model = ModernBertMLM(cfg, rngs=nnx.Rngs(0))

  try:
    weights_path = hf_hub_download(repo_id, "flax_model.safetensors")
    weights = load_flax(weights_path)
  except EntryNotFoundError:
    weights_path = hf_hub_download(repo_id, "model.safetensors")
    pt_state = load_numpy(weights_path)
    weights = convert_pt_to_flax(pt_state, cfg)

  if dtype != jnp.float32:
    weights = {k: v.astype(dtype) for k, v in weights.items()}

  load_state(model, weights)
  return model


def push_to_hub(
  model: ModernBertMLM,
  repo_id: str,
  config: Config | None = None,
  private: bool = False,
) -> str:
  cfg = config or model.cfg
  api = HfApi()
  api.create_repo(repo_id, exist_ok=True, private=private)

  with tempfile.TemporaryDirectory() as tmpdir:
    save_flax(flatten_state(model), os.path.join(tmpdir, "flax_model.safetensors"))

    with open(os.path.join(tmpdir, "config.json"), "w") as f:
      json.dump(asdict(cfg), f, indent=2)

    api.upload_folder(folder_path=tmpdir, repo_id=repo_id)

  return f"https://huggingface.co/{repo_id}"
