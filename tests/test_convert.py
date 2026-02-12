import numpy as np

from hub import convert_flax_to_pt


def test_round_trip(pt_model, flax_model):
  """Verify PT -> Flax -> PT round trip preserves weights."""
  pt_state_orig = {k: v.numpy() for k, v in pt_model.state_dict().items()}

  pt_state_rt = convert_flax_to_pt(flax_model)

  for key in pt_state_orig:
    if key == "decoder.weight":
      np.testing.assert_allclose(
        pt_state_rt[key],
        pt_state_orig["model.embeddings.tok_embeddings.weight"],
        atol=1e-7,
      )
    elif key in pt_state_rt:
      np.testing.assert_allclose(
        pt_state_rt[key], pt_state_orig[key], atol=1e-7, err_msg=f"Round-trip mismatch for {key}"
      )
