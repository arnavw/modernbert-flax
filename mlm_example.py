import jax.numpy as jnp
from transformers import AutoTokenizer

from hub import from_pretrained

REPO_ID = "answerdotai/ModernBERT-base"


def main():
  tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
  model = from_pretrained(REPO_ID)

  text = "The capital of France is [MASK]."
  input_ids = jnp.array(tokenizer(text, return_tensors="np")["input_ids"])
  logits = model(input_ids)

  mask_pos = (input_ids[0] == tokenizer.mask_token_id).argmax()
  mask_logits = logits[0, mask_pos]
  top5 = mask_logits.argsort()[::-1][:5]

  print(f"Input: {text}")
  for i, tid in enumerate(top5, 1):
    print(f"  {i}. {tokenizer.decode(int(tid))!r} (logit={float(mask_logits[tid]):.2f})")


if __name__ == "__main__":
  main()
