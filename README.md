# ModernBERT in JAX/Flax

A minimal implementation of [ModernBERT](https://huggingface.co/blog/modernbert) in [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html). Produces equivalent outputs to the PyTorch reference.

`modeling.py` is the NNX implementation
`hub.py` loads and converts weights from HF

## Usage

```python
from hub import from_pretrained

model = from_pretrained("answerdotai/ModernBERT-base")
```

## Tests

```bash
uv run --group test pytest tests/ -x -v
```
