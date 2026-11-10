# Open and inspect real weights

Time to open the binary. In this lesson you set up a proper weights workbench on your Mac, download Qwen3.5-0.8B, and read its 487 tensors like a file format you're reverse engineering. You'll measure them, peek inside the 9B without loading it, read the quantized GGUF that LM Studio runs, and then do the centerpiece: diff the original model against the MIT-edited one and find, yourself, exactly which 131,072 numbers Astra changed and what direction it removed.

> [!GOALS]
> - Build a Python environment for weight work on Apple Silicon and understand each package
> - Read safetensors files (including the raw header) and map every tensor name to a part of the architecture
> - Compute weight statistics: norms per layer, singular values, effective rank
> - Inspect the 9B lazily without loading 19 GB, and read GGUF tensors with the `gguf` package
> - Diff two checkpoints and verify the MIT study's edit: tensors, columns, hashes, and the removed direction

## Set up the workbench

Everything in this module lives in `~/ai-lab/weights/`. If you made a venv in the previous lesson, reuse it.

```bash
mkdir -p ~/ai-lab/weights/models && cd ~/ai-lab/weights
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch "transformers>=5.3" accelerate safetensors huggingface_hub numpy gguf datasets
```

What each package is for:

| Package | Role | Analogy |
|---|---|---|
| `torch` | Tensors, math, autograd, the MPS (Metal) GPU backend | The CPU and instruction set |
| `transformers` | Model code: knows how to wire Qwen3.5's layers from a config | The loader that knows the binary's layout |
| `safetensors` | Reads and writes `.safetensors` weight files, lazily if you want | A safe parser for the file format |
| `huggingface_hub` | Downloads repos; provides the `hf` CLI | `git` + `curl` for model repos |
| `accelerate` | Device placement and loading models bigger than memory | A memory manager |
| `numpy` | Plain arrays; what `gguf` returns | |
| `gguf` | Reads GGUF files (the format LM Studio and llama.cpp use) | A parser for the other binary format |
| `datasets` | Loads benchmarks like GSM8K (used in later lessons) | |

> [!TIP]
> `uv` is a much faster drop-in for these steps (`uv venv`, `uv pip install ...`). Optional; plain `pip` is fine.

Check the install:

```bash
python - <<'EOF'
import torch, transformers, safetensors, gguf
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())
x = torch.randn(1024, 1024, device="mps", dtype=torch.bfloat16)
print("bf16 matmul on MPS ok:", (x @ x).shape)
EOF
```

You want transformers 5.3 or newer (Qwen3.5 support arrived in 5.2.0; 5.22.x is current as of this writing) and a successful bf16 matmul on MPS. If MPS fails, everything in this module still runs on CPU, just slower.

> [!MAC]
> Unified memory means "GPU memory" and "RAM" are the same 16 GB. Qwen3.5-0.8B in bf16 is 1.75 GB, so you can hold two copies plus working tensors easily. The 9B in bf16 is 19.3 GB and **will not fit**; you'll never load it whole. Quit LM Studio (or at least unload its model) before heavy PyTorch work so the two don't fight over memory.

### A shared helper module

Later lessons reuse a few helpers. Save this once:

```python title="common.py"
"""Shared helpers for the ~/ai-lab/weights labs."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported MPS ops fall back to CPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.expanduser("~/ai-lab/weights")
BASE_DIR = os.path.join(ROOT, "models", "Qwen3.5-0.8B")          # official, pinned revision
EDITED_DIR = os.path.join(ROOT, "models", "Qwen3.5-0.8B-wsp")    # RikiyaT's edited model
SHARD = "model.safetensors-00001-of-00001.safetensors"           # the 0.8B's single weight file


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(path: str = BASE_DIR, device: str | None = None, dtype=torch.bfloat16):
    """Load tokenizer + text-only causal LM (vision tower and MTP head are skipped)."""
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"  # needed for batched generation
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    return tok, model.to(device).eval()


def chat(tok, user: str, thinking: bool = False) -> str:
    """Render one user turn with the model's chat template."""
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )


def ckpt_to_module(name: str) -> str:
    """'model.language_model.layers.15.mlp.down_proj.weight' -> 'model.layers.15.mlp.down_proj.weight'."""
    return name.replace("model.language_model.", "model.", 1)
```

`ckpt_to_module` exists because the checkpoint stores the multimodal layout (`model.language_model.*`) while the text-only class transformers gives you names its modules `model.layers.*`. Same tensors, different path prefix.

## Download the models

You need two checkpoints: the official Qwen3.5-0.8B at the exact revision the MIT edit was built against, and the edited release.

```bash
cd ~/ai-lab/weights
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --local-dir models/Qwen3.5-0.8B
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming --exclude "eval/*" \
  --local-dir models/Qwen3.5-0.8B-wsp
ls -lh models/*/*.safetensors
```

Each is about 1.75 GB. The weights live in one file with a slightly odd name, `model.safetensors-00001-of-00001.safetensors`, plus `model.safetensors.index.json` mapping each tensor to that file.

> [!NOTE]
> `hf` is the current Hugging Face CLI (it replaced `huggingface-cli`). Run `hf download --help` to confirm flags on your version. Pinning `--revision` matters for science: if Qwen ever updates the repo, your diff against the edited model would silently break.

## The safetensors format, byte by byte

You'll appreciate this one: `.safetensors` exists because the older PyTorch format (`.bin`/`.pt`) is a Python **pickle**, and unpickling untrusted data can execute arbitrary code. Safetensors is pure data:

```text
[8 bytes: header length N, little-endian u64]
[N bytes: JSON header {"tensor.name": {"dtype": "BF16", "shape": [1024, 3584], "data_offsets": [start, end]}, ...}]
[raw tensor bytes, back to back]
```

Parse it with no ML libraries at all:

```python title="peek_header.py"
"""Read a safetensors header with only the standard library."""
import json
import struct
import sys


def main(path: str) -> None:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))     # header length
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    print(f"header bytes: {n}, tensors: {len(header)}, metadata: {meta}")
    for name in sorted(header)[:5]:
        print(name, header[name]["dtype"], header[name]["shape"], header[name]["data_offsets"])


if __name__ == "__main__":
    main(sys.argv[1])
```

```bash
python peek_header.py models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors
```

Because the header sits at the start of the file, you can even read it **remotely** with an HTTP range request, without downloading the weights. That's how you'll peek at the 9B later.

## Every tensor, mapped to the architecture

```python title="list_tensors.py"
"""List every tensor in a safetensors checkpoint, grouped by architectural role."""
import math
import sys
from collections import defaultdict

from safetensors import safe_open

from common import BASE_DIR, SHARD


def role(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision tower"
    if name.startswith("mtp."):
        return "MTP head"
    if "embed_tokens" in name:
        return "embeddings"
    if ".linear_attn." in name:
        return "linear attention (Gated DeltaNet)"
    if ".self_attn." in name:
        return "full attention"
    if ".mlp." in name:
        return "MLP"
    if "norm" in name:
        return "norms"
    return "other"


def main(path: str) -> None:
    totals, counts = defaultdict(int), defaultdict(int)
    with safe_open(path, framework="pt") as f:
        names = list(f.keys())
        for name in names:
            sl = f.get_slice(name)               # metadata only, no data read
            shape, dtype = sl.get_shape(), sl.get_dtype()
            n = math.prod(shape)
            totals[role(name)] += n
            counts[role(name)] += 1
            if ".layers.15." in name or ".layers.0." in name or "layers" not in name and "visual" not in name:
                print(f"{name:62s} {dtype:5s} {shape}")
    print()
    for r in sorted(totals, key=totals.get, reverse=True):
        print(f"{r:36s} {counts[r]:4d} tensors {totals[r]:>13,d} params")
    print(f"{'TOTAL':36s} {len(names):4d} tensors {sum(totals.values()):>13,d} params")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{BASE_DIR}/{SHARD}")
```

```bash
python list_tensors.py
```

You should see 487 tensors and 873,438,784 parameters in total: about 752.4 M in the language model, 100.6 M in the vision tower and 20.5 M in the multi-token-prediction (MTP) head. A quarter of the language model is the embedding table alone: 248,320 tokens × 1,024 = 254,279,680 numbers.

Here's what the names mean for one linear-attention layer (0) and one full-attention layer (15), with the shapes you'll see. Shapes are `[out, in]`: a matrix of shape `[1024, 3584]` takes 3,584 numbers in and puts 1,024 out.

| Tensor (prefix `model.language_model.layers.N.`) | Shape (0.8B) | What it does |
|---|---|---|
| `input_layernorm.weight` | [1024] | RMSNorm scale before the attention-type block |
| `linear_attn.in_proj_qkv.weight` | [6144, 1024] | Queries, keys, values for Gated DeltaNet: 16 heads × 128 dims each for Q, K and V (3 × 2,048) |
| `linear_attn.in_proj_z.weight` | [2048, 1024] | Output gate |
| `linear_attn.in_proj_a.weight`, `in_proj_b.weight` | [16, 1024] each | Per-head gates: how fast the state decays (a) and how strongly to write (b) |
| `linear_attn.conv1d.weight` | [6144, 1, 4] | Short causal convolution (kernel 4) over recent tokens |
| `linear_attn.A_log`, `dt_bias` | [16] each (A_log is F32) | Decay-rate parameters per head |
| `linear_attn.norm.weight` | [128] (F32) | Per-head gated RMSNorm |
| `linear_attn.out_proj.weight` | [1024, 2048] | **Writes** the block's result into the residual stream |
| `self_attn.q_proj.weight` (full layers) | [4096, 1024] | 8 heads × 256 dims, doubled: queries plus an output gate |
| `self_attn.k_proj.weight`, `v_proj.weight` | [512, 1024] each | 2 KV heads × 256 dims (grouped-query attention) |
| `self_attn.q_norm.weight`, `k_norm.weight` | [256] each | Per-head normalization of queries and keys |
| `self_attn.o_proj.weight` | [1024, 2048] | **Writes** attention output into the residual stream |
| `post_attention_layernorm.weight` | [1024] | RMSNorm scale before the MLP |
| `mlp.gate_proj.weight`, `mlp.up_proj.weight` | [3584, 1024] each | Read the stream into 3,584 neurons (SiLU-gated) |
| `mlp.down_proj.weight` | [1024, 3584] | **Writes** the neurons' output into the residual stream |

Plus the globals: `model.language_model.embed_tokens.weight` [248320, 1024], `model.language_model.norm.weight` [1024] (final norm), the `mtp.*` tensors, and `model.visual.*` (12 vision blocks of width 768 and a merger that projects to 1,024). There is **no `lm_head.weight`**: the 0.8B ties its output layer to the embedding table. Your 9B does have a separate `lm_head.weight`.

> [!CONCEPT] Writers
> The tensors that add things into the residual stream: the **embeddings**, every attention output (`o_proj` or `out_proj`), and every `mlp.down_proj`. If you want to stop the model from ever writing a direction, these are the matrices you edit. Remember this list; it's the whole of [Direction ablation](#/weights/direction-ablation).

> [!NOTE]
> A detail that will matter later: layers 15 and 19, where the MIT edit lives, are both **full-attention** layers in the 0.8B. The edit touched their `mlp.down_proj`, not their attention.

## Weight statistics

Raw numbers mean little in isolation; statistics across layers show structure.

> [!CONCEPT] Singular values and effective rank
> Any matrix can be split into a sum of rank-one pieces, each "read along one direction, write along another, with strength s". Those strengths are the **singular values** (the SVD). If a few are huge and the rest tiny, the matrix mostly does a few things. **Effective rank** turns that into one number: treat the normalized singular values as probabilities and take exp(entropy). It ranges from 1 (does one thing) to the full rank (spreads effort evenly).

```python title="weight_stats.py"
"""Per-layer norms of the residual-stream writers, and SVD of one down_proj."""
import torch
from safetensors import safe_open

from common import BASE_DIR, SHARD

P = "model.language_model.layers.{}."


def effective_rank(s: torch.Tensor) -> float:
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def main() -> None:
    with safe_open(f"{BASE_DIR}/{SHARD}", framework="pt") as f:
        keys = set(f.keys())
        print("layer  type     |attn_out|  |down_proj|  mean(post_attn_norm)")
        for i in range(24):
            full = (P.format(i) + "self_attn.o_proj.weight") in keys
            attn = P.format(i) + ("self_attn.o_proj.weight" if full else "linear_attn.out_proj.weight")
            a = f.get_tensor(attn).float()
            d = f.get_tensor(P.format(i) + "mlp.down_proj.weight").float()
            g = f.get_tensor(P.format(i) + "post_attention_layernorm.weight").float()
            print(f"{i:5d}  {'FULL' if full else 'lin '}  {a.norm():10.1f}  {d.norm():11.1f}  {g.mean():10.3f}")

        w = f.get_tensor(P.format(15) + "mlp.down_proj.weight").float()   # [1024, 3584]
    s = torch.linalg.svdvals(w)
    print(f"\nlayer 15 down_proj: top-5 singular values {[round(float(x), 2) for x in s[:5]]}")
    print(f"smallest {float(s[-1]):.3f}, effective rank {effective_rank(s):.0f} of max {len(s)}")
    col_norms = w.norm(dim=0)                                          # one norm per neuron
    top = torch.topk(col_norms, 5)
    print("neurons with the largest write vectors:", [int(i) for i in top.indices])


if __name__ == "__main__":
    main()
```

```bash
python weight_stats.py
```

What to look for, without me promising exact numbers:

- **Norms by layer.** Do the writers get stronger in later layers? Are the full-attention layers (every fourth) different from their linear neighbors?
- **Effective rank.** Trained MLP matrices usually use most of their dimensions; an effective rank far below the maximum would mean the layer does a few dominant things.
- **Column norms.** Each `down_proj` column is one neuron's write vector. A few neurons with huge write vectors are worth noting; they often carry outsized effects.

## Peek inside the 9B without loading it

Your model's base, `Qwen/Qwen3.5-9B`, is four shards (5.28 + 5.34 + 5.37 + 3.33 GB = 19.3 GB). You can read its whole header **remotely** with two HTTP range requests:

```python title="remote_header.py"
"""Read a safetensors header over HTTP without downloading the weights."""
import json
import struct
import sys

import requests

REPO = "Qwen/Qwen3.5-9B"


def header(shard: str) -> dict:
    url = f"https://huggingface.co/{REPO}/resolve/main/{shard}"
    n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}).content)[0]
    raw = requests.get(url, headers={"Range": f"bytes=8-{7 + n}"}).content
    return json.loads(raw)


if __name__ == "__main__":
    shard = sys.argv[1] if len(sys.argv) > 1 else "model.safetensors-00001-of-00004.safetensors"
    h = header(shard)
    h.pop("__metadata__", None)
    for name in sorted(h):
        if ".layers.15." in name or "lm_head" in name or "embed" in name:
            print(name, h[name]["dtype"], h[name]["shape"])
```

```bash
pip install requests
python remote_header.py
```

You'll see layer 15's `mlp.down_proj.weight` at `[4096, 12288]`, `lm_head.weight` and `embed_tokens.weight` at `[248320, 4096]`. The index file (`model.safetensors.index.json`) tells you which shard holds what; for example, layer 15's `down_proj` is in shard 1 and layer 19's in shard 2, and some layers have their MLP matrices split across shards. That matters when you edit the 9B shard by shard.

When you do have the shards locally (you'll download them in [Direction ablation](#/weights/direction-ablation)), `safe_open` reads exactly one tensor, or even a slice of one, using memory-mapped I/O:

```python title="lazy_9b.py"
"""Read one tensor (or a slice) from a 9B shard without loading the shard."""
import os

from safetensors import safe_open

SHARD = os.path.expanduser("~/ai-lab/weights/models/Qwen3.5-9B/model.safetensors-00001-of-00004.safetensors")
NAME = "model.language_model.layers.15.mlp.down_proj.weight"

with safe_open(SHARD, framework="pt") as f:
    sl = f.get_slice(NAME)
    print(NAME, sl.get_dtype(), sl.get_shape())
    cols = sl[:, :64]                   # reads only these columns' bytes: 4096 x 64
    print("first 64 columns:", cols.shape, "norm", float(cols.float().norm()))
    full = f.get_tensor(NAME)           # ~100 MB in bf16, still tiny next to 19 GB
    print("full tensor:", full.shape, full.dtype)
```

## Reading the GGUF that LM Studio runs

Your LM Studio model is a GGUF file: a different container with its own tensor names and quantized blocks (see [Precision, quantization and file formats](#/foundations/quantization)). The `gguf` package (maintained in the llama.cpp repo) reads it.

```bash
find ~/.lmstudio/models -name "*.gguf" | grep -i qwen
```

```python title="read_gguf.py"
"""List GGUF tensors and dequantize one MLP down-projection."""
import sys
from collections import Counter

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main(path: str, name: str = "blk.15.ffn_down.weight") -> None:
    r = GGUFReader(path)                     # memory-mapped: nothing big is read yet
    arch = r.fields["general.architecture"]
    print("architecture:", bytes(arch.parts[arch.data[0]]).decode())
    print("tensor types:", Counter(t.tensor_type.name for t in r.tensors).most_common())
    for t in r.tensors:
        if t.name.startswith("blk.15.") or t.name in ("token_embd.weight", "output.weight"):
            print(f"{t.name:32s} {t.tensor_type.name:6s} ggml-shape {list(t.shape)}")
    t = next(t for t in r.tensors if t.name == name)
    w = dequantize(t.data, t.tensor_type)    # float32 NumPy array
    print(f"\n{name}: {t.tensor_type.name} -> float32 {w.shape}, norm {np.linalg.norm(w):.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
```

```bash
python read_gguf.py "$(find ~/.lmstudio/models -name '*Q4_K_M*.gguf' | grep -i qwen3.5-9b | head -1)"
```

Things you'll notice:

- **Different names.** llama.cpp renames tensors: `model.language_model.layers.15.mlp.down_proj.weight` becomes `blk.15.ffn_down.weight`, embeddings become `token_embd.weight`, the LM head `output.weight`.
- **Reversed shape order.** GGUF lists dimensions innermost-first, so a PyTorch `[4096, 12288]` shows as `[12288, 4096]`. The dequantized NumPy array comes back in PyTorch order; print `.shape` to confirm.
- **Mixed types.** "Q4_K_M" is a recipe, not one type: most big matrices are `Q4_K`, some (often `ffn_down` and attention value/output matrices) are `Q6_K`, and norms stay `F32`.

> [!WARNING]
> `gguf` can **dequantize** K-quants (`Q4_K`, `Q6_K` and others) but can only **quantize** simpler types such as `Q8_0`, `Q4_0` and `BF16`. That's one reason you won't edit your Q4_K_M file directly; you'll edit full-precision safetensors and re-quantize with llama.cpp's own tools. The full reasoning is in [Direction ablation](#/weights/direction-ablation).

## Centerpiece: find the MIT edit yourself

The study says two tensors changed, 64 columns each, by removing one direction. Don't believe it: diff the checkpoints.

> [!LAB] Diff the original and the edited Qwen3.5-0.8B
> 1. Save as `~/ai-lab/weights/compare_checkpoints.py`:
> ```python title="compare_checkpoints.py"
> """Compare two safetensors checkpoints tensor by tensor and dissect any edits."""
> import hashlib
> import sys
>
> import torch
> from safetensors import safe_open
>
> from common import BASE_DIR, EDITED_DIR, SHARD
>
> # SHA-256 of the raw bf16 bytes, as published on the edited model's card
> PUBLISHED = {
>     "model.language_model.layers.15.mlp.down_proj.weight": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
>     "model.language_model.layers.19.mlp.down_proj.weight": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
> }
>
>
> def sha(t: torch.Tensor) -> str:
>     # NumPy has no bfloat16, so view the same 2-byte values as int16 to get the raw bytes
>     return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
>
>
> def dissect(name: str, w0: torch.Tensor, w1: torch.Tensor) -> dict:
>     changed = w0 != w1
>     cols = torch.where(changed.any(dim=0))[0]            # neurons whose write vector changed
>     rows = torch.where(changed.any(dim=1))[0]
>     a, b = w0.float(), w1.float()
>     delta = b - a
>     print(f"\n{name}")
>     print(f"  changed entries {int(changed.sum()):,} | columns {len(cols)} | rows {len(rows)}")
>     print(f"  relative change |dW|/|W| = {float(delta.norm() / a.norm()):.4f}")
>     h = sha(w1)
>     print(f"  sha256 {h[:16]}... matches card: {PUBLISHED.get(name) == h}")
>     print(f"  changed columns: {cols.tolist()}")
>
>     # Is the change one direction? SVD of the changed block of dW
>     u_mat, s, _ = torch.linalg.svd(delta[:, cols], full_matrices=False)
>     print(f"  singular values of dW: {[round(float(x), 3) for x in s[:4]]}  (s2/s1 = {float(s[1] / s[0]):.4f})")
>     u = u_mat[:, 0]                                     # unit vector: the removed direction
>
>     # How much of each column's component along u was removed? (strength estimate)
>     before = u @ a[:, cols]
>     after = u @ b[:, cols]
>     strength = 1 - after / before                       # if W' = W - s*u(u^T W) then this equals s
>     print(f"  along u: strength median {float(strength.median()):.2f} "
>           f"(range {float(strength.min()):.2f} to {float(strength.max()):.2f})")
>     resid = delta[:, cols] - torch.outer(u, u @ delta[:, cols])   # change NOT along u
>     print(f"  change orthogonal to u: {float(resid.norm() / delta.norm()):.4f} of the total")
>     return {"u": u, "cols": cols}
>
>
> def main(path0: str, path1: str) -> None:
>     directions = {}
>     with safe_open(path0, framework="pt") as f0, safe_open(path1, framework="pt") as f1:
>         k0, k1 = set(f0.keys()), set(f1.keys())
>         print(f"tensors: {len(k0)} vs {len(k1)}; only in A: {len(k0 - k1)}, only in B: {len(k1 - k0)}")
>         same = 0
>         for name in sorted(k0 & k1):
>             w0, w1 = f0.get_tensor(name), f1.get_tensor(name)
>             if w0.shape == w1.shape and torch.equal(w0, w1):
>                 same += 1
>                 continue
>             directions[name] = dissect(name, w0, w1)
>         print(f"\nidentical tensors: {same}; edited tensors: {len(directions)}")
>     names = list(directions)
>     if len(names) == 2:
>         cos = float(directions[names[0]]["u"] @ directions[names[1]]["u"])
>         print(f"cosine between the two removed directions: {cos:+.4f}")
>     torch.save(directions, "astra_directions.pt")       # used in the direction-ablation lesson
>
>
> if __name__ == "__main__":
>     p0 = sys.argv[1] if len(sys.argv) > 2 else f"{BASE_DIR}/{SHARD}"
>     p1 = sys.argv[2] if len(sys.argv) > 2 else f"{EDITED_DIR}/{SHARD}"
>     main(p0, p1)
> ```
> 2. Run it (compares all 487 tensors; takes a minute):
> ```bash
> python compare_checkpoints.py | tee compare.txt
> ```
> 3. **Check the claims.** You should find exactly two edited tensors, `layers.15.mlp.down_proj.weight` and `layers.19.mlp.down_proj.weight`; 64 changed columns in each; changed entries of 65,260 and 64,928 (130,188 total); relative changes near 0.106 and 0.081; and both hashes matching the card. If anything differs, check you downloaded the pinned base revision.
> 4. **Check "one direction".** If the edit really removed a single direction from each layer, the singular values of the change should be one big number followed by tiny ones (bf16 rounding noise), so `s2/s1` should be very small, and the change orthogonal to `u` should be a small fraction. Is it?
> 5. **Read the strength.** The logs describe the final edit as "strength 6". Your median estimate says how much of each column's component along the unit direction was removed: 1.0 would mean "projected out exactly"; above 1 means the component was pushed past zero and reversed. What does your number say, and does it fit the log's description?
> 6. **Shared or separate?** Look at the cosine between the two layers' directions. Near ±1 means Astra used one shared direction for both layers; noticeably lower means each layer got its own (closely related) direction. The explanations file speaks of "a shared repetition direction" and of "saved directions"; your number settles which.
> 7. **Neuron 19:11.** Astra's notes say ablating neuron 11 in layer 19 alone gave a small, inconclusive gain. Is column 11 in layer 19's list of 64?
>
> You've just audited a published AI result from the raw weights. Keep `astra_directions.pt`; you'll compare your own loop direction against it.

> [!TIP]
> The logs repo also ships `exp8_math/code/apply_edit.py`, which rebuilds the edited tensors from the base file plus `edit/edit.npz` and checks hashes. Its usage line is `python apply_edit.py <base model.safetensors> <run's edit dir> <out>`; it expects the language-model tensors of the base model and a helper module (`st_numpy`) from the same folder. It's a good second, independent check if you cloned the code folder in [the study lesson](#/weights/mit-study).

## Optional: what did the uncensoring change?

The same diff idea answers a question you may have about your own model: which tensors did HauhauCS modify relative to the official Qwen3.5-9B? The card doesn't say. A fair comparison needs both in similar precision: the HauhauCS BF16 GGUF (17 GB, readable with `GGUFReader` and `dequantize`) against the official safetensors, one tensor at a time, mapping `blk.N.ffn_down.weight` to `model.language_model.layers.N.mlp.down_proj.weight` and so on. Comparing against your Q4_K_M file instead mostly measures quantization noise. Treat this as an advanced project with a large disk bill (about 36 GB for both), not a requirement.

## Recap

- A clean venv with torch (MPS), transformers ≥ 5.3, safetensors, huggingface_hub, accelerate and gguf is your weights workbench; `common.py` holds shared helpers.
- Safetensors is a JSON header plus raw bytes, safe to parse and readable lazily (even remotely via HTTP ranges); pickled `.bin` files are not safe to load from strangers.
- Qwen3.5-0.8B has 487 tensors and 873 M parameters (752 M language, 101 M vision, 20 M MTP); shapes are `[out, in]`, and the residual-stream writers are embeddings, `o_proj`/`out_proj` and `down_proj`.
- SVD and effective rank summarize what a matrix does; column norms of `down_proj` show each neuron's write strength.
- GGUF renames tensors (`blk.N.ffn_down.weight`), reverses shape order and mixes quant types; `gguf` can dequantize K-quants but not write them.
- Diffing the original and edited 0.8B confirms the study: two `down_proj` tensors, 64 columns each, a rank-one change along one direction.

## Check yourself

1. Why is loading a random `.bin` checkpoint from the internet riskier than loading a `.safetensors` file?
   <details><summary>Answer</summary>

   `.bin`/`.pt` files are Python pickles; unpickling can execute arbitrary code embedded in the file, which is effectively remote code execution. Safetensors is a JSON header plus raw numbers, with no code path to execute. (transformers now defaults to safetensors and warns about pickles, but the risk is why.)

   </details>
2. In `mlp.down_proj.weight` of shape `[1024, 3584]`, what is one column, and why did the MIT edit operate on columns?
   <details><summary>Answer</summary>

   With PyTorch's `[out, in]` layout, column j holds the 1,024 numbers that neuron j adds to the residual stream when it fires (its write vector). Editing a column changes what that one neuron writes. Astra picked the 64 neurons that wrote most along the loop direction and removed that direction from their write vectors, leaving the other 3,520 neurons untouched.

   </details>
3. The diff shows singular values of ΔW like [large, tiny, tiny, ...]. What does that tell you?
   <details><summary>Answer</summary>

   The change is (nearly) rank one: every changed column moved along the same direction, with only rounding noise elsewhere. That's the signature of "subtract a multiple of one vector u from each column", i.e. removing (or reversing) a single direction, and the top left singular vector is that direction.

   </details>
4. Why does `blk.15.ffn_down.weight` in the GGUF show shape `[12288, 4096]` while the safetensors shows `[4096, 12288]`?
   <details><summary>Answer</summary>

   GGUF/ggml lists dimensions innermost-first (ne0 is the fastest-varying dimension), while PyTorch lists them outermost-first. It's the same matrix with the same memory layout, described in opposite order.

   </details>
5. How can you list the tensors of a 19 GB checkpoint on Hugging Face without downloading it?
   <details><summary>Answer</summary>

   Read the safetensors header with HTTP range requests: fetch bytes 0–7 to get the header length N, then bytes 8 to 7+N to get the JSON header with every tensor's name, dtype, shape and offsets. The index JSON tells you which shard to query for which tensor.

   </details>

## Go deeper
- [Safetensors documentation](https://huggingface.co/docs/safetensors/index) — format spec, lazy loading and why it exists.
- [Transformers: Qwen3.5 model docs](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5) — the classes, config fields and the hybrid layer layout.
- [gguf-py in the llama.cpp repo](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) — the reader/writer and quant implementations used above.
- [lm_understand-logs, experiment 8](https://github.com/RikiyaT/lm_understand-logs) — the edit manifest and `apply_edit.py` you just cross-checked.
