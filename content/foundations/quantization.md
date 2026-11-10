# Precision, quantization and file formats

Your 9B model was trained as 17.9 GB of 16-bit numbers. You run it as a 5.6 GB file. This lesson explains how that compression works, from how a computer stores a decimal number up to the K-quant recipes inside your GGUF, how much quality it costs, and which file formats are safe to load. You finish by dissecting your own model file.

> [!GOALS]
> - Explain how FP32, FP16 and BF16 store numbers, and why BF16 became the training standard
> - Quantize a block of weights by hand and measure the error
> - Decode quant names (Q8_0, Q4_K_M, IQ4_XS, UD-Q4_K_XL) and estimate file size from bits per weight
> - Read perplexity and KL-divergence results and pick a quant for 16 GB with evidence
> - Tell safe weight formats from dangerous ones, and inspect a GGUF's metadata and tensors yourself

## Why this matters: size is speed

A model's weights are just a very long list of numbers (see [Inside the transformer](#/foundations/transformer)). Two facts make their storage format matter:

1. **It has to fit.** Your Mac has 16 GB shared between macOS, your apps and the GPU. The BF16 version of your model is 17.9 GB. It cannot run.
2. **Bandwidth sets speed.** When the model generates text, every new token requires reading essentially *all* the weights from memory once (the "decode" phase in [How text gets generated](#/foundations/generation)). The GPU's arithmetic is fast; memory is the bottleneck.

That gives a back-of-envelope ceiling: **tokens per second is at most memory bandwidth divided by model size.**

| File | Size | Ceiling at 100 GB/s (M2/M3 Air) | Ceiling at 120 GB/s (M4 Air) |
|---|---|---|---|
| Q4_K_M | 5.63 GB | 100 / 5.63 = 17.8 tok/s | 21.3 tok/s |
| Q6_K | 7.36 GB | 13.6 tok/s | 16.3 tok/s |
| Q8_0 | 9.53 GB | 10.5 tok/s | 12.6 tok/s |
| BF16 | 17.92 GB | does not fit | does not fit |

Real speeds land below these ceilings, but the ratios hold: halve the bytes, roughly double the speed. Quantization is the main reason a 9B model is usable on a laptop at all.

## Bits and floating point from zero

### Integers

A bit is a 0 or a 1. With n bits you can represent 2 to the power n different values:

- 4 bits: 16 values, e.g. 0 to 15, or -8 to 7
- 8 bits: 256 values, e.g. -128 to 127
- 16 bits: 65,536 values

Integers are cheap and exact, but evenly spaced. Model weights are small fractions like 0.0123 or -0.45, sometimes with rare large outliers, so we need something more flexible.

### Floating point: sign, exponent, mantissa

> [!CONCEPT] Floating point
> Scientific notation in binary. A number is stored as **sign** (plus or minus), **exponent** (which power of two: the *scale*) and **mantissa** (the significant digits: the *precision*). Like writing 6.02 times 10 to the 23: "6.02" is the mantissa, "23" the exponent.

Intuition: **exponent bits buy range** (how big or tiny a number can be), **mantissa bits buy precision** (how many significant digits survive).

```text
FP32  | s | eeeeeeee | mmmmmmmmmmmmmmmmmmmmmmm |  1 + 8 + 23 = 32 bits
FP16  | s | eeeee    | mmmmmmmmmm              |  1 + 5 + 10 = 16 bits
BF16  | s | eeeeeeee | mmmmmmm                 |  1 + 8 +  7 = 16 bits
```

| Format | Exponent / mantissa bits | Largest value | Significant decimal digits | Bytes per weight | Your 8.95B-parameter GGUF would be |
|---|---|---|---|---|---|
| FP32 | 8 / 23 | about 3.4 x 10^38 | about 7 | 4 | about 35.8 GB |
| FP16 | 5 / 10 | 65,504 | about 3 | 2 | about 17.9 GB |
| BF16 | 8 / 7 | about 3.4 x 10^38 | about 2 to 3 | 2 | about 17.9 GB |

Here are real values rounded into each format (computed with Python's `struct`, BF16 by rounding FP32 to its top 16 bits):

| True value | FP32 | FP16 | BF16 |
|---|---|---|---|
| 0.1 | 0.100000001 | 0.0999756 | 0.1000977 |
| 3.14159265 | 3.1415927 | 3.140625 | 3.140625 |
| 70,000 | 70,000 | **infinity** (overflow) | 70,144 |
| 0.00000001 | 0.00000001 | **0** (underflow) | 0.0000000100117 |

FP16 is more *precise* near ordinary values; BF16 has FP32's *range*.

### Why BF16 won

BF16 ("brain float 16", from Google Brain) keeps all 8 exponent bits of FP32 and throws away mantissa bits:

- **Same range as FP32**, so values in training (especially gradients) do not overflow to infinity or underflow to zero. FP16 training needs tricks like loss scaling; BF16 mostly does not.
- **Trivial conversion**: FP32 to BF16 is "keep the top 16 bits and round".
- **Neural networks tolerate imprecision far better than overflow.** A weight of 0.1000977 instead of 0.1 changes almost nothing; an activation that becomes infinity breaks everything.

Qwen3.5-9B's `config.json` says `"dtype": "bfloat16"`: BF16 is the "original" precision of your model. Everything below compares against it.

> [!NOTE]
> Datacenter GPUs also use 8-bit and 4-bit floating-point formats (Qwen publishes `-FP8` versions of its larger Qwen3.5 models; llama.cpp supports MXFP4 for some models). On your Mac the relevant formats are the GGUF integer quants and MLX's quantization.

## How quantization works

> [!CONCEPT] Quantization
> Storing each weight as a small integer (for example 4 bits: 16 possible values) plus a shared **scale** that converts the integers back to real numbers. Dequantized weight = integer times scale. Fewer bits per weight means a smaller file and faster decode, at the cost of rounding error.

> [!CONCEPT] Block (group) quantization
> Weights are split into small consecutive blocks (32 or 256 weights in llama.cpp) and each block gets its own scale. One scale for a whole matrix would be ruined by a single large value; one per block keeps the damage local.

### Worked example: 8 weights to 4 bits

Take a block of eight weights:

```text
w = [0.12, -0.45, 0.33, 0.05, -0.08, 0.91, -0.27, 0.18]
```

**Step 1: compute the scale.** Use symmetric 4-bit integers from -8 to 7. The largest magnitude is 0.91, and we want it to map to 7:

```text
scale = 0.91 / 7 = 0.13
```

**Step 2: divide and round.** q = round(w / scale), clamped to the range -8 to 7.

**Step 3: dequantize** (what the model actually computes with): w' = q times scale.

| w | w / 0.13 | q (4-bit) | w' = q x 0.13 | error w' - w |
|---|---|---|---|---|
| 0.12 | 0.92 | 1 | 0.13 | +0.01 |
| -0.45 | -3.46 | -3 | -0.39 | +0.06 |
| 0.33 | 2.54 | 3 | 0.39 | +0.06 |
| 0.05 | 0.38 | 0 | 0.00 | -0.05 |
| -0.08 | -0.62 | -1 | -0.13 | -0.05 |
| 0.91 | 7.00 | 7 | 0.91 | 0.00 |
| -0.27 | -2.08 | -2 | -0.26 | +0.01 |
| 0.18 | 1.38 | 1 | 0.13 | -0.05 |

Root-mean-square error: **0.043**, about 14% of the average weight magnitude (0.30). Note what happened to small weights: 0.05 became exactly zero.

**Storage:** 8 weights x 4 bits = 32 bits, plus one 16-bit scale = 48 bits, or **6 bits per weight**. With blocks of 32 instead of 8 the scale is shared more widely: (32 x 4 + 16) / 32 = **4.5 bits per weight**. That is exactly llama.cpp's `Q4_0` layout: 32 weights, one FP16 scale, 18 bytes per block.

**The outlier problem.** Replace 0.91 with 0.40 and repeat: the scale drops to 0.40 / 7 = 0.064, the grid gets finer, and the RMS error falls to **0.012**, less than a third. One large weight stretched the grid for the whole block. This is why real schemes use small blocks, keep sensitive tensors at higher precision, and weight errors by importance.

**Scale plus minimum.** Instead of a symmetric grid, map the block's range [min, max] onto 0 to 15: scale = (0.91 - (-0.45)) / 15 = 0.0907, w' = min + q x scale. For the original block the RMS error drops to **0.022**, because no grid levels are wasted on negative values below -0.45. This "scale and min" idea is what `Q4_1` and the K-quants use.

Reproduce it yourself:

```python title="quant_demo.py"
"""Quantize one block of weights to 4 bits, symmetric and with scale+min."""
import math

W = [0.12, -0.45, 0.33, 0.05, -0.08, 0.91, -0.27, 0.18]


def rmse(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def symmetric(w):
    scale = max(abs(x) for x in w) / 7          # largest magnitude -> 7
    q = [max(-8, min(7, round(x / scale))) for x in w]
    return q, [qi * scale for qi in q]


def scale_min(w):
    lo, hi = min(w), max(w)
    scale = (hi - lo) / 15                       # range -> 0..15
    q = [max(0, min(15, round((x - lo) / scale))) for x in w]
    return q, [lo + qi * scale for qi in q]


if __name__ == "__main__":
    for name, fn in [("symmetric", symmetric), ("scale+min", scale_min)]:
        q, deq = fn(W)
        print(f"{name:10} q={q}")
        print(f"{'':10} rmse={rmse(W, deq):.4f}")
```

```text
symmetric  q=[1, -3, 3, 0, -1, 7, -2, 1]
           rmse=0.0432
scale+min  q=[6, 0, 9, 6, 4, 15, 2, 7]
           rmse=0.0223
```

> [!NOTE] Dequantization is nearly free
> At inference time the integers are not converted back into a full-size copy. The GPU kernels unpack each block on the fly inside the matrix multiplication. Extra arithmetic is cheap; memory reads are expensive. That is why a quantized model is faster as well as smaller.

## The llama.cpp quant families

GGUF files are produced by llama.cpp's `llama-quantize` tool. Its quant types fall into three generations. Bits per weight (bpw) below come from the block layouts in llama.cpp's `ggml-common.h` and the descriptions in `llama-quantize`.

### Legacy quants: Q8_0, Q4_0 and friends

Blocks of 32 weights, each with an FP16 scale (the `_1` variants add an FP16 minimum).

| Type | Layout | bpw |
|---|---|---|
| Q8_0 | 32 x 8-bit + scale | 8.5 |
| Q5_1 | 32 x 5-bit + scale + min | 6.0 |
| Q5_0 | 32 x 5-bit + scale | 5.5 |
| Q4_1 | 32 x 4-bit + scale + min | 5.0 |
| Q4_0 | 32 x 4-bit + scale | 4.5 |

Q8_0 is effectively lossless and is the usual "reference quant". Q4_0 is simple and fast on some hardware, but noticeably worse than Q4_K_M at nearly the same size (numbers below). Unsloth says it ships Q4_0 and Q4_1 partly for efficiency on Apple Silicon and ARM.

### K-quants: super-blocks

> [!CONCEPT] Super-block (K-quants)
> A group of 256 weights split into sub-blocks (8 of 32 in Q4_K, 16 of 16 in Q6_K). Each sub-block has its own scale (and min), giving fine-grained accuracy, and those sub-block scales are **themselves quantized** to 6 or 8 bits relative to one FP16 super-scale. You get many scales without paying 16 bits for each.

The Q4_K layout, from `ggml-common.h`:

```text
block_q4_K (256 weights, 144 bytes)
  d      2 bytes   FP16 super-scale for the sub-block scales
  dmin   2 bytes   FP16 super-scale for the sub-block mins
  scales 12 bytes  8 sub-block scales + 8 mins, 6 bits each
  qs     128 bytes 256 weights x 4 bits
  => 144 x 8 / 256 = 4.5 bits per weight
```

| Base type | bpw |
|---|---|
| Q2_K | 2.625 |
| Q3_K | 3.4375 |
| Q4_K | 4.5 |
| Q5_K | 5.5 |
| Q6_K | 6.5625 |

### Mixes: what _S, _M and _L mean

`Q4_K_M` is not a single type. It is a **recipe** that assigns a type to every tensor. Not all weights are equally sensitive: errors in some tensors (the final output projection, attention "value" projections, the feed-forward "down" projections, especially in the first and last layers) hurt quality much more than others. So the recipes spend extra bits there.

From llama.cpp's source (`llama-quant.cpp`), the default **Q4_K_M** recipe does roughly this:

- Most matrices: **Q4_K**
- `output.weight` (the final projection onto the vocabulary): **Q6_K**
- `ffn_down` and `attn_v` in "use more bits" layers (the first eighth, the last eighth, and every third layer in between: about half of them): **Q6_K**
- Fused `attn_qkv` tensors: **Q5_K**
- Small 1-D tensors (norms): left in **F32**

`_S` (small) promotes fewer tensors, `_M` (medium) more, `_L` (large) more again. llama.cpp's built-in mixes are Q2_K_S, Q3_K_S/M/L, Q4_K_S/M and Q5_K_S/M; Q6_K is uniform. Names like `Q4_K_L`, `Q6_K_L` or `Q4_K_XL` are **uploader-specific recipes** (bartowski, Unsloth): read the card to see what they changed.

> [!NOTE] Why your Q4_K_M is "5.03 bits"
> Divide the file size by the parameter count: 5,627,044,224 bytes x 8 / 8,953,803,264 parameters = **5.03 bpw**. For Llama 3 8B, llama.cpp lists Q4_K_M at 4.58 GiB, about 4.9 bpw. Your model sits higher because Qwen3.5 has a huge 248,320-token vocabulary: the input embedding and the output projection are each 248,320 x 4,096 = 1.02 billion parameters, together about 23% of the model, and the output one is kept at Q6_K. (The 8.95B figure is the language model inside the GGUF. Qwen's full checkpoint counts 9.65B because it also includes the vision encoder and the multi-token-prediction head.)

### I-quants

The `IQ` types replace evenly spaced levels with carefully designed **non-uniform grids and codebooks**: a weight, or a small group of weights, is stored as an index into a table of good value combinations. That squeezes more accuracy out of each bit, which matters most below 4 bits.

| Type | bpw (llama.cpp) | Notes |
|---|---|---|
| IQ4_NL | 4.5 | Non-linear 4-bit, blocks of 32 |
| IQ4_XS | 4.25 | Non-linear 4-bit in super-blocks; close to Q4_K_S quality, smaller |
| IQ3_S | 3.44 | |
| IQ3_XXS | 3.06 | |
| IQ2_M | 2.7 | |
| IQ2_XXS | 2.06 | |
| IQ1_S | 1.56 | Mostly a curiosity at 9B |

I-quants are almost always made with an importance matrix, and on some backends they decode a little slower than K-quants because unpacking is more complex. Measure before assuming.

### Importance matrices (imatrix)

> [!CONCEPT] Importance matrix (imatrix)
> A statistic gathered by running calibration text through the full-precision model and recording how strongly each input channel of each matrix is used. During quantization, rounding error on heavily used weights is penalised more, so the quantizer picks scales that protect what matters. File size does not change; *which* errors you accept does.

Evidence from llama.cpp's own scoreboard (Llama 3 8B): Q4_K_M's KL divergence (explained below) drops from 0.0313 to 0.0282 with an imatrix, and Q3_K_M's from 0.102 to 0.084. The lower the bit width, the bigger the benefit.

The catch: the calibration text decides what is "important". An imatrix built on English Wikipedia may protect prose better than shell commands or HTTP payloads. Unsloth notes that text-only calibration is not effective for instruct models and uses curated chat data instead.

You can tell whether a GGUF used one: imatrix quants carry metadata keys `quantize.imatrix.file`, `quantize.imatrix.dataset`, `quantize.imatrix.entries_count` and `quantize.imatrix.chunks_count`. Uploader conventions also help: mradermacher labels imatrix repos `i1`, and bartowski publishes the imatrix file itself (`Qwen_Qwen3.5-9B-imatrix.gguf`, about 5 MB).

### Unsloth Dynamic ("UD") quants

Unsloth's Dynamic 2.0 GGUFs use a **custom per-model recipe**: they choose the quant type layer by layer based on their own analysis, with an imatrix from a hand-curated calibration set of over 1.5 million tokens. Files are prefixed `UD-`, for example in `unsloth/Qwen3.5-9B-GGUF`:

| File | Size |
|---|---|
| Qwen3.5-9B-Q4_K_M | 5.68 GB |
| Qwen3.5-9B-UD-Q4_K_XL | 5.97 GB |
| Qwen3.5-9B-UD-Q3_K_XL | 5.05 GB |
| Qwen3.5-9B-UD-IQ2_XXS | 3.19 GB |

Unsloth publishes KL-divergence comparisons for its quants, which is the right way to evaluate them. Remember from [the previous lesson](#/foundations/model-types) that Unsloth's Qwen3.5 GGUFs also ship a chat template with thinking **off** by default.

## Measuring quality loss

### Perplexity

> [!CONCEPT] Perplexity (PPL)
> Run the model over a test text and, at every position, look at the probability it gave the *actual* next token. Perplexity is e raised to the average negative log of those probabilities. Intuition: "how many tokens the model is effectively choosing between". Lower is better. For quant evaluation you care about **delta PPL**: the quant's perplexity minus the original's.

Caveats (from llama.cpp's own docs): perplexity is not comparable across models or tokenizers, fine-tuned chat models score worse than base models even when they are more useful, and errors in opposite directions can cancel out.

### KL divergence

> [!CONCEPT] KL divergence (KLD)
> Compares the quant's **whole** next-token probability distribution with the original model's at every position, not just the probability of the right answer. Zero means identical. It catches changes perplexity can hide, such as the quant shifting probability between two wrong tokens.

llama.cpp's `llama-perplexity` also reports "same top p": how often both models picked the same most likely token.

Here are selected rows from llama.cpp's published scoreboard for **Llama 3 8B** (not your model, but the shape of the curve is general):

| Quant | imatrix | Size (GiB) | delta PPL | KLD |
|---|---|---|---|---|
| Q8_0 | no | 7.96 | +0.003 | 0.0014 |
| Q6_K | no | 6.14 | +0.022 | 0.0055 |
| Q5_K_M | no | 5.33 | +0.057 | 0.0108 |
| Q4_K_M | yes | 4.58 | +0.151 | 0.0282 |
| Q4_K_M | no | 4.58 | +0.175 | 0.0313 |
| IQ4_XS | yes | 4.14 | +0.228 | 0.0363 |
| Q4_0 | no | 4.34 | +0.469 | 0.0719 |
| Q3_K_M | no | 3.74 | +0.657 | 0.1019 |
| IQ3_XXS | yes | 3.05 | +1.227 | 0.1836 |
| Q2_K | yes | 2.96 | +2.416 | 0.3322 |

Notice the knee: going from 8 to 4.5 bits costs little; each step below 4 bits costs a lot more than the last.

To measure it yourself (advanced; needs the BF16 model, so do it on a bigger machine or a cloud box, not your 16 GB Air):

```bash
# 1. Save the reference model's logits (the file is huge: llama.cpp reports 37 GiB for Llama 3 on Wikitext-2)
llama-perplexity -m model-BF16.gguf -f wiki.test.raw --kl-divergence-base base-logits.kld
# 2. Compare a quant against them
llama-perplexity -m model-Q4_K_M.gguf --kl-divergence-base base-logits.kld --kl-divergence
```

### Rules of thumb

These summarise the measurements above plus widespread community experience. Treat the last column as folklore that matches the data, not as law.

| Quant | Measured picture | Practical rule of thumb |
|---|---|---|
| Q8_0 | KLD around 0.001 | Indistinguishable from BF16 in practice |
| Q6_K | KLD around 0.005 | Very close; differences hard to notice in use |
| Q5_K_M | KLD around 0.01 | Close |
| Q4_K_M | KLD around 0.03 | The sweet spot for most people |
| Q3_K_* | KLD around 0.08 to 0.2 | Noticeable: more mistakes, less precise recall |
| Q2 / IQ2 | KLD above 0.3 | Large losses; only to squeeze a huge model in |

Two more patterns worth knowing:

- **Small models suffer more.** A 9B model has less redundancy than a 70B one, so each bit removed hurts relatively more. Below Q4 on a 9B model, degradation is usually obvious.
- **Some tasks break first**: exact recall (precise flag names, payload syntax, API signatures), code, maths, and long reasoning chains where one small error compounds over thousands of thinking tokens. For a security researcher, "it remembered `sqlmap --risk` slightly wrong" is exactly the failure to watch for (see [Trust but verify](#/security/hallucination-verification)).

## File formats

### safetensors: the Hugging Face standard

A safetensors file is an 8-byte header length, a JSON header (each tensor's name, dtype, shape and byte offsets), then raw tensor bytes. **No code, ever**: loading it can only produce numbers. It is memory-mappable, so loading is fast. `Qwen/Qwen3.5-9B` ships its BF16 weights as four shards (`model.safetensors-00001-of-00004.safetensors` and so on) plus `model.safetensors.index.json`, which maps tensor names to shards. You will parse one by hand in [Open and inspect real weights](#/weights/inspect-weights).

### PyTorch `.bin`, `.pt`, `.pth`, `.ckpt`: pickles

Older checkpoints are saved with Python's `pickle`, which is not a data format: it is a small **program** that the unpickler executes to rebuild objects, and it can call any importable function. Here is a harmless demonstration:

```python title="pickle_demo.py"
"""Show that unpickling runs code chosen by whoever made the file."""
import pickle


class Payload:
    def __reduce__(self):
        # Tells pickle: "to rebuild me, call print(...)". Any callable works, os.system included.
        return (print, ("this ran inside pickle.loads(), before you got any data",))


if __name__ == "__main__":
    blob = pickle.dumps(Payload())
    pickle.loads(blob)
```

> [!DANGER] Loading a pickle checkpoint is executing untrusted code
> A malicious `.bin` or `.pt` model can run arbitrary commands the moment you load it. Mitigations, in order of strength:
> - Prefer safetensors or GGUF. Avoid repos that only offer pickles.
> - PyTorch 2.6 made `torch.load(..., weights_only=True)` the default, which uses a restricted unpickler. Do not override it with `weights_only=False` for files you do not trust.
> - Keep PyTorch current: CVE-2025-32434 showed that `weights_only=True` could still be bypassed (remote code execution) in versions up to 2.5.1, fixed in 2.6.0.
> - Hugging Face scans uploads for dangerous pickles, but researchers have repeatedly found ways to evade pickle scanners. A scan is not a guarantee.
> - If you must convert an untrusted pickle, do it in a throwaway container or VM with no credentials and no network.

### GGUF: one file, everything inside

GGUF is llama.cpp's format, used by LM Studio, Ollama, Jan, koboldcpp and others.

```text
GGUF file
  magic "GGUF", version, tensor count, metadata count
  metadata key/values
    general.architecture = qwen35, general.file_type, ...
    qwen35.context_length, qwen35.block_count, ...   (hyperparameters)
    tokenizer.ggml.*                                  (vocabulary, merges, special token IDs)
    tokenizer.chat_template                           (the Jinja template)
    quantize.imatrix.* (if an imatrix was used)
  tensor infos: name, shape, quant type, offset for every tensor
  padding to alignment
  tensor data (memory-mapped at load time)
```

Everything a runtime needs is in one file: weights, hyperparameters, tokenizer and chat template. The vision encoder is a second GGUF, the `mmproj` file (see [the mmproj section](#/foundations/model-types)).

> [!WARNING] Data-only is not the same as safe
> GGUF contains no executable code, but it is parsed by C and C++. In early 2024 Databricks reported several heap overflows in ggml's GGUF parser (patched January 2024), and the chat template inside is a program that runtimes render (see the template CVE in [the previous lesson](#/foundations/model-types)). Keep LM Studio's runtimes updated and prefer reputable uploaders.

### MLX: Apple's format

**MLX** is Apple's open-source array framework for Apple Silicon. MLX models are safetensors files plus a config, with MLX's own group-wise quantization (typically 4-bit with a scale and bias per group of 64 weights, about 4.5 bpw before any mixed-precision layers). The `mlx-community` organisation on Hugging Face publishes conversions, for example `mlx-community/Qwen3.5-9B-4bit` (about 5.95 GB). LM Studio runs MLX models on Apple Silicon through its MLX engine, and MLX is also how you will fine-tune on your Mac ([Fine-tune with MLX](#/finetune/mlx-mac)). Whether MLX or GGUF is faster for a given model on your machine is something to measure, not assume ([Performance tuning](#/lmstudio/performance)).

| Format | Contains code? | Typical use | Quantized? |
|---|---|---|---|
| safetensors | No | Hugging Face originals, training, MLX | Usually BF16; MLX and GPTQ/AWQ variants are quantized |
| `.bin` / `.pt` (pickle) | **Yes, executable** | Legacy PyTorch checkpoints | Usually not |
| GGUF | No (but a template to render) | llama.cpp, LM Studio, Ollama | Yes: Q2 to Q8, plus F16/BF16 |
| MLX | No | Apple Silicon inference and fine-tuning | Yes, group-wise |

## Your quants: which one should you run?

The HauhauCS repo offers four language-model files plus the vision encoder. Exact byte counts from Hugging Face:

| File | Bytes | GB (decimal: Finder, HF) | GiB (binary) | Effective bpw | Card says |
|---|---|---|---|---|---|
| Q4_K_M | 5,627,044,224 | 5.63 | 5.24 | 5.03 | 5.3 GB |
| Q6_K | 7,359,259,008 | 7.36 | 6.85 | 6.58 | 6.9 GB |
| Q8_0 | 9,527,501,184 | 9.53 | 8.87 | 8.51 | 8.9 GB |
| BF16 | 17,920,696,704 | 17.92 | 16.69 | 16.01 | 17 GB |
| mmproj BF16 | 921,704,448 | 0.92 | 0.86 | n/a | 880 MB |

The card's sizes are roughly binary GiB; the Hugging Face file list and macOS Finder show decimal GB. Same files, different units.

My recommendation for your 16 GB Air:

- **Q4_K_M: your default.** It leaves room for long context, a browser, Burp and Docker. The quality cost versus BF16 is real but small.
- **Q6_K: worth it when precision beats speed and context.** Use it for code review, PoC writing, exact command syntax and careful reasoning on short-to-medium contexts (up to about 32K), with heavy apps closed. Expect decode around 25% slower (5.63 / 7.36 = 0.76) and about 1.6 GB less headroom.
- **Q8_0: an experiment, not a daily driver.** It loads (8.87 GiB is under the GPU's default limit of roughly two-thirds of 16 GB), but leaves almost nothing for context, compute buffers and macOS. Useful as a near-lossless reference for short A/B tests.
- **BF16: not on this machine.**

Exact budgets per context length are in [Memory math](#/foundations/memory-math).

## Lab

> [!LAB] Dissect your GGUF
> 1. Find your model file. LM Studio's default models folder is `~/.lmstudio/models`; if this finds nothing, check the models directory shown in LM Studio's My Models tab.
>
> ```bash
> find ~/.lmstudio/models -name "*.gguf" -exec ls -l {} \;
> ```
>
> 2. Store the path in a variable (adjust to what `find` printed):
>
> ```bash
> export MODEL="$HOME/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
> ```
>
> 3. Install llama.cpp's Python GGUF library in your foundations workspace. It provides the `gguf-dump` command (plus `gguf-set-metadata`, `gguf-new-metadata` and others). Check the options with `--help`; current options include `--no-tensors`, `--json`, `--json-array` and `--markdown`.
>
> ```bash
> cd ~/ai-lab/foundations && source .venv/bin/activate
> pip install gguf
> gguf-dump --help
> ```
>
> 4. Dump the metadata only. Long strings are cut to 60 characters in this view.
>
> ```bash
> gguf-dump --no-tensors "$MODEL"
> ```
>
> 5. Print the full chat template (JSON mode keeps full strings) and confirm it is the template from [the previous lesson](#/foundations/model-types):
>
> ```bash
> gguf-dump --json --no-tensors "$MODEL" \
>   | python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"]["tokenizer.chat_template"]["value"])'
> ```
>
> 6. Compare a linear-attention block (0) with a full-attention block (3), and look at the embedding and output tensors:
>
> ```bash
> gguf-dump "$MODEL" | grep -E "blk\.(0|3)\.|token_embd|output\.weight"
> ```
>
> 7. Run the summary script below: `python gguf_summary.py "$MODEL"`.
> 8. In your notebook record: architecture, context length, block count, KV head count, `general.file_type`, whether `quantize.imatrix.*` keys exist, the share of parameters per quant type, and the effective bpw. Check it matches 5,627,044,224 x 8 / 8,953,803,264 = 5.03.

What the metadata dump looks like (abridged; your counts and numbering will differ):

```text
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping ... key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = ...
      3: UINT64     |        1 | GGUF.kv_count = ...
      4: STRING     |        1 | general.architecture = 'qwen35'
    ...
    ...: UINT32     |        1 | qwen35.context_length = 262144
    ...
    ...: STRING     |        1 | tokenizer.chat_template = '{%- set image_count = namespace(value=0) %}\n{%- set video...'
```

Things to notice in step 6: block 0 has linear-attention tensors (names starting `ssm_`, plus a fused `attn_qkv`), while block 3 has classic attention tensors (`attn_q`, `attn_k`, `attn_v`, `attn_output`). That is the hybrid 3:1 layout from [Inside the transformer](#/foundations/transformer), visible in the file. `output.weight` should show `Q6_K`.

```python title="gguf_summary.py"
"""Summarise a GGUF: key metadata, quant types per tensor, effective bits per weight."""
import sys
from collections import defaultdict
from pathlib import Path

from gguf import GGUFReader


def value(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field is not None else "(not present)"


def main(path: str) -> None:
    reader = GGUFReader(path)  # memory-maps the file; does not load it all into RAM
    arch = value(reader, "general.architecture")

    keys = [
        "general.architecture", "general.name", "general.file_type",
        f"{arch}.context_length", f"{arch}.block_count", f"{arch}.embedding_length",
        f"{arch}.attention.head_count", f"{arch}.attention.head_count_kv",
        f"{arch}.full_attention_interval",
        "quantize.imatrix.file", "quantize.imatrix.dataset",
    ]
    for key in keys:
        print(f"{key:40} {value(reader, key)}")
    template = reader.get_field("tokenizer.chat_template")
    if template is not None:
        print(f"{'chat template length (chars)':40} {len(template.contents())}")

    # Group tensors by quantization type.
    count, elems, nbytes = defaultdict(int), defaultdict(int), defaultdict(int)
    for t in reader.tensors:
        kind = t.tensor_type.name
        count[kind] += 1
        elems[kind] += int(t.n_elements)
        nbytes[kind] += int(t.n_bytes)

    total_elems = sum(elems.values())
    print(f"\n{'type':8} {'tensors':>7} {'parameters':>15} {'share':>7} {'bpw':>6}")
    for kind in sorted(elems, key=elems.get, reverse=True):
        bpw = nbytes[kind] * 8 / elems[kind]
        print(f"{kind:8} {count[kind]:7} {elems[kind]:15,} {elems[kind] / total_elems:7.1%} {bpw:6.2f}")

    file_bytes = Path(path).stat().st_size
    print(f"\nparameters in file:          {total_elems:,}")
    print(f"effective bpw (tensors only): {sum(nbytes.values()) * 8 / total_elems:.3f}")
    print(f"effective bpw (whole file):   {file_bytes * 8 / total_elems:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python gguf_summary.py /path/to/model.gguf")
    main(sys.argv[1])
```

Expect `general.file_type` to be 15 (llama.cpp's code for "mostly Q4_K_M"), most parameters in `Q4_K`, a sizeable share in `Q6_K` and `Q5_K`, a few small `F32` tensors, and a whole-file figure around 5.03.

## Recap

- Weights are numbers; their storage format sets file size, and because decode reads every weight per token, it also sets speed (tokens/s ceiling is roughly bandwidth / model size).
- Floating point = sign + exponent (range) + mantissa (precision). BF16 keeps FP32's range with less precision, which is why training uses it and your model's originals are BF16.
- Quantization stores small integers plus per-block scales. Errors come from rounding and are made worse by outliers; small blocks, scale+min, quantized scales (K-quants), non-uniform codebooks (I-quants) and importance matrices all fight that error.
- `Q4_K_M` is a recipe: Q4_K for most tensors, Q6_K/Q5_K for sensitive ones. Your file averages 5.03 bpw because of Qwen3.5's huge vocabulary matrices.
- Judge quants by KL divergence and delta perplexity: Q8 is about lossless, Q6_K very close, Q4_K_M the sweet spot, below Q4 degrades fast, especially on small models.
- safetensors and GGUF are data formats; pickle checkpoints are executable code. Keep parsers updated either way.
- On 16 GB: Q4_K_M by default, Q6_K for precision work at moderate context, Q8_0 only for short experiments.

## Check yourself

1. Why does a Q8_0 model generate tokens roughly half as fast as a Q4_K_M model of the same network on your Mac?
   <details><summary>Answer</summary>

   Decode is memory-bandwidth bound: each token requires reading essentially all weights. Q8_0 is about 9.53 GB versus 5.63 GB for Q4_K_M, so each token moves about 1.7 times more data, and speed drops roughly in proportion.

   </details>
2. A block contains seven weights near 0.05 and one weight of 2.0. Quantized symmetrically to 4 bits with one scale, what happens to the small weights, and what techniques reduce the damage?
   <details><summary>Answer</summary>

   The scale becomes 2.0 / 7, about 0.29, so every small weight rounds to 0: their information is lost. Smaller blocks, per-sub-block scales and mins (K-quants), keeping sensitive tensors at higher precision, non-uniform grids (I-quants) and importance-weighted rounding (imatrix) all reduce this.

   </details>
3. Why is BF16 preferred over FP16 for training even though FP16 has more mantissa bits?
   <details><summary>Answer</summary>

   BF16 has the same 8-bit exponent as FP32, so values do not overflow (FP16 tops out at 65,504) or underflow to zero during training. Networks tolerate the lost precision far better than infinities or zeros, and FP32 to BF16 conversion is a simple truncation with rounding.

   </details>
4. A teammate wants to load a `pytorch_model.bin` from an unknown Hugging Face account to compare it with your model. What do you tell them?
   <details><summary>Answer</summary>

   It is a pickle, so loading it can execute arbitrary code. Prefer a safetensors or GGUF version. If there is none, use a current PyTorch with `weights_only=True` (the default since 2.6; older versions had the CVE-2025-32434 bypass), and do it in a disposable container or VM with no credentials or network.

   </details>
5. Your file is 5,627,044,224 bytes with 8,953,803,264 parameters. Compute its bits per weight and explain why it is above Q4_K's nominal 4.5.
   <details><summary>Answer</summary>

   5,627,044,224 x 8 / 8,953,803,264 = 5.03 bpw. Q4_K_M promotes sensitive tensors (output projection, about half of `ffn_down` and `attn_v`) to Q6_K and fused `attn_qkv` to Q5_K, keeps norms in F32, and Qwen3.5's 248K vocabulary makes the Q6_K output matrix unusually large (about 1 billion parameters).

   </details>

## Go deeper
- [llama.cpp perplexity and KL-divergence README](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity) — how quant quality is measured, with the full Llama 3 scoreboard
- [llama.cpp quantize tool](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize) — every quant type with sizes and options like `--imatrix`
- [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) — the reasoning behind UD quants and why KLD beats perplexity
- [safetensors](https://github.com/huggingface/safetensors) — the format spec and why it exists
- [Databricks: GGML GGUF file format vulnerabilities](https://www.databricks.com/blog/ggml-gguf-file-format-vulnerabilities) — memory-safety bugs found in GGUF parsing
- [MLX](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon
