# Fine-tune on your Mac with MLX

This is where you train your first adapter on your own machine. You will install Apple's MLX tooling, learn what actually fits in 16 GB, run a LoRA fine-tune on a Qwen3.5 model with the dataset you built, watch the loss, test the adapter, and fuse it. You will also learn, honestly, why the 9B itself is not a realistic training target on a fanless Air today.

> [!GOALS]
> - Explain what MLX is and why it suits Apple Silicon
> - Install `mlx-lm` and convert a Qwen3.5 base model to 4-bit
> - Run LoRA/QLoRA with `mlx_lm.lora` using a YAML config, and read its output
> - Plan memory and run time for 16 GB and a fanless chassis
> - Test an adapter with `mlx_lm.generate`, `--test` and `mlx_lm.server`, then fuse it with `mlx_lm.fuse`

## MLX in one paragraph

MLX is Apple's open-source array framework for machine learning on Apple Silicon, roughly "NumPy plus automatic gradients, built for unified memory". Because the CPU and GPU share the same 16 GB, MLX never copies tensors between "host" and "device" memory the way CUDA frameworks do; it builds a lazy computation graph and runs it on the GPU through Metal. **mlx-lm** is the companion package for language models: it loads Hugging Face models, converts and quantizes them, generates text, serves an OpenAI-compatible API, and trains LoRA, DoRA or full fine-tunes. LM Studio's MLX engine is built on the same libraries, which is why MLX models also show up in LM Studio on your Mac.

## The reality check for Qwen3.5 (read this first)

As of mlx-lm 0.32.2 (October 2026), Qwen3.5 is supported: the `qwen3_5` model type loads, generates and can be LoRA-trained. But training it is much slower and hungrier than training a classic transformer, for a specific reason.

Your model's 24 Gated DeltaNet layers process a sequence with a recurrence (see [Inside the transformer](#/foundations/transformer)). For inference, mlx-lm runs that recurrence in a fast fused Metal kernel. That kernel has no backward pass implemented, so in training mode mlx-lm switches to a reference implementation that loops over the sequence one token at a time (`use_kernel=not self.training` in `mlx_lm/models/qwen3_5.py`). Autodiff then has to keep a state for every token of every trained linear-attention layer.

Public numbers from mlx-lm issue #1956 (an M4 Pro with 24 GB, Qwen3.5-**4B** in 4-bit, batch 1, gradient checkpointing, prompt masking, examples of about 600 tokens):

| `--num-layers` | Linear-attention layers trained | Iterations per second | Peak memory |
|---|---|---|---|
| 16 | 12 | 0.073 (about 14 s each) | 14.8 GB |
| 8 | 6 | 0.139 (about 7 s each) | 14.8 GB |
| 5 | 3 | 0.195 (about 5 s each) | 12.6 GB |

Another open issue (#1206) reports LoRA on `mlx-community/Qwen3.5-9B-4bit` failing with a Metal out-of-memory error on the first backward pass even on a 36 GB machine. Community pull requests adding a trainable kernel were proposed and closed without being merged. Check the issue tracker and `pip show mlx-lm` when you read this; it may have improved.

What that means for your Air:

| Model | 4-bit weights | LoRA in MLX on 16 GB | Verdict |
|---|---|---|---|
| Qwen3.5-0.8B | about 0.5 GB | Comfortable | Fastest way to iterate on pipeline and data |
| **Qwen3.5-2B** | about 1.2–1.6 GB | Works with short sequences and few layers | **This lesson's lab model** |
| Qwen3.5-4B | about 2.5 GB | Borderline: the table above peaked at 12.6 GB on a bigger Mac | Try only with `num_layers` 4–5 and 512-token sequences |
| Qwen3.5-9B | about 5.3 GB | Not realistic today | Train it in the cloud: [Fine-tune in the cloud with Unsloth](#/finetune/unsloth-cloud) |
| Qwen3-4B-Instruct-2507 (classic transformer) | about 2.3 GB | Fast, fully kernel-backed | Fallback for fast dataset iteration; different family |

> [!NOTE]
> Why the 2B and not a non-Qwen3.5 model? The 2B shares the 9B's tokenizer, chat template, hybrid architecture and thinking behaviour, so everything you learn about data format, masking and evaluation transfers directly. The cost is speed. If you want quick feedback on whether your *dataset* works, train the same JSONL on `mlx-community/Qwen3-4B-Instruct-2507-4bit` first: it trains much faster, but its adapter only works on that model.

And the HauhauCS question: HauhauCS publishes the uncensored 9B only as GGUF files, and **you cannot train a GGUF**. Training frameworks need the Hugging Face weights (safetensors) to build the model and compute gradients. Your options for an uncensored fine-tune are covered in [Merge, convert to GGUF, load in LM Studio](#/finetune/export-gguf); on the Mac you train on the official Qwen3.5 weights.

## Install

```bash
mkdir -p ~/ai-lab/finetune && cd ~/ai-lab/finetune
python3 -m venv .venv && source .venv/bin/activate
pip install -U "mlx-lm[train]"
```

The `[train]` extra pulls in the training dependencies. Confirm the CLI is there and note the version:

```bash
mlx_lm.lora --help | head -20
pip show mlx-lm | grep Version
```

> [!TIP]
> `uv` works too (`uv venv && source .venv/bin/activate && uv pip install "mlx-lm[train]"`) and is faster, but nothing here requires it. All `mlx_lm.*` commands accept `--help`; flags change between releases, so check there when something in this lesson does not match.

## Get a 4-bit base model

You need the model in MLX format. Two options:

- **Use a community conversion**, such as `mlx-community/Qwen3.5-2B-4bit`. These were produced with `mlx-vlm` and include the vision tower; mlx-lm drops those weights when it loads the model for text.
- **Convert it yourself** from the official repo. You know exactly what you got, and the result is text-only:

```bash
mlx_lm.convert --hf-path Qwen/Qwen3.5-2B -q --q-bits 4 --q-group-size 64 \
  --mlx-path ~/ai-lab/finetune/models/qwen3.5-2b-4bit
```

This downloads the bf16 weights (a few GB) once and writes a 4-bit copy. `-q` turns on quantization, `--q-bits 4` and `--q-group-size 64` give the usual 4-bit layout (see [Precision, quantization and file formats](#/foundations/quantization)).

> [!CONCEPT] QLoRA in MLX
> There is no separate QLoRA switch in mlx-lm. If the model you pass to `mlx_lm.lora` is quantized, the frozen base stays 4-bit and the LoRA matrices train in 16-bit. That is QLoRA.

## Baseline first

Before training, record what the untouched model does on a few test prompts, so you can tell later whether the adapter changed anything.

```bash
mlx_lm.generate --model models/qwen3.5-2b-4bit \
  --chat-template-config '{"enable_thinking": false}' \
  --max-tokens 400 --temp 0.7 --top-p 0.8 --top-k 20 \
  --system-prompt "You are a senior application security engineer helping with authorized testing (bug bounty scope, own labs, CTFs). Be precise, base claims on the evidence given, never invent CVE IDs or results, and say when something needs manual verification." \
  --prompt "Triage this httpx result. Return JSON only. https://jenkins.dev.target.example [200] [Dashboard [Jenkins]] [Jetty(10.0.13)]"
```

`--chat-template-config` passes `enable_thinking: false` into Qwen3.5's chat template, so the model answers directly; the sampling values are Qwen's non-thinking recommendations (see [Sampling and inference settings](#/lmstudio/sampling-settings)). Save the outputs somewhere. You will compare against them, and [Evaluate your fine-tune](#/finetune/evaluate) automates the comparison.

## The training config

`mlx_lm.lora` takes flags, but a YAML config is easier to version and is the only place you can set the LoRA rank and scale. Command-line flags override config values.

```yaml title="lora_config.yaml"
# Base model and data
model: "models/qwen3.5-2b-4bit"
data: "data"                       # folder with train.jsonl, valid.jsonl, test.jsonl
train: true
fine_tune_type: lora               # lora | dora | full
mask_prompt: true                  # loss only on the final assistant message

# Which layers get adapters: the LAST N transformer blocks
num_layers: 8

# LoRA shape (config-file only)
lora_parameters:
  rank: 16
  scale: 2.0                       # multiplier on B·A (PEFT's alpha/r); 2.0 = alpha 32 at r 16
  dropout: 0.05

# Batch and memory
batch_size: 1
grad_accumulation_steps: 4         # optimizer step every 4 iterations (effective batch 4)
max_seq_length: 1024
grad_checkpoint: true

# Schedule
optimizer: adamw
learning_rate: 1e-4
lr_schedule:
  name: cosine_decay
  warmup: 10
  warmup_init: 1e-7
  arguments: [1e-4, 110, 1e-6]     # peak, decay steps, final value
iters: 120                         # micro-batches; see the formula below

# Monitoring and checkpoints
steps_per_report: 10
steps_per_eval: 20
val_batches: 25
save_every: 20
adapter_path: "adapters/qwen35-2b-sec"
seed: 0
```

What each knob means, in the terms from [Training 101](#/finetune/concepts):

| Setting | Meaning | Your starting point |
|---|---|---|
| `num_layers` | Adapters go on the last N blocks (default 16, `-1` for all). Fewer layers = less memory and, for Qwen3.5, much less time | 8 for the 2B (24 blocks); 4–5 if you try the 4B |
| `lora_parameters.rank` | Bottleneck width r | 8–16 for style and format; 32 for a harder task |
| `lora_parameters.scale` | mlx-lm's direct multiplier on the adapter output. Defaults to 20.0 with learning rate 1e-5 | 2.0 with 1e-4 is the same effective step size (2.0 × 1e-4 = 20 × 1e-5) in a scale most guides use |
| `keys` (optional, under `lora_parameters`) | Restrict adapters to named modules, for example `["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]` | Leave unset: all linear layers in the chosen blocks |
| `mask_prompt` | Ignore system and user tokens in the loss | Always on for chat SFT |
| `batch_size`, `grad_accumulation_steps` | Examples per forward pass, and passes per optimizer update | 1 and 4 on 16 GB |
| `max_seq_length` | Longer examples are truncated, with a warning | 1024 for the 2B; 512 for the 4B |
| `grad_checkpoint` | Recompute activations in the backward pass instead of storing them: slower, much less memory | On |
| `iters` | Number of micro-batches (not optimizer updates) | epochs × training examples ÷ batch size |

The `iters` formula matters. With 60 training examples and batch size 1, one epoch is 60 iterations, so `iters: 120` is two epochs and 30 optimizer updates. With 600 examples, two epochs is 1,200 iterations; scale the cosine `arguments` decay steps to match (iters minus warmup).

> [!WARNING]
> The `lora_parameters` block must contain `scale`. mlx-lm deliberately raises an error if it finds `alpha` without `scale`, because unlike PEFT it does not use alpha.

## Memory planning on 16 GB

Add up what has to live in GPU-visible memory during a step:

```text
4-bit base (2B)                            ~1.3 GB
LoRA weights + grads + AdamW (r=16, 8 blocks)   < 0.2 GB
Activations, checkpointed                   grows with sequence length x layers trained
Linear-attention training states (Qwen3.5)  per token x per trained linear-attention layer
macOS, LM Studio, browser                   several GB
```

The last two lines dominate for Qwen3.5, which is why the knobs that matter most are `max_seq_length` and `num_layers`, then `batch_size`.

> [!MAC]
> By default Metal lets the GPU wire about two-thirds of unified memory, around 10–11 GB on your machine. Close LM Studio (or at least unload its model) before training: a loaded 9B Q4_K_M holds about 5.3 GB plus its KV cache. The temporary `sudo sysctl iogpu.wired_limit_mb=<MB>` raise covered in [Performance tuning on Apple Silicon](#/lmstudio/performance) also helps training, with the same caveats (resets at reboot; leave about 4 GB for macOS; too high and the system can freeze). Treat it as a last resort, not step one.

If you hit an out-of-memory error, change one thing at a time in this order: halve `max_seq_length` (and filter long examples out of the data), lower `num_layers`, keep `batch_size` at 1, lower `rank`.

## Thermals and realistic run times

Your Air has no fan. Under a sustained GPU load the chip heats up and macOS lowers its clock speed to stay within temperature limits, so a training run can be noticeably faster in its first minutes than later on. That is normal.

- Run plugged in, on a hard surface, lid open, nothing heavy in the background.
- Keep the Mac awake for the whole run: `caffeinate -i mlx_lm.lora -c lora_config.yaml`
- Check whether macOS is throttling with `pmset -g therm` in a second terminal.

Do not trust anyone's run-time estimate for your exact chip, including mine. Measure: set `iters: 20`, time it, then multiply. As a reference point, the M4 Pro in the table above, a much bigger chip than an Air, needed about 7 seconds per iteration for the 4B at 8 layers with 600-token examples. The 2B is smaller and you can use shorter sequences, so expect seconds per iteration on an Air, not iterations per second. A 120-iteration run on the 2B fits comfortably in an evening; a 1,200-iteration run is an overnight job.

## Run it

> [!LAB] Train a security adapter on Qwen3.5-2B
> 1. Activate the environment and go to the workspace: `cd ~/ai-lab/finetune && source .venv/bin/activate`
> 2. Make sure `data/train.jsonl`, `data/valid.jsonl` and `data/test.jsonl` exist (from [Build a cybersecurity dataset](#/finetune/dataset)). Count training lines with `wc -l data/*.jsonl` and set `iters` to about twice the train count.
> 3. Convert the base model as shown above, and save `lora_config.yaml`.
> 4. Unload models in LM Studio (`lms unload --all`) to free memory.
> 5. Smoke test: `mlx_lm.lora -c lora_config.yaml --iters 20` and note the seconds per iteration and whether memory is stable.
> 6. Full run: `caffeinate -i mlx_lm.lora -c lora_config.yaml`
> 7. Watch the training and validation loss (next section). Stop with <kbd>Ctrl</kbd>+<kbd>C</kbd> if validation loss climbs for several evaluations; checkpoints are saved every 20 iterations.

Your terminal shows a run header (model, layers, rank, learning rate, batch, iterations, max sequence length), then a progress table. The exact layout changes between versions; the information looks roughly like this:

```text
  iter    train loss    tok/s    trained
    10    2.412 v         180      5.9k
    20    val 2.105    3.41s
    20    1.874 v         176     11.8k
    20    save 0000020_adapters.safetensors
   ...
   120    val 1.322    3.38s
   120    1.047 v         171     70.4k
```

Afterwards `adapters/qwen35-2b-sec/` contains `adapters.safetensors` (the final adapter), `adapter_config.json` (every setting used, which you need for testing and fusing), and numbered checkpoints such as `0000060_adapters.safetensors`.

## Reading the loss

| What you see | What it means | What to do |
|---|---|---|
| Training and validation loss both fall, then flatten | Healthy learning | Done; use the final adapter |
| Training loss keeps falling, validation turns upward | Overfitting | Use the checkpoint with the lowest validation loss; fewer iterations, lower rank or learning rate |
| Loss barely moves | Learning rate too low, too few layers, or data the model already predicts well | Raise the learning rate a little or train more layers; check your data is really different from what the model does already |
| Loss jumps up or shows `nan` | Learning rate too high (divergence) | Lower it (try 5e-5), keep warmup |
| Training loss near 0 within a few dozen iterations | Memorising a tiny or repetitive dataset | More and more varied data; fewer iterations |

To use a specific checkpoint, copy it over the final file name, keeping `adapter_config.json`:

```bash
cp adapters/qwen35-2b-sec/0000060_adapters.safetensors adapters/qwen35-2b-sec/adapters.safetensors
```

## Test the adapter

**Test-set loss and perplexity**, computed on `test.jsonl`, for the adapter and then for the base model (an empty adapter path means "no adapter"):

```bash
mlx_lm.lora --model models/qwen3.5-2b-4bit --data data --test --batch-size 1 \
  --mask-prompt --adapter-path adapters/qwen35-2b-sec
mlx_lm.lora --model models/qwen3.5-2b-4bit --data data --test --batch-size 1 \
  --mask-prompt --adapter-path ""
```

```text
Test loss 1.093, Test ppl 2.983.
```

Lower test loss than the base is necessary but not sufficient: it says the model finds your reference answers more predictable, not that its own answers are better. Generation tests decide that.

**Generate with the adapter** using exactly the system prompt from training:

```bash
mlx_lm.generate --model models/qwen3.5-2b-4bit --adapter-path adapters/qwen35-2b-sec \
  --chat-template-config '{"enable_thinking": false}' \
  --max-tokens 400 --temp 0.7 --top-p 0.8 --top-k 20 \
  --system-prompt "You are a senior application security engineer helping with authorized testing (bug bounty scope, own labs, CTFs). Be precise, base claims on the evidence given, never invent CVE IDs or results, and say when something needs manual verification." \
  --prompt "Triage this httpx result. Return JSON only. https://jenkins.dev.target.example [200] [Dashboard [Jenkins]] [Jetty(10.0.13)]"
```

Compare with your baseline. You are looking for the trained habits: the exact JSON keys, the scope check, the refusal to invent CVEs.

**Serve it over an OpenAI-compatible API**, so your scripts and [eval.py](#/finetune/evaluate) can talk to it the same way they talk to LM Studio:

```bash
mlx_lm.server --model models/qwen3.5-2b-4bit --adapter-path adapters/qwen35-2b-sec \
  --port 8080 --chat-template-args '{"enable_thinking": false}'
```

The base URL is then `http://localhost:8080/v1`. The server binds to 127.0.0.1 by default; keep it that way.

## Fuse the adapter into the model

Serving with `--adapter-path` works, but a single self-contained model folder is easier to move around.

```bash
mlx_lm.fuse --model models/qwen3.5-2b-4bit --adapter-path adapters/qwen35-2b-sec \
  --save-path fused/qwen35-2b-sec
```

> [!CONCEPT] Fusing
> Computing `W + scale · B·A` for every adapted matrix and saving the result as ordinary weights. The output runs at the base model's speed with no adapter logic. By default mlx-lm re-quantizes fused weights to the base's 4-bit format; `--dequantize` saves a 16-bit model instead (bigger, and without a second rounding of your adapter's changes).

What you can do with the fused folder:

- **Run it with `mlx_lm.generate` or `mlx_lm.server`** by passing `--model fused/qwen35-2b-sec`.
- **Load it in LM Studio's MLX engine.** Create `~/.lmstudio/models/<your-name>/qwen35-2b-sec/` and copy the folder's contents there, then look for it under My Models. The fused model is text-only (mlx-lm drops the vision tower). If LM Studio refuses to load it, `mlx_lm.server` is the reliable fallback.
- **Not** turn it into a GGUF. `mlx_lm.fuse --export-gguf` only supports Llama, Mistral and Mixtral-style models. And for Qwen3.5, mlx-lm stores some weights in its own layout (it transposes the linear-attention convolution weights and folds an offset into the norm weights), so pointing llama.cpp's converter at an MLX-saved Qwen3.5 folder will produce a broken model. The GGUF path starts from Hugging Face weights; see [Merge, convert to GGUF, load in LM Studio](#/finetune/export-gguf).

## Where this leaves the 9B

The workflow you just ran is exactly the workflow for the 9B: same JSONL, same chat template, same masking, same evaluation. Only the hardware differs. Your realistic plan:

1. Iterate on the dataset locally with the 2B (or with Qwen3-4B-Instruct-2507 for speed) until the adapter clearly improves your evals.
2. Train the 9B with Unsloth on a rented GPU using the same files.
3. Export to GGUF and run it in LM Studio next to the model you use today.

If a future mlx-lm release adds a trainable kernel for Gated DeltaNet, a 9B QLoRA with few layers and short sequences may become feasible on 16 GB. Check the mlx-lm releases page before assuming either way.

## Recap

- MLX is Apple's array framework for unified memory; `mlx-lm` converts, runs, serves and fine-tunes LLMs. Install with `pip install "mlx-lm[train]"`.
- Qwen3.5 trains in mlx-lm, but its linear-attention layers use a slow token-by-token path in training. On 16 GB, the 2B is the practical model; the 4B is borderline; the 9B belongs in the cloud. HauhauCS ships GGUF only, which cannot be trained.
- A quantized base passed to `mlx_lm.lora` gives you QLoRA. Put rank and scale in the YAML `lora_parameters`; `scale` is a direct multiplier (default 20 with learning rate 1e-5).
- Memory levers in order: `max_seq_length`, `num_layers`, `batch_size` 1 with gradient accumulation, `grad_checkpoint`. `iters` counts micro-batches.
- Watch training and validation loss; pick the checkpoint with the lowest validation loss. Test with `--test`, `mlx_lm.generate --adapter-path`, and `mlx_lm.server`.
- `mlx_lm.fuse` bakes the adapter in; the Qwen3.5 result is an MLX-only, text-only model, not a GGUF source.

## Check yourself

1. You have 400 training examples, `batch_size: 1`, `grad_accumulation_steps: 4`, and want 2 epochs. What should `iters` be, and how many optimizer updates happen?
   <details><summary>Answer</summary>

   `iters` counts micro-batches: 2 × 400 ÷ 1 = 800 iterations. With an update every 4 iterations, that is 200 optimizer updates.

   </details>
2. Why does lowering `num_layers` speed up Qwen3.5 training in mlx-lm so much more than it does for a classic transformer?
   <details><summary>Answer</summary>

   Every trained block above the lowest adapter needs a backward pass, and for Qwen3.5's linear-attention blocks that backward pass runs a token-by-token reference loop instead of a fused kernel. Fewer trained blocks means fewer of those slow loops.

   </details>
3. Your config says `alpha: 32` under `lora_parameters` and the run fails immediately. Why?
   <details><summary>Answer</summary>

   mlx-lm's LoRA config requires `scale` (a direct multiplier) and raises an error when it sees `alpha` without `scale`. Use `scale: 2.0` for the equivalent of alpha 32 at rank 16.

   </details>
4. How do you measure the base model's test loss with the same command you use for the adapter?
   <details><summary>Answer</summary>

   Run `mlx_lm.lora --model ... --data data --test --adapter-path ""`. An empty adapter path skips loading adapters.

   </details>
5. Why can't you take `fused/qwen35-2b-sec` and run llama.cpp's `convert_hf_to_gguf.py` on it?
   <details><summary>Answer</summary>

   mlx-lm saves Qwen3.5 in its own internal layout: renamed keys, transposed linear-attention convolution weights, norm weights with an offset folded in, and no vision or MTP weights. The converter expects the original Hugging Face layout, so the result would be wrong. Start the GGUF path from Hugging Face or PEFT weights instead.

   </details>

## Go deeper
- [mlx-lm LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) — the official reference for data formats, flags, fusing and memory tips
- [mlx-lm repository](https://github.com/ml-explore/mlx-lm) — check releases and the issues tracker for Qwen3.5 training updates
- [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html) — the array framework underneath, including unified memory and lazy evaluation
- [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) — a community package that adds DPO, ORPO, GRPO and other trainers on top of MLX
