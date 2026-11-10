# Your machine and your model

Before learning how LLMs work, it pays to understand the two things you'll be working with every day: a 16 GB Apple Silicon MacBook Air and a 9-billion-parameter model squeezed into a 5.3 GB file. This lesson explains what's special about your hardware, what every part of your model's long name means, where your 16 GB actually goes, and what you can honestly expect from this setup.

> [!GOALS]
> - Explain unified memory and why memory bandwidth sets your token speed
> - Decode every part of the name `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` Q4_K_M GGUF
> - Budget your 16 GB between macOS, your apps, the model and its working memory
> - Set realistic expectations for a 9B model on a fanless laptop
> - Measure your own chip, memory pressure, model files and tokens per second

## Unified memory, from zero

### How a typical PC does it

On a gaming PC or a deep-learning workstation, there are two separate pools of memory:

- **System RAM**, attached to the CPU (say 32 GB of DDR5).
- **VRAM** (video RAM), soldered onto the graphics card next to the GPU (say 8, 12 or 24 GB).

The GPU is what makes LLMs fast, but it can only work quickly on data that sits in its *own* VRAM. Anything else has to be copied over the PCIe bus, which is far slower than VRAM. So on a PC, the question "can I run this model?" really means "does the model fit in VRAM?". A 12 GB card can't comfortably hold a 17 GB model, even if the PC has 64 GB of system RAM. You either shrink the model, split it (some layers on the GPU, the rest on the much slower CPU), or buy a bigger card.

### How your Mac does it

Apple Silicon does away with the split. The CPU, GPU and Neural Engine sit on the same chip and all share **one pool of memory** in the same package. Your 16 GB *is* your VRAM, and also your RAM.

> [!CONCEPT] Unified memory
> A design where the CPU and GPU read and write the same physical memory instead of each having its own. Nothing has to be copied between "system RAM" and "VRAM", because they're the same thing. The upside is that the GPU can use most of your 16 GB. The catch is that macOS, Safari, Burp, Docker and the model are all competing for that same 16 GB.

```text
Typical PC                                  Your MacBook Air
+-----------+   PCIe (slow)   +---------+   +-------------------------------+
| CPU       | <-------------> | GPU     |   | CPU | GPU | Neural Engine      |
| 32 GB RAM |                 | 12 GB   |   |-------------------------------|
+-----------+                 | VRAM    |   |   16 GB unified memory        |
                              +---------+   |   (shared by everything)      |
                                            +-------------------------------+
```

### The GPU's share of the pool

Unified doesn't mean the GPU can grab all 16 GB. By default, macOS lets the GPU *wire* (pin in place for its own use) roughly two-thirds of memory on a 16 GB machine, which is about **10.7 GB**. The rest stays available for macOS and your apps, so the system doesn't fall over. When LM Studio estimates whether a model "fits on the GPU", that ~10.7 GB is the ceiling it's working against, not 16.

> [!MAC]
> Advanced and optional: the GPU limit can be raised until the next reboot with `sudo sysctl iogpu.wired_limit_mb=<MB>`. For example, `12288` allows about 12 GB. It resets when you restart. Don't go anywhere near 16384: macOS needs roughly 4 GB to itself, and if you starve it you'll get heavy swapping, frozen apps or a kernel panic. You don't need this tweak for anything in Part 1. The [performance lesson](#/lmstudio/performance) covers when it's worth it.

## Why Apple Silicon is good at this: memory bandwidth

Here's the most important performance fact in this course, explained properly in [How text gets generated](#/foundations/generation): **to produce each new token, the model has to read essentially all of its weights from memory.** Every single token. For your 5.3 GB model, that's about 5.3 GB of memory reads per word-fragment of output.

So the speed limit on generation usually isn't how fast the GPU can do arithmetic. It's how fast memory can deliver the weights.

> [!CONCEPT] Memory bandwidth
> How many bytes per second can move between memory and the processor, measured in GB/s. Think of it as the width of the pipe the weights flow through. For LLM text generation it's usually *the* bottleneck, so a rough rule is: **tokens per second ≈ bandwidth ÷ model size.**

Apple Silicon has unusually high bandwidth for a laptop, because the memory sits right next to the chip on a wide bus. Here's what each MacBook Air generation offers (Apple's published figures for the base chips):

| Chip | Memory bandwidth | Rough ceiling for a 5.6 GB model |
|---|---|---|
| M1 | ~68 GB/s | ~12 tok/s |
| M2 | 100 GB/s | ~18 tok/s |
| M3 | 100 GB/s | ~18 tok/s |
| M4 | 120 GB/s | ~21 tok/s |
| M5 | 153 GB/s | ~27 tok/s |

The "ceiling" column is just bandwidth divided by 5.63 GB, the size of your Q4_K_M file in decimal gigabytes. Real speeds come in lower, typically 70–85% of the ceiling, because no system moves memory perfectly. You'll measure your own in the lab. For comparison, ordinary laptop DDR5 serving a CPU is often well under 100 GB/s, while a high-end desktop GPU like the RTX 4090 reaches about 1,000 GB/s but has only 24 GB of VRAM and costs more than your laptop. Your Mac sits in a useful middle ground: decent bandwidth *and* a large pool the GPU can actually use.

> [!NOTE]
> Bandwidth decides how fast tokens come *out* (generation). How fast your prompt is *read in* depends mostly on GPU compute instead, which is why pasting a huge HTTP response gives a noticeable pause before the first word appears. [How text gets generated](#/foundations/generation) explains both phases.

## What "9B" means

**9B means about 9 billion parameters.** A parameter is one learned number inside the model. You'll see exactly what these numbers do in [What is an LLM, really?](#/foundations/what-is-an-llm). For now, think of the model as a huge mathematical function with 9 billion adjustable knobs, all set during training.

> [!CONCEPT] Parameter (weight)
> A single number the model learned during training. A model is mostly a big collection of these, arranged in grids called matrices. "9B" counts them: roughly 9,000,000,000 numbers. More parameters generally means more capacity to store knowledge and patterns, and more memory and time per token.

Parameter count translates directly into file size, depending on how many bits store each number:

```text
Language-model parameters:   ~8.95 billion  (you'll count them yourself in "Inside the transformer")

BF16 (16 bits = 2 bytes each):  8.95e9 x 2 bytes  = 17.9 GB  (16.7 GiB)  -> the "17 GB" BF16 file
Q4_K_M (about 5 bits each):     8.95e9 x 5/8 byte =  5.6 GB  ( 5.2 GiB)  -> the "5.3 GB" file you run
```

The original BF16 weights are larger than your entire RAM, so you can't run them as they are. Your Q4_K_M file stores each weight in about 5 bits instead of 16, roughly a 3x reduction, at a small cost in quality. That process is called **quantization**, and it gets a full lesson in [Precision, quantization and file formats](#/foundations/quantization).

> [!NOTE]
> GB vs GiB will trip you up constantly. 1 GB = 1,000,000,000 bytes (what Hugging Face's file list and Apple's bandwidth figures use). 1 GiB = 1,073,741,824 bytes (what `ls -lh` and many tools show as "G"). Your Q4_K_M file is exactly 5,627,044,224 bytes: that's 5.63 GB, or 5.24 GiB, and the model card's table rounds it to "5.3 GB". Same file, different units. When you divide bandwidth (GB/s) by model size, use decimal GB for both.

## Decoding the model name

Your model's full name packs in a lot of information. Here's each piece, with a pointer to the lesson that covers it in depth.

| Part of the name | What it means | Deep dive |
|---|---|---|
| **Qwen3.5** | The model family, made by Alibaba's Qwen team. Qwen3.5 was released on 2 March 2026 under the Apache-2.0 license (free for commercial use). The small series comes in 0.8B, 2B, 4B and 9B sizes. | [Inside the transformer](#/foundations/transformer) |
| **9B** | About 9 billion parameters, the largest of the small Qwen3.5 models. | [What is an LLM, really?](#/foundations/what-is-an-llm) |
| **Uncensored** | A community modification that removes the model's trained tendency to refuse requests. Not made or endorsed by Qwen. | [Alignment, refusals and "uncensored"](#/foundations/alignment-uncensored) |
| **HauhauCS** | The Hugging Face user who made and published the modification. The card doesn't say exactly which uncensoring method was used. | [Direction ablation](#/weights/direction-ablation) |
| **Aggressive** | The more thorough refusal-removal variant. The author describes it as stronger uncensoring and mentions a possible milder "Balanced" variant. | [Alignment, refusals and "uncensored"](#/foundations/alignment-uncensored) |
| **GGUF** | The file format used by llama.cpp, the engine LM Studio uses for these files. One file holds the weights plus metadata: architecture, tokenizer, chat template. | [Precision, quantization and file formats](#/foundations/quantization) |
| **Q4_K_M** | The quantization type: mostly 4-bit "K-quants" in a "Medium" mix that keeps some sensitive tensors at higher precision, averaging about 5 bits per weight. | [Precision, quantization and file formats](#/foundations/quantization) |
| **mmproj** | An optional second file (880 MB, BF16) holding the **vision encoder**, which turns images and video frames into something the language model can read. You only need it for image or video input. | [Inside the transformer](#/foundations/transformer) |

The repository also publishes the same model at other precisions: Q6_K (6.9 GB), Q8_0 (8.9 GB) and BF16 (17 GB). Higher precision means slightly better quality, a bigger file and slower generation, for the bandwidth reason above.

> [!WARNING]
> "Uncensored" means the refusals are gone, not that the model got smarter or more truthful. The author reports 0 refusals out of 465 test prompts and "no capability loss", but that's their own test and hasn't been independently checked. In practice an uncensored model has no safety net: it will confidently produce wrong commands, invented CVE numbers and plausible-looking nonsense as readily as correct answers, and responsibility for what you do with the output sits entirely with you. One known quirk: it may still add a short disclaimer at the end of an answer. That habit comes from the base model, and the full answer is still there.

## Budgeting your 16 GB

Here's where the memory goes on a typical working day, with the model loaded at a 32K-token context. A **context** is how much text the model can see at once, measured in tokens. Tokens are covered in [Tokens and tokenization](#/foundations/tokens) and the full context math in [Memory math](#/foundations/memory-math).

| Consumer | Approximate size | Notes |
|---|---|---|
| macOS and background services | 3–4 GB | WindowServer, Spotlight, system daemons. Varies. |
| Your apps | 2–5 GB | A browser with many tabs, Burp (Java, can be 1–2 GB alone), VS Code, Docker Desktop's VM |
| Model weights (Q4_K_M) | 5.24 GiB (5.63 GB) | The GGUF file, loaded into memory |
| KV cache at 32K context | 1 GiB | 32 KiB per token × 32,768 tokens. Grows with context length. |
| Linear-attention state | ~0.05 GiB | Fixed size, doesn't grow with context |
| Runtime compute buffers | ~0.5–1 GiB | Scratch space for the math. Varies with settings. |
| Vision encoder (only if loaded) | 0.9 GB | Only when you use images |
| **Total** | **~12–17 GB** | Tight. Close heavy apps when you work with the model. |

The **KV cache** is the model's working memory for the conversation so far. It's explained in [How text gets generated](#/foundations/generation). Your model is unusually frugal here: thanks to its hybrid design, only 8 of its 32 layers keep a growing cache, so each token of context costs 32 KiB. A classic design of similar size, Qwen3-8B, needs 144 KiB per token, 4.5 times as much. [Inside the transformer](#/foundations/transformer) explains why.

Two takeaways:

1. **The model side (~7 GB at 32K context) fits comfortably under the ~10.7 GB GPU limit.** Push the context to 128K and the KV cache alone becomes 4 GiB, putting the model side at about 10 GiB, right at the edge.
2. **The squeeze usually comes from your other apps, not the model.** If macOS starts swapping (moving memory pages to the SSD), everything slows down. The lab shows you how to see this happening.

> [!MAC]
> The Qwen model card suggests keeping at least 128K of context to preserve thinking quality on long tasks. On 16 GB that's a real trade-off against the memory your other apps need. The [Recommended configs for your Mac](#/lmstudio/presets-for-you) lesson gives concrete settings for different kinds of work.

## Honest expectations

A 9B model in 2026 is genuinely useful, but it's not a frontier model, and pretending otherwise leads to disappointment and bad security decisions. Frontier cloud models are far larger (their sizes are often not disclosed) and run on data-center hardware.

| A local 9B model is good at | Frontier models still clearly win at |
|---|---|
| Explaining code, configs, HTTP requests and responses, stack traces | Long multi-step reasoning on novel problems (subtle logic bugs, complex chains) |
| Summarizing and extracting from logs, recon output and docs | Long, open-ended agent tasks without heavy scaffolding |
| Drafting and rewriting report sections from your notes | Precise recall of niche facts: exact CVE ids, rare tool flags, version details |
| Triage and classification at volume ("is this response interesting?") | Holding and reasoning over very large codebases at once |
| Producing JSON that follows a schema, with a good prompt | Knowledge of recent events (its training data ends before its March 2026 release) |
| Working on data you're not allowed to send to a cloud provider | Hard math and formal reasoning |

Speed is in the same honest range: somewhere around 12–22 tokens per second on an Air, depending on the chip. That's comfortable reading speed for chat, but slow for bulk jobs, which is why later lessons teach you to keep prompts and outputs tight.

The theme of this whole course is **closing that gap with engineering**: retrieval to supply facts the model doesn't know, tools to replace its guesses with real data, structured output to make it parseable, and verification to catch hallucinations. A well-scaffolded 9B model beats a lazily used big one on a lot of real tasks.

## The fanless reality

Your MacBook Air has no fan. It cools passively, through the chassis. That's fine for bursts of work, but generating text keeps the GPU and memory system busy continuously. After a few minutes of sustained load the chip heats up, and macOS lowers clock speeds to stay within safe temperatures. This is called **thermal throttling**.

In practice:

- A short chat runs at full speed.
- A long generation, a batch job or an agent loop running for 10+ minutes may slow down noticeably, depending on the room, the surface and the chip.
- Benchmarks taken in the first minute are more optimistic than what you'll see in sustained work.

> [!MAC]
> To keep sustained performance as high as possible: work on a hard surface (not a blanket or your lap), stay plugged in for long jobs, and turn off Low Power Mode when benchmarking (System Settings → Battery). To watch the thermal state directly, run:
>
> ```bash
> sudo powermetrics --samplers thermal -i 5000 -n 1
> ```
>
> Look for the "Current pressure level" line. Nominal is good; anything higher means macOS is starting to manage heat. Remove `-n 1` to sample every 5 seconds until you press <kbd>Ctrl</kbd>+<kbd>C</kbd>.

## Lab: measure your setup

> [!LAB] Identify your chip, memory, model files and real speed
> **Part 1: chip and memory**
>
> 1. Print the chip name:
>
>    ```bash
>    sysctl -n machdep.cpu.brand_string
>    ```
>
>    ```text
>    Apple M2
>    ```
>
> 2. Get the full hardware summary:
>
>    ```bash
>    system_profiler SPHardwareDataType
>    ```
>
>    Find the `Chip`, `Total Number of Cores` and `Memory` lines.
> 3. Get the exact memory size in bytes, then convert it:
>
>    ```bash
>    sysctl hw.memsize
>    echo "$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GiB"
>    ```
>
>    ```text
>    hw.memsize: 17179869184
>    16 GiB
>    ```
>
> 4. Check your GPU core count (it affects prompt-processing speed):
>
>    ```bash
>    system_profiler SPDisplaysDataType | grep -i "cores"
>    ```
>
> 5. Look up your chip's bandwidth in the table above and work out your ceiling: bandwidth ÷ 5.63. Write the chip, cores, memory, bandwidth and ceiling in `~/ai-lab/NOTES.md`.
>
> **Part 2: memory pressure**
>
> 6. Eject the model in LM Studio if it's loaded. Open Activity Monitor (`open -a "Activity Monitor"`) and go to the **Memory** tab. At the bottom, note **Memory Used**, **Cached Files**, **Swap Used** and the colour of the **Memory Pressure** graph.
> 7. Load the model in LM Studio and send it a message. Watch what changes: Memory Used should jump by roughly the model size plus its buffers, and LM Studio's processes will appear near the top of the list when sorted by Memory.
> 8. Read the pressure from the terminal too:
>
>    ```bash
>    memory_pressure | tail -1
>    sysctl vm.swapusage
>    ```
>
>    Green pressure and little or no swap growth mean you're fine. Yellow means memory is getting tight. Red, or swap climbing while you generate, means close something (browser tabs and Docker first). Record before and after numbers.
>
> **Part 3: find the model files**
>
> 9. LM Studio stores models under `~/.lmstudio/models/<publisher>/<model>/` by default. List the GGUF files and their sizes:
>
>    ```bash
>    find ~/.lmstudio/models -name "*.gguf" -exec ls -lh {} \;
>    ```
>
>    You should see something like `.../HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive.../Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` at about `5.2G` (`ls -lh` reports GiB). If nothing shows up, open LM Studio's **My Models** tab, which shows the models directory in use (you can change it there). Very old installs used a different folder.
> 10. Cross-check with the CLI, which also prints size and architecture:
>
>     ```bash
>     lms ls
>     ```
>
> **Part 4: measure tokens per second**
>
> 11. Start a new chat in LM Studio and send: `Explain, in about 300 words, how a browser validates a TLS certificate chain.`
> 12. When it finishes, find the generation stats LM Studio shows with the reply: tokens per second, number of tokens, time to first token and stop reason. Depending on your version, they're shown under the message or behind a small info icon. Record them.
> 13. Repeat twice more in fresh chats and average the three tokens-per-second values. Divide by your ceiling from step 5. That's your efficiency; somewhere between 0.6 and 0.85 is normal.
> 14. Optional thermal test: send five long requests back to back (`Explain, in about 800 words...`) and compare the first and last tokens-per-second figures.

> [!TIP]
> If the reply starts with a `<think>...</think>` block, thinking mode is on. Thinking tokens count toward tokens per second just like answer tokens, so the speed figure is still valid. They just make you wait longer for the actual answer.

## Recap

- Unified memory means the CPU and GPU share your 16 GB. By default macOS lets the GPU use about two-thirds of it (~10.7 GB).
- Generating each token means reading nearly all the weights, so **memory bandwidth sets your speed**: tok/s ≈ bandwidth ÷ model size (100–153 GB/s on recent Airs, so roughly 12–22 tok/s in practice for a 5.6 GB model).
- "9B" means about 9 billion learned numbers. In BF16 that's a 17 GB file, too big for your RAM. Q4_K_M stores about 5 bits per weight, giving 5.3 GiB.
- The name decodes as: Qwen3.5 family, 9B size, uncensored by HauhauCS (Aggressive variant), GGUF file format, Q4_K_M quantization. mmproj is the optional vision encoder.
- At 32K context the model side needs about 7 GB. The squeeze usually comes from your other apps.
- A 9B model is strong at explaining, summarizing, extracting and drafting, and weaker at long novel reasoning, niche facts and recent knowledge. Engineering closes much of the gap.
- The fanless Air throttles under sustained load. Benchmark with that in mind.

## Check yourself

1. On a PC with 64 GB of RAM and a 12 GB GPU, why can't you comfortably run a 17 GB model on the GPU, and why is your Mac's situation different?
   <details><summary>Answer</summary>

   On the PC, the GPU works fast only on data in its own 12 GB of VRAM. The 64 GB of system RAM sits across a slow PCIe link, so the model would have to be split, with part of it running slowly on the CPU. On your Mac, the CPU and GPU share one unified pool, so the GPU can directly use most of the 16 GB (about 10.7 GB by default) without copying anything.

   </details>
2. Your friend has an M4 Air (120 GB/s) and you have an M2 Air (100 GB/s). Both run the same Q4_K_M file. Roughly what generation speeds should each of you expect, and why?
   <details><summary>Answer</summary>

   Generation is memory-bandwidth bound: each token reads about 5.63 GB of weights. Ceilings: M2 ≈ 100 / 5.63 ≈ 18 tok/s, M4 ≈ 120 / 5.63 ≈ 21 tok/s. At a typical 70–85% efficiency that's roughly 12–15 tok/s on the M2 and 15–18 tok/s on the M4. The speed ratio follows the bandwidth ratio (about 1.2x).

   </details>
3. Why does the same model file show up as 5.3 GB in one place and 5.63 GB in another?
   <details><summary>Answer</summary>

   Different units. 5.63 GB is decimal gigabytes (10^9 bytes). The same number of bytes is about 5.24 GiB (2^30 bytes), which the model card rounds to 5.3. Neither number is wrong.

   </details>
4. You load the model at 32K context and everything is fine, but with Burp, Docker and 40 browser tabs open, generation suddenly crawls. What's the likely cause, and how would you confirm it?
   <details><summary>Answer</summary>

   Memory pressure: the total demand from macOS, your apps and the model exceeds 16 GB, so macOS compresses memory and swaps to the SSD. Confirm it in Activity Monitor's Memory tab (pressure graph yellow or red, Swap Used growing) or with `memory_pressure` and `sysctl vm.swapusage`. Close heavy apps rather than shrinking the model first.

   </details>
5. Name two things a local 9B model is genuinely good at for security work, and one thing you should always verify.
   <details><summary>Answer</summary>

   Good at: explaining code and HTTP traffic, summarizing logs and recon output, drafting report text, triaging large volumes of responses, structured extraction, and working on confidential data that can't leave your laptop. Always verify: niche facts such as CVE ids, exact tool flags, version numbers and payload behaviour, which a 9B model (especially an uncensored one with no safety net) will happily invent.

   </details>

## Go deeper

- [Apple MacBook Air technical specifications](https://www.apple.com/macbook-air/specs/) — official memory and bandwidth figures for the current Air.
- [Performance of llama.cpp on Apple Silicon M-series](https://github.com/ggml-org/llama.cpp/discussions/4167) — community benchmarks for every M-series chip, showing that generation speed tracks memory bandwidth.
- [View memory usage in Activity Monitor on Mac](https://support.apple.com/guide/activity-monitor/view-memory-usage-actmntr1004/mac) — Apple's explanation of memory pressure, wired memory, compression and swap.
- [LM Studio documentation](https://lmstudio.ai/docs) — including the models directory layout and the `lms` CLI.
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) — official architecture summary, context length and recommended settings for the base model.
