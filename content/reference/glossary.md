# Glossary

Every term this lab defines, in one place, in plain language. Each entry links to the lesson that explains it properly. Use search (<kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>K</kbd>) to jump straight to a term.

> [!GOALS]
> - Look up any term from the course in seconds
> - Jump from a short definition to the lesson that explains it in depth
> - Use this page as a self-test: cover the definitions and explain each term aloud

## A–C

**Abliteration**: removing the "refusal direction" from a model's weights so it stops refusing. Found as a difference of means between activations on harmful and harmless prompts, then projected out of every matrix that writes to the residual stream. [Alignment](#/foundations/alignment-uncensored), [Direction ablation](#/weights/direction-ablation)

**Activation**: the vector of numbers a layer produces for a token during a forward pass. Unlike weights, activations exist only while the model is running. [Interpretability 101](#/weights/interpretability-101)

**Activation patching (causal tracing)**: replacing an activation from one run with the activation from another run to find which component causes a behaviour. [Interpretability 101](#/weights/interpretability-101)

**Activation steering**: adding a vector to the residual stream during generation to push behaviour in a direction (more concise, more formal) without changing any weights. [Steering](#/weights/steering)

**Adam / AdamW**: the standard optimizer. It keeps running averages of each weight's gradients and squared gradients, which costs 8 extra bytes per parameter. [Training 101](#/finetune/concepts)

**Agent**: an LLM in a loop with tools, memory and a goal. It decides an action, your code executes it, the result goes back in, and it repeats until a stop condition. [What is an agent?](#/agents/what-is-an-agent)

**Alignment**: the post-training that turns a raw text predictor into an assistant with preferred behaviour, including what it refuses. [Alignment](#/foundations/alignment-uncensored)

**Approval gate**: a point where an agent must show the exact command it wants to run and wait for a human "yes" before doing anything noisy or irreversible. [Agent loop](#/bugbounty/agent-loop)

**Attention (Q, K, V)**: the mechanism that lets each token look back at earlier tokens. Each position's *query* is compared with earlier *keys* to produce weights over their *values*. [Transformer](#/foundations/transformer)

**Autoregressive generation**: producing text one token at a time, feeding each new token back in as input for the next. [Generation](#/foundations/generation)

**Base model**: a model trained only on next-token prediction. It continues documents rather than answering questions. [Model types](#/foundations/model-types)

**BF16 (bfloat16)**: a 16-bit floating-point format with the same range as FP32 but less precision. The standard format for training and for "unquantized" weights. [Quantization](#/foundations/quantization)

**Bits per weight (bpw)**: file size in bits divided by parameter count. Q4_K_M is roughly 4.8 to 5 bpw once its higher-precision tensors are counted. [Quantization](#/foundations/quantization)

**Blob (Ollama)**: a content-addressed file in `~/.ollama/models/blobs`, named after its SHA-256 hash. A manifest maps `name:tag` to the blobs that make up a model. [Ollama basics](#/ollama/basics)

**BM25**: a classic keyword-ranking formula. It's essential in RAG for exact strings like CVE IDs, function names and headers that embeddings handle poorly. [RAG](#/using/rag)

**BPE (byte-pair encoding)**: the algorithm that builds a tokenizer's vocabulary by repeatedly merging the most frequent adjacent pair of symbols. [Tokens](#/foundations/tokens)

**Catastrophic forgetting**: narrow fine-tuning erasing general abilities the model had before. [Training 101](#/finetune/concepts)

**Causal mask**: the rule that a token can only attend to itself and earlier tokens, never later ones. [Transformer](#/foundations/transformer)

**Chain-of-thought**: reasoning written out before the answer, so later tokens can build on it. Thinking models do this inside a `<think>` block. [Prompt engineering](#/using/prompt-engineering)

**Chat template**: a Jinja program that turns a list of messages (system, user, assistant, tool) into the exact token string the model was trained on. Stored inside the GGUF. [Model types](#/foundations/model-types)

**ChatML**: the `<|im_start|>role … <|im_end|>` message format that Qwen models use. [Model types](#/foundations/model-types)

**Chunking**: splitting documents into retrievable pieces for RAG, usually with some overlap between neighbours. [RAG](#/using/rag)

**Constrained decoding**: masking out every token that would break a grammar or JSON schema before sampling, so the output is guaranteed to parse. [Structured output](#/using/structured-output-tools)

**Contamination (benchmark)**: benchmark questions that leaked into a model's training data, so the score reflects memory rather than skill. [Benchmarks and your own evals](#/models/evaluate)

**Context checkpoints**: saved snapshots of a hybrid model's recurrent state, so it can resume after a prompt edit instead of reprocessing everything. [Load settings](#/lmstudio/load-settings)

**Context length (context window)**: the maximum number of tokens the model can consider at once. The system prompt, history, tool results and the answer all share it. [Load settings](#/lmstudio/load-settings)

**Context overflow policy**: what LM Studio does when the context fills up: truncate the middle, use a rolling window, or stop. [Sampling settings](#/lmstudio/sampling-settings)

**Continued pretraining**: more next-token training on raw domain text. It needs far more data and compute than fine-tuning. [Beyond SFT](#/finetune/beyond)

**Cosine similarity**: how closely two vectors point in the same direction, from -1 to 1. The usual way to compare embeddings. [RAG](#/using/rag)

**CVSS**: the standard formula for scoring vulnerability severity from a vector of metrics. The model can help reason about it, but you decide. [Reporting](#/bugbounty/reporting)

## D–G

**Decode**: the token-by-token generation phase. It's limited by memory bandwidth, because every token reads all the weights. [Generation](#/foundations/generation), [Performance](#/lmstudio/performance)

**Delimiter**: a marker such as `<scan>…</scan>` that separates data from instructions in a prompt. Helpful, but not a security boundary. [Prompt engineering](#/using/prompt-engineering)

**Dense model**: a model that uses every parameter for every token. Your Qwen3.5-9B is dense. [Scaling](#/infra/scaling)

**Difference of means**: the simplest way to find a direction: average activations on examples with a property, subtract the average on examples without it. [Interpretability 101](#/weights/interpretability-101)

**Direction ablation**: removing a direction from activations or weights by projecting it out: x' = x − (x·r̂) r̂. [Direction ablation](#/weights/direction-ablation)

**Distillation**: training a smaller "student" model to imitate a stronger "teacher". [Beyond SFT](#/finetune/beyond)

**DPO (Direct Preference Optimization)**: preference training directly on pairs of chosen and rejected answers, with no separate reward model. [Beyond SFT](#/finetune/beyond)

**Draft model**: a small model with the same vocabulary that proposes tokens for speculative decoding. [Right model for each job](#/models/use-cases)

**Effective parameters (E2B, E4B)**: the parameters doing per-token compute, excluding large embedding tables. A naming style used by some small models. [Open-model landscape](#/models/landscape)

**Embedding**: a vector that represents meaning. Inside the model, each token id maps to a 4,096-number embedding; in RAG, a separate embedding model turns passages into vectors. [Transformer](#/foundations/transformer), [RAG](#/using/rag)

**`enable_thinking`**: the Qwen3.5 chat-template variable that switches reasoning on or off. The official template thinks unless it's explicitly false. [Prompts and templates](#/lmstudio/prompts-templates)

**Epoch / step / batch**: one full pass over the training data / one weight update / the examples used per update. [Training 101](#/finetune/concepts)

**Excessive agency**: an agent holding more capability than its task needs (OWASP LLM06). [Threat model](#/security/ai-threat-model)

**Few-shot prompting**: putting worked examples in the prompt so the model copies the pattern. The weights don't change. [Prompt engineering](#/using/prompt-engineering)

**Fine-tuning (SFT)**: further training on (prompt, ideal answer) examples to change style, format or task behaviour. [Training 101](#/finetune/concepts)

**Flash Attention**: an exact, tiled way of computing attention that never stores the full score matrix. In llama.cpp it's required before the V cache can be quantized. [Load settings](#/lmstudio/load-settings)

**Forward pass**: running input through every layer once to get output logits. [What is an LLM?](#/foundations/what-is-an-llm)

**Gated DeltaNet**: the linear-attention layer type in Qwen3.5. Instead of looking back at every token, it keeps a fixed-size memory per head, updated by a decay gate and a corrective "delta" rule. [Transformer](#/foundations/transformer)

**GBNF**: llama.cpp's grammar format for constrained decoding. JSON schemas are converted into it. [Structured output](#/using/structured-output-tools)

**GGUF**: llama.cpp's single-file model format: tensors, metadata, tokenizer and chat template together. What LM Studio loads. [Quantization](#/foundations/quantization)

**GPU offload**: how many layers run on the GPU. On Apple Silicon, set it to the maximum. [Load settings](#/lmstudio/load-settings)

**Gradient / gradient descent**: the slope of the loss for each weight, and the loop of stepping each weight against it to reduce the loss. [What is an LLM?](#/foundations/what-is-an-llm)

**Grounding**: putting authoritative text in the prompt and telling the model to answer only from it. The main defence against hallucinated facts. [Hallucinations](#/security/hallucination-verification)

**Grouped-query attention (GQA)**: several query heads sharing one key/value head. In your model 16 query heads share 4 KV heads, cutting the KV cache to a quarter. [Transformer](#/foundations/transformer)

**GRPO / RLVR**: reinforcement learning where rewards come from an automatic checker (a correct flag, valid JSON). GRPO compares each answer with a group of samples. [Beyond SFT](#/finetune/beyond)

**Guard model**: a small classifier that checks inputs or outputs against a policy. [Right model for each job](#/models/use-cases)

## H–L

**Hallucination**: fluent, confident output that is false or unsupported. It follows from training to produce plausible text. [Hallucinations](#/security/hallucination-verification)

**Hybrid architecture**: mixing layer types. Qwen3.5 uses three Gated DeltaNet layers for every full-attention layer, so only 8 of 32 layers keep a growing KV cache. [Transformer](#/foundations/transformer)

**Hybrid search**: combining keyword (BM25) and vector results, often with Reciprocal Rank Fusion. [RAG](#/using/rag)

**I-quants (IQ)**: llama.cpp quant types using non-uniform grids, strongest below 4 bits. [Quantization](#/foundations/quantization)

**Importance matrix (imatrix)**: calibration statistics showing which weights matter most, used to reduce quantization error. [Quantization](#/foundations/quantization)

**Indirect prompt injection**: malicious instructions hidden inside content the model reads (a web page, a JS file, a DNS record), not typed by the user. The central risk for security agents. [Threat model](#/security/ai-threat-model)

**Inference**: running a trained model to get outputs. Everything you do in LM Studio is inference. [What is an LLM?](#/foundations/what-is-an-llm)

**Inference engine (runtime)**: the software that loads weights and runs the forward pass: llama.cpp, MLX, vLLM. LM Studio manages llama.cpp and MLX runtimes separately from the app. [Ecosystem](#/using/ecosystem)

**Instruct model**: a base model post-trained on conversations so it follows instructions. [Model types](#/foundations/model-types)

**JIT loading / TTL**: loading a model when a request arrives, and unloading it after an idle time. [Server and API](#/lmstudio/server-api)

**JSONL**: one JSON object per line. The standard format for datasets and traces. [Dataset](#/finetune/dataset)

**K-quants**: llama.cpp quant types (Q4_K, Q6_K…) that use 256-weight super-blocks with their own quantized scales. [Quantization](#/foundations/quantization)

**Keep-alive (Ollama)**: how long a model stays loaded after its last request (default 5 minutes). A big reason Ollama can look lighter than LM Studio. [Server tuning](#/ollama/tuning)

**KL divergence**: how far one probability distribution is from another. Used to measure how much quantization or editing changed a model. [Quantization](#/foundations/quantization)

**Knowledge cutoff**: the date the training data ends. The model knows nothing after it, including new CVEs. [What is an LLM?](#/foundations/what-is-an-llm)

**Knowledge editing (ROME, MEMIT)**: changing specific facts by making targeted updates to MLP weights. [Editing and merging](#/weights/editing-merging)

**KV cache**: the stored keys and values of past tokens, so each new token only computes its own. For your model: 32 KiB per token at FP16, so 32K tokens is about 1 GiB. [Memory math](#/foundations/memory-math)

**KV cache quantization**: storing the KV cache at lower precision. Q8_0 roughly halves it. [Load settings](#/lmstudio/load-settings)

**Lethal trifecta**: an agent that has private data, reads untrusted content and can communicate externally, all at once. The combination that makes data exfiltration possible. [MCP](#/using/mcp)

**Licence inheritance**: a fine-tune, merge or quant carries the licence obligations of the model it came from. [Open-model landscape](#/models/landscape)

**Linear representation hypothesis**: the idea that concepts are encoded as directions in activation space. [Interpretability 101](#/weights/interpretability-101)

**LLM-as-judge**: using a model to grade outputs against a rubric. Useful but biased; always spot-check it. [Evaluate your fine-tune](#/finetune/evaluate)

**Logit lens**: projecting an intermediate layer's residual stream through the output head to see what the model "would say" at that depth. [Interpretability 101](#/weights/interpretability-101)

**Logits**: the raw score the model gives every token in its vocabulary before softmax. [Generation](#/foundations/generation)

**LoRA**: Low-Rank Adaptation. The original weights W stay frozen, and a small trainable product B·A is added. Trains well under 1% of the parameters. [Training 101](#/finetune/concepts)

**Loss**: one number measuring how wrong the model's predictions are. Training lowers it. [What is an LLM?](#/foundations/what-is-an-llm)

## M–P

**Matryoshka embeddings**: vectors whose leading dimensions still work as an embedding when you truncate them, trading accuracy for storage. [Right model for each job](#/models/use-cases)

**MCP (Model Context Protocol)**: an open protocol for connecting AI hosts like LM Studio to tool and data servers. [MCP](#/using/mcp)

**Memory bandwidth**: bytes per second between memory and the processor. It caps decode speed: tokens/s ≈ bandwidth ÷ model size. [Performance](#/lmstudio/performance)

**Mixture of experts (MoE)**: a router picks a few expert MLPs per token. Total parameters set memory use; active parameters set speed. [Scaling](#/infra/scaling)

**Min-p**: a sampler that keeps only tokens whose probability is at least min-p times the top token's. It adapts automatically to how confident the model is. [Sampling settings](#/lmstudio/sampling-settings)

**MLP (feed-forward block)**: the per-token part of each layer: expand 4,096 → 12,288, gate, compress back. Much of a model's stored knowledge lives here. [Transformer](#/foundations/transformer)

**MLX**: Apple's machine-learning framework and model format for Apple Silicon. [Quantization](#/foundations/quantization), [Fine-tune with MLX](#/finetune/mlx-mac)

**mmap / mlock**: memory-mapping the model file instead of reading it all in / pinning it so the OS can't page it out ("Keep Model in Memory"). [Load settings](#/lmstudio/load-settings)

**mmproj**: a separate GGUF holding the vision encoder. Needed only for image or video input. [Model types](#/foundations/model-types)

**Model merging**: combining the weights of several models with the same architecture (linear, SLERP, TIES, DARE). [Editing and merging](#/weights/editing-merging)

**Model provenance**: being able to prove where a model file came from: repository, commit and SHA-256. [Hardening](#/infra/hardening)

**Model shootout**: running the same tasks on several models under identical conditions and comparing scores, speed and memory. [Benchmarks and your own evals](#/models/evaluate)

**Modelfile**: a Dockerfile-like recipe (`FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`) that `ollama create` turns into a custom Ollama model. [Modelfiles](#/ollama/modelfile)

**MTP (multi-token prediction)**: an extra head that predicts several tokens ahead. It can act as a built-in draft model for speculative decoding. [Sampling settings](#/lmstudio/sampling-settings)

**Next-token prediction**: the training objective of every LLM: given text so far, predict the next token. [What is an LLM?](#/foundations/what-is-an-llm)

**Ollama**: an open-source local model runner: a server on `127.0.0.1:11434`, a CLI, and a model registry. [Ollama basics](#/ollama/basics)

**Open weights**: downloadable model parameters under a licence. Not the same as open source, which would also release the training data and code. [Open-model landscape](#/models/landscape)

**Over-refusal**: refusing safe requests because they look like unsafe ones. A common frustration in security work. [Alignment](#/foundations/alignment-uncensored)

**Overfitting**: training loss keeps falling while validation loss rises. The model is memorising, not learning. [Training 101](#/finetune/concepts)

**Parameter (weight)**: one learned number in the model. "9B" means about 9 billion of them. [What is an LLM?](#/foundations/what-is-an-llm)

**Perplexity**: e to the power of the average loss. Roughly how many tokens the model is "torn between". Lower is better. [Quantization](#/foundations/quantization)

**Pickle checkpoint**: a PyTorch `.bin` or `.pt` file. Loading one can execute arbitrary code, which makes it a malware vector. Prefer safetensors or GGUF. [Threat model](#/security/ai-threat-model)

**Prefill**: processing the whole prompt in parallel before the first output token. It's compute-bound and sets time to first token. [Generation](#/foundations/generation)

**Preset**: a saved bundle of system prompt and inference settings in LM Studio. [Prompts and templates](#/lmstudio/prompts-templates)

**Presence / frequency penalty**: subtracting from the logits of tokens that already appeared, to reduce repetition. Qwen recommends a presence penalty of 1.5 for general use. [Sampling settings](#/lmstudio/sampling-settings)

**Probllama (CVE-2024-37032)**: an Ollama path-traversal bug through `/api/pull` that led to file write and remote code execution. A reminder not to expose or pull from untrusted sources. [Securing Ollama](#/ollama/api-integration)

**Prompt injection**: attacker-controlled text that the model treats as instructions, much like SQL injection where data is executed as code. [Threat model](#/security/ai-threat-model)

**Prompt masking**: computing training loss only on the assistant's tokens, not on the prompt. [Unsloth](#/finetune/unsloth-cloud)

## Q–S

**QAT (quantization-aware training)**: simulating low precision during training so a quantized model loses less quality than one quantized afterwards. [Open-model landscape](#/models/landscape)

**QLoRA**: LoRA trained on top of a 4-bit quantized frozen base model. [Training 101](#/finetune/concepts)

**Quantization**: storing weights as small integers plus scale factors, to cut size and increase speed. [Quantization](#/foundations/quantization)

**RAG (retrieval-augmented generation)**: retrieving relevant passages from your documents and putting them in the prompt. An open-book exam for the model. [RAG](#/using/rag)

**ReAct**: the Reason + Act pattern: the agent alternates thoughts, tool calls and observations. [What is an agent?](#/agents/what-is-an-agent)

**Reasoning (thinking) model**: a model trained, mostly with reinforcement learning, to write a scratchpad before answering. [Model types](#/foundations/model-types)

**Repeat penalty**: llama.cpp's penalty that divides the logits of recently used tokens. Qwen recommends leaving it at 1.0 (off) and using presence penalty instead. [Sampling settings](#/lmstudio/sampling-settings)

**Reranker (cross-encoder)**: a model that scores a query and a passage together. More accurate than embeddings, so it's used to re-order the top results. [RAG](#/using/rag)

**Residual stream**: the vector for each position that every layer reads from and adds to. The model's shared "bus". [Transformer](#/foundations/transformer)

**Reward model / RLHF**: a model trained to score answers by human preference, and reinforcement learning against it. [Alignment](#/foundations/alignment-uncensored)

**RMSNorm**: rescales a vector to a standard size, with a learned gain per dimension. [Transformer](#/foundations/transformer)

**RoPE / YaRN**: rotary position encoding (position as a rotation of query and key pairs), and a method for stretching context beyond the trained length. [Load settings](#/lmstudio/load-settings)

**Router model**: a tiny model that classifies a request and sends it to the right specialist. [Right model for each job](#/models/use-cases)

**Runner (Ollama)**: the engine process that runs a loaded model, either llama.cpp's `llama-server` or MLX. [Ollama basics](#/ollama/basics)

**safetensors**: a weight format that is just a header plus raw bytes. It contains no code, so loading it is safe. [Quantization](#/foundations/quantization)

**Sampler chain**: the ordered set of filters applied to logits. In llama.cpp's default order, temperature comes last, after top-k, top-p and min-p. [Sampling settings](#/lmstudio/sampling-settings)

**Scope guard**: code that decides whether an agent may touch a host or URL. Deny by default, out-of-scope wins, fail closed. [Recon pipeline](#/bugbounty/recon-pipeline)

**Self-consistency**: sampling the same question several times and comparing answers. Disagreement signals a likely hallucination. [Hallucinations](#/security/hallucination-verification)

**Size tier**: a rough parameter-count class (about 1B, 4B, 9B, 30B…) that predicts what a model can do and what it needs. [Open-model landscape](#/models/landscape)

**Skill**: a packaged set of instructions (and optional scripts) that an agent loads only when it needs it. [Skills and instructions](#/agents/skills-instructions)

**Softmax**: turns logits into probabilities by exponentiating and normalising. [Generation](#/foundations/generation)

**Sparse autoencoder (SAE)**: a model trained to decompose activations into many interpretable features. [Interpretability 101](#/weights/interpretability-101)

**Special tokens**: reserved tokens that mark structure, such as `<|im_start|>`, `<|im_end|>` and `<think>`. [Model types](#/foundations/model-types)

**Speculative decoding**: a small draft model proposes several tokens and the big model checks them in one pass. Changes speed, never output. [Sampling settings](#/lmstudio/sampling-settings)

**Superposition**: models packing more features than they have dimensions by overlapping them. Why individual neurons are hard to interpret. [Interpretability 101](#/weights/interpretability-101)

**Swap**: memory pages moved to the SSD when RAM runs out. It makes generation drastically slower. [Memory math](#/foundations/memory-math)

**System prompt**: text placed in the chat template's system slot. It guides the model but enforces nothing. [Prompts and templates](#/lmstudio/prompts-templates)

## T–Z

**Task vector (task arithmetic)**: fine-tuned weights minus base weights. Can be added, scaled or subtracted from other models. [Editing and merging](#/weights/editing-merging)

**Temperature**: divides the logits before softmax. Below 1 sharpens the distribution, above 1 flattens it, 0 means always pick the top token (greedy). [Generation](#/foundations/generation)

**Tensor**: a named grid of numbers. A model file is mostly tensors plus metadata. [Inspect weights](#/weights/inspect-weights)

**Thermal throttling**: the chip slowing down under sustained heat. Common on the fanless MacBook Air during long runs. [Performance](#/lmstudio/performance)

**Time to first token (TTFT)**: the delay before output starts, dominated by prefill. [Performance](#/lmstudio/performance)

**Token**: the unit a model reads and writes, usually a subword, each with an integer id. Qwen3.5 has 248,320 of them. [Tokens](#/foundations/tokens)

**Tool calling (function calling)**: the model emits a structured request to call a function, your code runs it, and the result goes back as a `tool` message. [Structured output](#/using/structured-output-tools)

**Tool poisoning**: hidden instructions placed in a tool's description, which the model reads and obeys. [MCP](#/using/mcp)

**Top-k / top-p**: keep only the k most likely tokens / the smallest set of tokens whose probabilities add up to p. [Sampling settings](#/lmstudio/sampling-settings)

**Total vs active parameters**: in a mixture-of-experts model, memory follows the total count and speed follows the (smaller) active count. [Open-model landscape](#/models/landscape)

**Trusted quantizer**: a long-standing publisher of GGUF or MLX conversions (ggml-org, unsloth, bartowski, lmstudio-community, mlx-community). [Open-model landscape](#/models/landscape)

**Unified memory**: one pool of RAM shared by the CPU and GPU, as on Apple Silicon. No separate VRAM. [Your setup](#/start/your-setup)

**Validation set**: held-out data checked during training to catch overfitting. [Dataset](#/finetune/dataset)

**Weight-space programming**: an AI reading and directly editing another network's weights to change its behaviour, without gradient training. [MIT study](#/weights/mit-study)

**Wired memory limit**: the cap macOS puts on how much memory the GPU may pin, about two-thirds of RAM on a 16 GB Mac by default. [Memory math](#/foundations/memory-math)

## Recap

- Every definition here is a summary; the linked lesson has the full explanation, worked example and lab.
- Use search to find a term quickly, and the links to go deeper.

## Check yourself

1. Cover the definitions and explain these five terms out loud: KV cache, temperature, LoRA, indirect prompt injection, direction ablation.
   <details><summary>Answer</summary>

   KV cache: stored keys and values of past tokens (32 KiB per token for your model). Temperature: divides logits before softmax to sharpen or flatten choices. LoRA: a small trainable low-rank update added to frozen weights. Indirect prompt injection: instructions hidden in content the model reads. Direction ablation: projecting a direction out of activations or weights.

   </details>
2. What's the difference between an activation and a weight?
   <details><summary>Answer</summary>

   Weights are the learned numbers stored in the model file and fixed during inference. Activations are the intermediate vectors computed from them for a specific input during a forward pass, and they disappear when the pass ends.

   </details>
