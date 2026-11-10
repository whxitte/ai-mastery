# Resources and reading list

A curated set of the best primary sources behind this lab: docs, papers, repos and videos, grouped by topic and ordered roughly from "start here" to "go deep". Every lesson also ends with its own Go deeper list.

> [!GOALS]
> - Know which official docs to check when a setting or flag changes
> - Pick the right next resource for whatever you want to go deeper on
> - Read the key papers in a sensible order

> [!TIP]
> When something in this lab disagrees with the current official docs, trust the docs and note it in your lab notebook. Tools in this space change monthly.

## Your model

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B): architecture, recommended sampling profiles, thinking mode. Read the "Best Practices" section.
- [Qwen3.5-9B config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json): the ground truth for layer counts, heads and dimensions.
- [Qwen3.5-9B chat template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja): exactly how messages, thinking and tool calls become tokens.
- [HauhauCS Qwen3.5-9B Uncensored Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive): your GGUF files and the author's notes.
- [Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5): running, quant choices and known issues.

## Foundations

- [Andrej Karpathy, Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g): the best one-hour overview.
- [Andrej Karpathy, Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI): pretraining to RLHF, in depth.
- [3Blue1Brown, Neural networks series](https://www.3blue1brown.com/topics/neural-networks): the visual intuition for gradients and attention.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/): an interactive transformer in your browser.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): the classic diagram-first explanation.
- [Andrej Karpathy, Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) and [minbpe](https://github.com/karpathy/minbpe): BPE from scratch.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course): free, hands-on, covers tokenizers through fine-tuning.
- [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/) and the [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464): the linear-attention layers inside Qwen3.5.

## LM Studio and inference engines

- [LM Studio docs](https://lmstudio.ai/docs): the app, presets, per-model defaults, MCP.
- [LM Studio developer docs](https://lmstudio.ai/docs/developer): OpenAI-compatible API, native REST API, structured output, tool use.
- [LM Studio CLI docs](https://lmstudio.ai/docs/cli) and [lms source](https://github.com/lmstudio-ai/lms).
- [LM Studio Python SDK](https://lmstudio.ai/docs/python).
- [llama.cpp](https://github.com/ggml-org/llama.cpp): the engine under LM Studio. Read the [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for every flag.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm): Apple's framework, plus LLM inference and fine-tuning on it.

## Prompting, structure, RAG and MCP

- [Lost in the Middle](https://arxiv.org/abs/2307.03172): why position in a long context matters.
- [Prompt Engineering Guide](https://www.promptingguide.ai/): a broad catalogue of techniques.
- [promptfoo](https://github.com/promptfoo/promptfoo): test prompts like code.
- [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md): how constrained decoding works under the hood.
- [Pydantic](https://docs.pydantic.dev/latest/): schemas and validation for structured output.
- [Chroma docs](https://docs.trychroma.com/) and [sqlite-vec](https://github.com/asg017/sqlite-vec): local vector stores.
- [Model Context Protocol](https://modelcontextprotocol.io), its [specification](https://modelcontextprotocol.io/specification/latest) and the [Python SDK](https://github.com/modelcontextprotocol/python-sdk).

## Security of and with AI

- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/): the shared vocabulary for LLM risks.
- [Simon Willison, The lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/): the clearest explanation of why agents leak data.
- [Invariant Labs, MCP tool poisoning](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks).
- [Databricks, GGML/GGUF file format vulnerabilities](https://www.databricks.com/blog/ggml-gguf-file-format-vulnerabilities) and [llama.cpp security advisories](https://github.com/ggml-org/llama.cpp/security/advisories): model files are attack surface.
- [JFrog, GGUF chat-template SSTI](https://research.jfrog.com/model-threats/gguf-ssti/).
- [Hugging Face, pickle scanning](https://huggingface.co/docs/hub/security-pickle): why `.bin` and `.pt` files are dangerous.
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) and [Self-Consistency](https://arxiv.org/abs/2203.11171).
- [NVD API](https://nvd.nist.gov/developers/vulnerabilities), [cve.org](https://www.cve.org/) and [GitHub Advisories](https://github.com/advisories): the sources to verify CVE claims against.

## Agents

- [Anthropic, Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): workflows vs agents, and the five patterns. Read this first.
- [Anthropic, Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents).
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
- [Agent Skills specification](https://agentskills.io/specification) and [anthropics/skills](https://github.com/anthropics/skills).
- [ReAct](https://arxiv.org/abs/2210.03629), [Reflexion](https://arxiv.org/abs/2303.11366) and [MemGPT](https://arxiv.org/abs/2310.08560): the key agent papers.
- Frameworks: [smolagents](https://huggingface.co/docs/smolagents/index), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), [PydanticAI](https://pydantic.dev/docs/ai/core-concepts/output/), [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).

## Bug bounty and web testing

- [OWASP Web Security Testing Guide](https://owasp.org/www-project-web-security-testing-guide/): the test categories your hypothesis generator maps to.
- [PortSwigger Web Security Academy](https://portswigger.net/web-security): free labs for every vulnerability class.
- [OWASP Juice Shop](https://owasp.org/www-project-juice-shop/) and [Pwning OWASP Juice Shop](https://pwning.owasp-juice.shop/): your practice target and its companion guide.
- [ProjectDiscovery docs](https://docs.projectdiscovery.io/): subfinder, dnsx, httpx, katana, nuclei.
- [ffuf](https://github.com/ffuf/ffuf).
- [FIRST CVSS v3.1](https://www.first.org/cvss/v3-1/) and [CVSS v4.0](https://www.first.org/cvss/v4-0/) specifications.
- [Bugcrowd on AI-slop submissions](https://www.bugcrowd.com/blog/bugcrowd-policy-changes-to-address-ai-slop-submissions/) and [Daniel Stenberg, The end of the curl bug-bounty](https://daniel.haxx.se/blog/2026/01/26/the-end-of-the-curl-bug-bounty/): why every AI-assisted finding must be verified by hand.

## Fine-tuning and training

- [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314): the two papers to read first.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673): what LoRA is and isn't good for.
- [LIMA](https://arxiv.org/abs/2305.11206): why a few hundred great examples can beat thousands of mediocre ones.
- [mlx-lm LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).
- [Unsloth Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) and [LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).
- [TRL docs](https://huggingface.co/docs/trl/sft_trainer) (SFT, DPO, GRPO) and [PEFT](https://huggingface.co/docs/peft/index).
- [DPO](https://arxiv.org/abs/2305.18290), [ORPO](https://arxiv.org/abs/2403.07691), [KTO](https://arxiv.org/abs/2402.01306) and [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300).
- Security benchmarks: [CTIBench](https://arxiv.org/abs/2406.07599), [CyberMetric](https://arxiv.org/abs/2402.07688), [Cybench](https://cybench.github.io/).
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Interpretability and weight editing

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), its [logs](https://github.com/RikiyaT/lm_understand-logs), the [edited model](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) and the [demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo).
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): the residual stream view.
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) and [Maxime Labonne's abliteration explainer](https://huggingface.co/blog/mlabonne/abliteration).
- Steering: [ActAdd](https://arxiv.org/abs/2308.10248), [Representation Engineering](https://arxiv.org/abs/2310.01405), [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) and [repeng](https://github.com/vgel/repeng).
- Editing: [ROME](https://arxiv.org/abs/2202.05262), [MEMIT](https://arxiv.org/abs/2210.07229), [Transformer FFN layers are key-value memories](https://arxiv.org/abs/2012.14913) and [EasyEdit](https://github.com/zjunlp/EasyEdit).
- Merging: [Task arithmetic](https://arxiv.org/abs/2212.04089), [TIES](https://arxiv.org/abs/2306.01708), [DARE](https://arxiv.org/abs/2311.03099) and [mergekit](https://github.com/arcee-ai/mergekit).
- [Claude Code setup](https://code.claude.com/docs/en/setup) and [permissions](https://code.claude.com/docs/en/permissions): for the weight-programming lab.

## Private infrastructure

- [Open WebUI](https://github.com/open-webui/open-webui) and its [environment reference](https://docs.openwebui.com/reference/env-configuration).
- [SearXNG with Docker](https://docs.searxng.org/admin/installation-docker.html).
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
- [LuLu outbound firewall](https://objective-see.org/products/lulu.html).
- [huggingface_hub environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables): for offline mode.
- [LM Studio offline operation](https://lmstudio.ai/docs/app/offline) and [privacy](https://lmstudio.ai/app-privacy).

## Recap

- Official docs first, whenever a setting or flag disagrees with a lesson.
- For each module, read the "start here" link before the papers.
- The MIT study links (blog, logs, edited model) are worth reading in full before Part 8.

## Check yourself

1. Where do you look first if an `lms` flag in a lesson doesn't work?
   <details><summary>Answer</summary>

   `lms <command> --help` on your machine, then the [LM Studio CLI docs](https://lmstudio.ai/docs/cli). Note the difference in your lab notebook.

   </details>
2. Which single article should you read before building any agent?
   <details><summary>Answer</summary>

   Anthropic's "Building effective agents", because it explains when a simple workflow beats an autonomous agent.

   </details>
