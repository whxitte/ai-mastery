# 14-week study plan

A week-by-week path through the whole lab, sized for about 6 to 8 hours a week alongside a full-time job. Every week ends with something concrete in `~/ai-lab/`, and the checkboxes on this page remember your progress in this browser.

> [!GOALS]
> - Know exactly what to study each week and in what order
> - Have a concrete deliverable to prove each week is done
> - Hit five milestones that each unlock a new level of capability
> - Know the short paths if you have less time or want to go faster

## How to use this plan

Each week lists the lessons to read, the labs to finish, and one **deliverable**: a file, script or result that proves you can do the thing, not just that you read about it. Tick the boxes as you go. Lessons also have their own "Mark as complete" button at the bottom, which drives the progress bar in the sidebar.

A typical week looks like this:

| Session | Time | What to do |
|---|---|---|
| 1 | ~2 h | Read the first lessons slowly. Do every `CONCEPT` box in your own words in `NOTES.md`. |
| 2 | ~2 h | Do the labs. Record settings, numbers and surprises in the notebook. |
| 3 | ~2 h | Remaining lessons and labs. Build the deliverable. |
| 4 | ~1 h | Answer every "Check yourself" question without looking. Re-read what you missed. |

> [!TIP]
> Don't skip the "Check yourself" questions. Retrieval (trying to recall before you look) is the single most effective study technique there is, and it takes ten minutes per lesson.

> [!MAC]
> Labs that run the model for a long time (benchmarks, agents, fine-tuning) heat up a fanless Air and slow it down. Run them on a hard surface, plugged in, with heavy apps closed, and expect speeds to drop after a few minutes. The [performance lesson](#/lmstudio/performance) shows you how to measure this.

## Phase 1: Foundations (weeks 1–2)

### Week 1: what an LLM is

- [ ] [Welcome and how to use this lab](#/start/welcome)
- [ ] [Your machine and your model](#/start/your-setup)
- [ ] [What is an LLM, really?](#/foundations/what-is-an-llm)
- [ ] [Tokens and tokenization](#/foundations/tokens)
- [ ] [Inside the transformer](#/foundations/transformer)
- [ ] **Deliverable:** `~/ai-lab/NOTES.md` with your chip, memory, tokens/sec baseline, the gradient-descent script output, real Qwen tokenizer counts for five kinds of text, and your parameter-count estimate.

### Week 2: generation, formats and memory

- [ ] [How text gets generated](#/foundations/generation)
- [ ] [Base, instruct, thinking and chat templates](#/foundations/model-types)
- [ ] [Precision, quantization and file formats](#/foundations/quantization)
- [ ] [Memory math: will it fit?](#/foundations/memory-math)
- [ ] [Alignment, refusals and "uncensored"](#/foundations/alignment-uncensored)
- [ ] **Deliverable:** a table in your notebook of measured memory at 8K / 32K / 64K context next to your calculated numbers, plus the GGUF metadata dump of your model.

> [!NOTE] Milestone 1: you can explain every number
> By the end of week 2 you should be able to look at any model on Hugging Face and say, with arithmetic, whether it fits on your Mac, at what context, and roughly how fast it will run. You should also be able to explain what temperature does to the probability of each token.

## Phase 2: Mastering your model (weeks 3–5)

### Week 3: LM Studio, every knob

- [ ] [A guided tour of LM Studio](#/lmstudio/tour)
- [ ] [Load settings, every knob explained](#/lmstudio/load-settings)
- [ ] [Sampling and inference settings](#/lmstudio/sampling-settings)
- [ ] [System prompts, templates, thinking and presets](#/lmstudio/prompts-templates)
- [ ] **Deliverable:** a log of one-knob-at-a-time experiments (load settings and sampling settings) with memory, speed and output quality for each change.

### Week 4: presets, speed and the API

- [ ] [Recommended configs for your Mac](#/lmstudio/presets-for-you)
- [ ] [Performance tuning on Apple Silicon](#/lmstudio/performance)
- [ ] [Local server, API and the lms CLI](#/lmstudio/server-api)
- [ ] [Prompt engineering for small local models](#/using/prompt-engineering)
- [ ] **Deliverable:** six saved presets, a benchmark script with a results table (TTFT and tokens/sec), and `~/ai-lab/prompts/harness.py` comparing prompt variants.

### Week 5: structure, knowledge and tools

- [ ] [Structured output and tool calling](#/using/structured-output-tools)
- [ ] [RAG: give your model your knowledge](#/using/rag)
- [ ] [MCP: plugging tools into LM Studio](#/using/mcp)
- [ ] [The local AI ecosystem](#/using/ecosystem)
- [ ] **Deliverable:** a working RAG over your own notes and writeups that cites its sources, a custom MCP server connected to LM Studio, and `llama-server` running your GGUF directly.

> [!NOTE] Milestone 2: your model is a programmable service
> Your Qwen now runs as a local API that returns validated JSON, calls tools, searches your own knowledge and plugs into other apps. Everything after this point is built on that.

## Phase 3: Security, agents and the capstone (weeks 6–9)

### Week 6: AI for security work

- [ ] [Why local AI for security work](#/security/why-local)
- [ ] [Security workflows with your model](#/security/workflows)
- [ ] [Security personas and system prompts](#/security/system-prompts)
- [ ] [Trust but verify: hallucinations](#/security/hallucination-verification)
- [ ] [Threat-model your own AI setup](#/security/ai-threat-model)
- [ ] **Deliverable:** `~/ai-lab/sec/` with your stdin helper scripts, five tested personas saved as presets, and a write-up of the prompt-injection lab: what worked as a defence and what didn't.

### Week 7: building agents

- [ ] [What is an agent?](#/agents/what-is-an-agent)
- [ ] [Build an agent from scratch](#/agents/agent-from-scratch)
- [ ] [Memory, context and state](#/agents/memory-state)
- [ ] [Skills, instructions and structure](#/agents/skills-instructions)
- [ ] [Agent frameworks compared](#/agents/frameworks)
- [ ] [Making a 9B agent reliable](#/agents/reliability)
- [ ] **Deliverable:** your from-scratch agent with sandboxed tools, memory, a skills folder and a golden-task eval that it passes.

### Week 8: capstone, part one

- [ ] [Architecture of a local hunting agent](#/bugbounty/architecture)
- [ ] [Toolchain and a safe practice lab](#/bugbounty/setup-tools)
- [ ] [Scope guard and recon pipeline](#/bugbounty/recon-pipeline)
- [ ] [LLM analysis modules](#/bugbounty/llm-analysis)
- [ ] **Deliverable:** Juice Shop running locally, `scope.py` with passing unit tests, and the recon pipeline filling SQLite with LLM-triaged hosts and JS findings.

### Week 9: capstone, part two, and private infrastructure

- [ ] [Putting it together: the agent loop](#/bugbounty/agent-loop)
- [ ] [Reporting and human-in-the-loop](#/bugbounty/reporting)
- [ ] [Evaluate and improve your agent](#/bugbounty/evaluate)
- [ ] [Design a private offline AI stack](#/infra/offline-stack)
- [ ] [Hardening and air-gapping](#/infra/hardening)
- [ ] [Scaling up beyond 16 GB](#/infra/scaling)
- [ ] **Deliverable:** an end-to-end `hunt.py` run on Juice Shop producing a hunt brief, an eval scorecard, and your hardening checklist completed.

> [!NOTE] Milestone 3: a working, scoped hunting assistant
> You now have a private assistant that does recon with real tools, reads and prioritises everything with your local model, never leaves scope, and hands you a brief to verify by hand. This is the point where it starts saving you real time on authorised programs.

## Phase 4: Training and weight surgery (weeks 10–12)

### Week 10: fine-tuning foundations

- [ ] [Training 101 and when not to fine-tune](#/finetune/concepts)
- [ ] [Build a cybersecurity dataset](#/finetune/dataset)
- [ ] [Fine-tune on your Mac with MLX](#/finetune/mlx-mac)
- [ ] [Fine-tune in the cloud with Unsloth](#/finetune/unsloth-cloud)
- [ ] **Deliverable:** a clean train / valid / test JSONL dataset built from your own writeups, and one completed LoRA training run with its loss curve saved.

### Week 11: shipping and judging a fine-tune, then the study

- [ ] [Merge, convert to GGUF, load in LM Studio](#/finetune/export-gguf)
- [ ] [Evaluate your fine-tune](#/finetune/evaluate)
- [ ] [Beyond SFT: DPO, distillation and more](#/finetune/beyond)
- [ ] [The MIT weight-space programming study](#/weights/mit-study)
- [ ] [Mechanistic interpretability 101](#/weights/interpretability-101)
- [ ] **Deliverable:** your fine-tuned model running in LM Studio, a base-vs-fine-tune scorecard, and a logit-lens plot of Qwen3.5-0.8B.

### Week 12: editing weights directly

- [ ] [Open and inspect real weights](#/weights/inspect-weights)
- [ ] [Activation steering](#/weights/steering)
- [ ] [Direction ablation: editing weights](#/weights/direction-ablation)
- [ ] [Knowledge editing and model merging](#/weights/editing-merging)
- [ ] [Lab: Claude as your weight programmer](#/weights/claude-lab)
- [ ] **Deliverable:** a tensor diff that confirms which weights the MIT edit changed, a working steering vector, your own direction-ablation result with before/after loop rates, and one full Claude Code weight-editing session with its edit log and held-out evaluation.

> [!NOTE] Milestone 4: you can change how a model thinks
> You can now change a model's behaviour four different ways (prompting, fine-tuning, steering and editing weights directly) and you know when each one is the right tool. That's genuinely rare knowledge, even among people who use LLMs every day.

## Phase 5: Ollama and model choice (weeks 13–14)

> [!TIP]
> These two parts don't depend on Phase 4. If you already prefer Ollama, do week 13 right after week 5, and week 14 right after week 6, so every later lab runs on whichever engine you like.

### Week 13: Ollama mastery

- [ ] [Ollama from zero: install, run, inspect](#/ollama/basics)
- [ ] [Modelfiles: import, customize, build your own](#/ollama/modelfile)
- [ ] [Server tuning and a fair benchmark vs LM Studio](#/ollama/tuning)
- [ ] [API, integrations and securing Ollama](#/ollama/api-integration)
- [ ] **Deliverable:** your HauhauCS GGUF imported into Ollama as three custom models (security, code, thinking), a tuned server config, a side-by-side benchmark table of Ollama vs LM Studio at identical settings, and one earlier script ported to Ollama.

### Week 14: choosing the right model

- [ ] [The open-model landscape](#/models/landscape)
- [ ] [The right model for each job](#/models/use-cases)
- [ ] [Benchmarks, leaderboards and your own evals](#/models/evaluate)
- [ ] **Deliverable:** a `model_shootout.py` scorecard comparing 3 to 4 models on your own security tasks, and a written decision: your daily driver for chat, code, security analysis, agents and embeddings.

> [!NOTE] Milestone 5: you choose tools on evidence
> You can run any open model on either engine, tune both, and pick the right model for a job based on measurements on your own Mac and your own tasks, not on leaderboard headlines.

## If you have less time

If you only have 3 to 4 hours a week, take each week over two weeks (a 28-week plan). If you need results fast, this is the shortest path to a useful security assistant, about 4 weeks:

1. [Your machine and your model](#/start/your-setup), [Memory math](#/foundations/memory-math), [How text gets generated](#/foundations/generation)
2. [Sampling settings](#/lmstudio/sampling-settings), [Recommended configs](#/lmstudio/presets-for-you), [Server and API](#/lmstudio/server-api)
3. [Prompt engineering](#/using/prompt-engineering), [Structured output](#/using/structured-output-tools), [Security workflows](#/security/workflows), [Hallucinations](#/security/hallucination-verification)
4. [Threat model](#/security/ai-threat-model), [Agent from scratch](#/agents/agent-from-scratch), then the capstone

Come back to the foundations afterwards. You'll understand them faster once you've felt the problems they explain.

## If you want to go faster

With 15+ hours a week you can do two weeks per week (a 7-week sprint). The one thing not to compress is the labs in Part 1 and Part 2: the intuition you build there, by watching memory and speed change as you turn each knob, is what makes everything later make sense.

## Keeping your lab notebook

Keep one Markdown file, `~/ai-lab/NOTES.md`, for the whole course. For every experiment, write down:

```text
## 2026-11-12  Context length vs memory
Setup:    Qwen3.5-9B Q4_K_M, LM Studio 0.x.y, llama.cpp runtime vX, flash attention on
Changed:  context 8K -> 32K -> 64K (KV cache F16)
Measured: 6.1 GB / 7.0 GB / 8.1 GB in Activity Monitor, 18.4 / 18.1 / 17.6 tok/s
Expected: +1 GiB per 32K tokens (32 KiB/token). Matches within ~10%.
Learned:  the hybrid architecture really does make long context cheap.
```

Six months from now this file will be worth more than any tutorial, including this one, because it's a record of what is actually true on your machine.

## Recap

- 14 weeks at 6 to 8 hours a week, in five phases: foundations, mastering your model, security and agents, training and weight surgery, then Ollama and model choice.
- Every week has a deliverable in `~/ai-lab/` that proves the skill.
- Five milestones: explain every number, model as a service, scoped hunting assistant, change how a model thinks, choose tools on evidence.
- Short on time? Halve the pace, or take the 4-week fast path to a security assistant first.

## Check yourself

1. What should exist on your disk at the end of week 5?
   <details><summary>Answer</summary>

   A RAG pipeline over your own notes that cites sources, a custom MCP server connected to LM Studio, and `llama-server` running your GGUF directly, on top of the presets, benchmark script and prompt harness from week 4.

   </details>
2. Why does the plan put fine-tuning *after* the capstone rather than before it?
   <details><summary>Answer</summary>

   Because you need a real workload and real failures before fine-tuning makes sense. The capstone shows you where prompting, RAG and structure stop being enough, and it produces your own data (traces, reports, triage outputs) that becomes the best training set you can get.

   </details>
3. Which part of the course should you never compress, even on the fast track?
   <details><summary>Answer</summary>

   The hands-on labs in Parts 1 and 2. Measuring memory, speed and output changes yourself is what builds the intuition every later module depends on.

   </details>
