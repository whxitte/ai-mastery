// Course structure. Each lesson id maps to content/<id>.md
window.COURSE = {
  title: "Local AI Lab",
  tagline: "Zero to elite: local LLMs, agents, security and weight-space engineering",
  parts: [
    {
      id: "start",
      title: "Start Here",
      lessons: [
        { id: "start/welcome", title: "Welcome and how to use this lab" },
        { id: "start/your-setup", title: "Your machine and your model" },
      ],
    },
    {
      id: "foundations",
      title: "1 · How LLMs Actually Work",
      lessons: [
        { id: "foundations/what-is-an-llm", title: "What is an LLM, really?" },
        { id: "foundations/tokens", title: "Tokens and tokenization" },
        { id: "foundations/transformer", title: "Inside the transformer" },
        { id: "foundations/generation", title: "How text gets generated" },
        { id: "foundations/model-types", title: "Base, instruct, thinking and chat templates" },
        { id: "foundations/quantization", title: "Precision, quantization and file formats" },
        { id: "foundations/memory-math", title: "Memory math: will it fit?" },
        { id: "foundations/alignment-uncensored", title: "Alignment, refusals and \"uncensored\"" },
      ],
    },
    {
      id: "lmstudio",
      title: "2 · LM Studio Mastery",
      lessons: [
        { id: "lmstudio/tour", title: "A guided tour of LM Studio" },
        { id: "lmstudio/load-settings", title: "Load settings, every knob explained" },
        { id: "lmstudio/sampling-settings", title: "Sampling and inference settings" },
        { id: "lmstudio/prompts-templates", title: "System prompts, templates, thinking and presets" },
        { id: "lmstudio/presets-for-you", title: "Recommended configs for your Mac" },
        { id: "lmstudio/performance", title: "Performance tuning on Apple Silicon" },
        { id: "lmstudio/server-api", title: "Local server, API and the lms CLI" },
      ],
    },
    {
      id: "using",
      title: "3 · Using Your Model Like a Pro",
      lessons: [
        { id: "using/prompt-engineering", title: "Prompt engineering for small local models" },
        { id: "using/structured-output-tools", title: "Structured output and tool calling" },
        { id: "using/rag", title: "RAG: give your model your knowledge" },
        { id: "using/mcp", title: "MCP: plugging tools into LM Studio" },
        { id: "using/ecosystem", title: "The local AI ecosystem" },
      ],
    },
    {
      id: "security",
      title: "4 · AI for Cybersecurity",
      lessons: [
        { id: "security/why-local", title: "Why local AI for security work" },
        { id: "security/workflows", title: "Security workflows with your model" },
        { id: "security/system-prompts", title: "Security personas and system prompts" },
        { id: "security/hallucination-verification", title: "Trust but verify: hallucinations" },
        { id: "security/ai-threat-model", title: "Threat-model your own AI setup" },
      ],
    },
    {
      id: "agents",
      title: "5 · Building AI Agents",
      lessons: [
        { id: "agents/what-is-an-agent", title: "What is an agent?" },
        { id: "agents/agent-from-scratch", title: "Build an agent from scratch" },
        { id: "agents/memory-state", title: "Memory, context and state" },
        { id: "agents/skills-instructions", title: "Skills, instructions and structure" },
        { id: "agents/frameworks", title: "Agent frameworks compared" },
        { id: "agents/reliability", title: "Making a 9B agent reliable" },
      ],
    },
    {
      id: "bugbounty",
      title: "6 · Capstone: Bug Bounty Agent",
      lessons: [
        { id: "bugbounty/architecture", title: "Architecture of a local hunting agent" },
        { id: "bugbounty/setup-tools", title: "Toolchain and a safe practice lab" },
        { id: "bugbounty/recon-pipeline", title: "Scope guard and recon pipeline" },
        { id: "bugbounty/llm-analysis", title: "LLM analysis modules" },
        { id: "bugbounty/agent-loop", title: "Putting it together: the agent loop" },
        { id: "bugbounty/reporting", title: "Reporting and human-in-the-loop" },
        { id: "bugbounty/evaluate", title: "Evaluate and improve your agent" },
      ],
    },
    {
      id: "finetune",
      title: "7 · Fine-Tuning and Training",
      lessons: [
        { id: "finetune/concepts", title: "Training 101 and when not to fine-tune" },
        { id: "finetune/dataset", title: "Build a cybersecurity dataset" },
        { id: "finetune/mlx-mac", title: "Fine-tune on your Mac with MLX" },
        { id: "finetune/unsloth-cloud", title: "Fine-tune in the cloud with Unsloth" },
        { id: "finetune/export-gguf", title: "Merge, convert to GGUF, load in LM Studio" },
        { id: "finetune/evaluate", title: "Evaluate your fine-tune" },
        { id: "finetune/beyond", title: "Beyond SFT: DPO, distillation and more" },
      ],
    },
    {
      id: "weights",
      title: "8 · AI Editing AI: Weight Surgery",
      lessons: [
        { id: "weights/mit-study", title: "The MIT weight-space programming study" },
        { id: "weights/interpretability-101", title: "Mechanistic interpretability 101" },
        { id: "weights/inspect-weights", title: "Open and inspect real weights" },
        { id: "weights/steering", title: "Activation steering" },
        { id: "weights/direction-ablation", title: "Direction ablation: editing weights" },
        { id: "weights/editing-merging", title: "Knowledge editing and model merging" },
        { id: "weights/claude-lab", title: "Lab: Claude as your weight programmer" },
      ],
    },
    {
      id: "infra",
      title: "9 · Private and Offline AI Infra",
      lessons: [
        { id: "infra/offline-stack", title: "Design a private offline AI stack" },
        { id: "infra/hardening", title: "Hardening and air-gapping" },
        { id: "infra/scaling", title: "Scaling up beyond 16 GB" },
      ],
    },
    {
      id: "ollama",
      title: "10 · Ollama Mastery",
      lessons: [
        { id: "ollama/basics", title: "Ollama from zero: install, run, inspect" },
        { id: "ollama/modelfile", title: "Modelfiles: import, customize, build your own" },
        { id: "ollama/tuning", title: "Server tuning and a fair benchmark vs LM Studio" },
        { id: "ollama/api-integration", title: "API, integrations and securing Ollama" },
      ],
    },
    {
      id: "models",
      title: "11 · Choosing the Right Model",
      lessons: [
        { id: "models/landscape", title: "The open-model landscape" },
        { id: "models/use-cases", title: "The right model for each job" },
        { id: "models/evaluate", title: "Benchmarks, leaderboards and your own evals" },
      ],
    },
    {
      id: "reference",
      title: "Reference",
      lessons: [
        { id: "reference/roadmap", title: "14-week study plan" },
        { id: "reference/cheatsheet", title: "Command cheatsheet" },
        { id: "reference/glossary", title: "Glossary" },
        { id: "reference/resources", title: "Resources and reading list" },
      ],
    },
  ],
};
