# Memory, context and state

Your agent from the last lesson forgets nothing and remembers nothing: it simply re-sends the entire conversation on every step until the context window fills up. This lesson treats the context window as what it is, a small and expensive working memory, and gives you the techniques to manage it: budgeting tokens, trimming and summarizing history, and moving knowledge out of the context into files, databases and vector stores.

> [!GOALS]
> - Explain why the model is stateless and what "memory" really means for an agent
> - Do token accounting for a 32K working budget on your Mac
> - Choose between sliding windows, tool-result eliding, rolling summaries, notes files, structured state and vector memory
> - Build a `Memory` class that counts tokens and summarizes old turns with the local model
> - Plug it into the agent from the previous lesson and watch it compact

## The context window is working memory

The model has no memory between calls. None. Each request to LM Studio contains everything the model will know for that step: system prompt, tool schemas, the task, every tool call and every result so far. The model reads it all, produces one reply, and forgets. "Memory" for an agent is a design decision about what you put in the next request.

> [!CONCEPT] Context window
> The **context window** is the maximum number of tokens (see [Tokens and tokenization](#/foundations/tokens)) the model can process in one request: the prompt plus the reply it generates. You set it when you load the model (`--context-length 32768`). Think of it as RAM: fast, directly usable, small, and wiped after every call. Files, databases and vector stores are the disk: large and persistent, but the model only sees what you load from them into RAM.

Why not just load the model with its native 262,144-token context and never worry? Three reasons, in order of how much they'll hurt you:

1. **Speed.** Before the first output token, the model must process the whole prompt (prefill, see [How text gets generated](#/foundations/generation)). On an Air, a 100K-token prompt takes a long time to process, and an agent sends a growing prompt on every step. LM Studio can reuse its cache when a new request starts with exactly the same tokens as the previous one, so appending is relatively cheap, but anything that changes the beginning of the prompt forces a full re-process.
2. **Quality.** Models use long contexts unevenly. "Lost in the Middle" (Liu et al., 2023) showed recall is best for information at the start and end of the context and worse in the middle; Anthropic's context-engineering guidance calls the gradual decline as context grows "context rot". A 9B model drowning in 80K tokens of old `grep` output makes worse decisions than one looking at 8K tokens of relevant facts.
3. **Memory.** For this model the KV cache is cheap, thanks to its hybrid attention (only 8 of 32 layers keep a growing cache): 32 KiB per token, so 32K is about 1 GiB and 128K about 4 GiB, on top of the ~5.3 GB of Q4_K_M weights. 32K fits comfortably under the default ~10–11 GB GPU limit with room for everything else; 128K is possible but tight. Details in [Memory math](#/foundations/memory-math).

<div data-widget="memory"></div>

> [!NOTE]
> The Qwen3.5 model card suggests keeping at least 128K context to preserve thinking quality on long tasks. That advice is about very long single reasoning chains. For a tool-using agent on 16 GB, a managed 32K window is the better trade: faster steps, and you control what's in it. If you run thinking-heavy tasks, test 64K too and measure.

## What fills the window

Here is the accounting for the lesson-2 agent with a 32,768-token context. Numbers are measured from `agent.py` (characters) and converted at roughly 3.5 characters per token, which is approximate; you'll calibrate it below.

| Item | Size | Tokens (approx.) | Notes |
|---|---|---|---|
| System prompt | 1,017 chars | ~300 | Every request |
| Tool schemas (4 tools) | 1,406 chars of JSON | ~400 | Rendered into the system prompt on every request |
| Template's tool instructions | fixed text in the chat template | ~300 | Added by Qwen3.5's template whenever tools are present |
| Skills catalog (next lesson) | ~100 tokens per skill | ~200 | Two skills |
| Reply reserve | `max_tokens` | 2,048–4,096 | Raise it with thinking on; reasoning counts too |
| Safety margin | estimation error | ~3,000–4,500 | Character counts are only an estimate |
| **Left for history** | | **~24,000** | Task, tool calls, tool results |

The arithmetic: 32,768 − 4,096 (reply, with some thinking) − ~4,700 (margin) ≈ 24,000 tokens for the whole prompt, of which about 1,200 are fixed overhead. Each tool result is capped at 4,000 characters (about 1,000–1,300 tokens), so roughly 17 maximum-size observations fill the window. A 15-step run that reads a few files gets there easily.

> [!TIP]
> Don't trust estimates when you can measure. LM Studio returns `usage.prompt_tokens` with every response, and since version 0.3.17 that count includes the system prompt and tool definitions. Your trace already logs it. The `Memory` class below uses it to correct its own estimates.

Exact counting options, from cheapest to most precise:

- **Characters ÷ ratio.** Free and instant. English prose with Qwen3.5's 248K-token vocabulary runs around 4 characters per token; code, JSON and logs run lower (3 or less). 3.5 is a conservative start.
- **Calibrate from `usage.prompt_tokens`.** After each call, compare your character count with the server's real count and update the ratio. Exact on average, after the fact.
- **A real tokenizer.** The `lmstudio` Python SDK has `model.count_tokens(text)`, and Hugging Face's `transformers` can load the Qwen3.5 tokenizer. Exact for a string, but chat-template tokens are extra and it adds a dependency.

## Strategies for managing memory

There's no single right answer, so learn the menu. In practice you combine several.

| Strategy | How it works | Good for | Watch out for |
|---|---|---|---|
| **Sliding window** | Keep the system prompt, the task, and the last N messages; drop the rest | Short tasks, chat | Silently loses early findings |
| **Elide old tool results** | Replace old observations with a stub like `[elided: 3,912 chars]`; keep the calls | Agents, whose bulk is tool output | The model may re-run tools to see results again |
| **Rolling summarization** | Summarize old turns with the model, keep the summary, drop the turns | Long tasks | Costs a model call; summaries can drop or distort facts |
| **Scratchpad / notes file** | The agent writes key facts to a file with a tool, and reads it back | Multi-step work with milestones | The model must remember to write notes |
| **Structured state** | Code stores facts in JSON or SQLite with a schema | Hosts, findings, TODOs: anything you'd put in a table | Needs tools and a schema up front |
| **Vector memory** | Embed past items, retrieve the most similar into the prompt | Large knowledge bases, past runs | Retrieval can miss or surface the wrong thing |

### Sliding window and the tool-call pairing rule

The simplest strategy has one trap. In the message list, an assistant message with `tool_calls` must be followed by the `tool` messages answering it. If your cut point lands between them, the window starts with an orphaned tool result that answers a call the model can't see. Servers and templates may reject it, and the model gets confused. So the cut point must move to a turn boundary: never start the recent window on a `tool` message.

There's a second rule from Qwen3.5's chat template: the system message must be the first message, and only the first. The template raises an error ("System message must be at the beginning.") if a system message appears later. So don't inject summaries as a second system message; merge them into a user message instead.

### Rolling summarization

When eliding isn't enough, summarize. Take everything between the task and the recent window, ask the model to compress it into notes, and replace those turns with the notes. Two rules make summaries trustworthy:

- **Keep the original task verbatim** at the top. Goal drift often starts when the task itself gets paraphrased away.
- **Summarize for recall first.** Exact paths, numbers, names, errors and dead ends matter more than prose. The prompt below asks for exactly that.

Because compaction rewrites the beginning of the prompt, LM Studio can't reuse its cache for the next step and must re-process the whole prompt. Compact rarely and in big chunks (when you cross the budget), not a little every step.

### Notes and structured state: memory outside the window

The most robust memory isn't in the context at all. Anthropic's context-engineering post calls this **structured note-taking**: the agent writes persistent notes outside the window and reads them back when needed. For a security agent, the most important facts (hosts, endpoints, findings, what was already tested) shouldn't live in a summary the model wrote; they should be rows in a table that code maintains. Deduplication, sorting and counting are then exact.

### Long-term memory: vector stores

For knowledge that outlives a single run, like notes from last week's recon or a library of past findings, you embed text into vectors and retrieve by similarity. This is RAG applied to the agent's own history; the mechanics (embeddings, chunking, vector databases) are in [RAG: give your model your knowledge](#/using/rag).

### Episodic, semantic and procedural memory

The "Cognitive Architectures for Language Agents" paper (Sumers et al., 2023) borrows three terms from cognitive science that are useful for deciding where things go:

> [!CONCEPT] Episodic, semantic and procedural memory
> **Episodic** memory is what happened: past runs, traces, "last Tuesday the login endpoint returned 500 for long usernames". **Semantic** memory is facts about the world: "api.example.com runs nginx 1.25 and is in scope". **Procedural** memory is how to do things: the system prompt, tool descriptions and playbooks. In your agent: traces and run summaries are episodic, the SQLite facts table is semantic, and skills (next lesson) are procedural.

## Build it: the `Memory` class

`memory.py` sits next to `agent.py` and does four things: estimates the prompt size, calibrates its estimate against the server's real count, elides old tool results (cheap, no model call), and if that isn't enough, summarizes old turns with the local model.

```python title="memory.py"
"""memory.py: token-aware working memory with tool-result eliding and rolling summaries."""
from __future__ import annotations

import json

from agent import MODEL, Agent, strip_think

SUMMARY_PROMPT = """You compress the working memory of a software agent.
Write compact notes (at most 250 words) that keep:
- facts discovered, with exact file paths, numbers, names and values
- actions already taken and what they returned (including errors and dead ends)
- open questions and what still needs doing
Never invent anything that is not in the transcript. Use bullet points."""


class Memory:
    def __init__(self, client, budget: int = 24_000, keep_last: int = 6, chars_per_token: float = 3.5):
        self.client, self.budget, self.keep_last = client, budget, keep_last
        self.cpt = chars_per_token        # refined by calibrate()
        self.task: str | None = None      # the original user task, never summarized away
        self.summary = ""                 # rolling summary of everything cut so far

    def _chars(self, messages: list[dict], tools: list | None) -> int:
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) + len(json.dumps(tools or []))

    def estimate(self, messages: list[dict], tools: list | None = None) -> int:
        return int(self._chars(messages, tools) / self.cpt) + 4 * len(messages)  # +4: per-message template tokens

    def calibrate(self, messages: list[dict], tools: list | None, prompt_tokens: int) -> None:
        """Learn the real chars-per-token ratio from the server's usage.prompt_tokens."""
        if prompt_tokens > 0:
            observed = self._chars(messages, tools) / prompt_tokens
            self.cpt = 0.7 * self.cpt + 0.3 * observed  # smooth it; one odd call should not swing it

    def _cut_index(self, messages: list[dict]) -> int:
        """Start of the 'recent' window. Never start on a tool result: it must follow its tool call."""
        cut = max(2, len(messages) - self.keep_last)
        while cut > 2 and messages[cut]["role"] == "tool":
            cut -= 1
        return cut

    def compact(self, messages: list[dict], tools: list | None = None) -> list[dict]:
        if self.task is None:
            self.task = messages[1]["content"]
        if self.estimate(messages, tools) <= self.budget:
            return messages
        cut = self._cut_index(messages)
        # Pass 1 (cheap, no model call): stub out old tool results.
        slim = [dict(m, content=f"[old tool result elided: {len(m['content'])} chars; re-run the tool if needed]")
                if m["role"] == "tool" and i < cut and not m["content"].startswith("[old tool result") else m
                for i, m in enumerate(messages)]
        if self.estimate(slim, tools) <= self.budget or cut <= 2:
            return slim
        # Pass 2: summarize everything between the task and the recent window.
        self.summary = self.summarize(messages[2:cut])
        head = [messages[0], {"role": "user", "content": f"{self.task}\n\n## Progress so far "
                              f"(summary of earlier steps)\n{self.summary}"}]
        return head + messages[cut:]

    def summarize(self, old: list[dict]) -> str:
        lines = [f"Previous summary:\n{self.summary}"] if self.summary else []
        for m in old:
            if m["role"] == "assistant":
                calls = [f"{c['function']['name']}({c['function']['arguments']})" for c in m.get("tool_calls", [])]
                lines.append(f"ASSISTANT: {m['content'][:500]} {' '.join(calls)}")
            else:
                lines.append(f"{m['role'].upper()}: {m['content'][:1500]}")
        resp = self.client.chat.completions.create(
            model=MODEL, temperature=0.2, max_tokens=800,
            messages=[{"role": "system", "content": SUMMARY_PROMPT},
                      {"role": "user", "content": "\n".join(lines)}])
        return strip_think(resp.choices[0].message.content)


class MemoryAgent(Agent):
    """The lesson-2 agent with a context budget."""

    def __init__(self, *args, budget: int = 24_000, keep_last: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = Memory(self.client, budget=budget, keep_last=keep_last)

    def call_model(self, use_tools: bool = True):
        tools = [t.schema() for t in self.tools.values()] if use_tools else None
        before = self.memory.estimate(self.messages, tools)
        self.messages = self.memory.compact(self.messages, tools)
        after = self.memory.estimate(self.messages, tools)
        if after < before:
            self.log(type="compact", before=before, after=after, summary=self.memory.summary)
        choice = super().call_model(use_tools)
        if self.last_usage:
            self.memory.calibrate(self.messages, tools, self.last_usage.prompt_tokens)
        return choice


if __name__ == "__main__":
    import argparse

    from agent import set_jail

    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--budget", type=int, default=24_000)
    a = ap.parse_args()
    set_jail(a.workdir)
    agent = MemoryAgent(budget=a.budget)
    print(agent.run(a.task))
    print(f"\ntrace: {agent.trace_path}")
```

How it fits together:

- **`compact()` runs before every model call.** Under budget, it returns the messages untouched, so there's zero cost on short runs.
- **Pass 1 elides** tool results older than the recent window. The calls stay, so the model still knows what it did and can re-run a tool if it needs the output again. This alone often buys back half the window.
- **Pass 2 summarizes** messages between the task and the recent window into at most 250 words, merges them with any previous summary, and rebuilds the history as: system prompt, a user message with the verbatim task plus the summary, then the recent turns.
- **`_cut_index()`** enforces the pairing rule by moving the cut point back until it isn't on a `tool` message.
- **`calibrate()`** blends the observed characters-per-token ratio from `usage.prompt_tokens` into its estimate (70% old, 30% new), so it self-corrects for code-heavy or log-heavy runs.
- **`MemoryAgent`** subclasses the lesson-2 `Agent`, overrides only `call_model`, and logs a `compact` event with the before/after estimate and the summary. The loop, tools and guardrails are unchanged.

> [!WARNING]
> The summarizer is the same 9B model, and summaries can lose or invent facts. That's why the task stays verbatim, why the prompt forbids invention, and why anything you must not lose belongs in a notes file or SQLite (below), not only in a summary.

### Watch it compact

The default budget of 24,000 rarely triggers on the sandbox. Force it with a tiny budget:

```bash
python memory.py "Audit this folder: list every TODO, then summarize each log file." --workdir sandbox --budget 3000
jq -c 'select(.type=="compact") | {before, after}' traces/run-*.jsonl | tail -n 3
jq -r 'select(.type=="compact") | .summary' traces/run-*.jsonl | tail -n 20
```

Read the summary and compare it with the tool results earlier in the same trace. Is every number right?

## Memory outside the context

### A notes file the agent writes to

Two tiny tools give the agent a scratchpad. The file lives inside the jail, so the path checks still apply.

```python title="notes_tools.py"
"""notes_tools.py: a scratchpad file the agent writes to, so facts survive compaction."""
from agent import Tool, inside_jail

NOTES_FILE = "AGENT_NOTES.md"


def write_note(text: str) -> str:
    path = inside_jail(NOTES_FILE)
    with path.open("a") as f:
        f.write(f"- {text.strip()}\n")
    return f"saved to {NOTES_FILE}"


def read_notes() -> str:
    path = inside_jail(NOTES_FILE)
    return path.read_text() if path.exists() else "(no notes yet)"


NOTE_TOOLS = {
    "write_note": Tool("write_note", "Save one short fact, decision or TODO to your notes file. Use it for anything "
                       "you must remember later (paths, counts, findings). One fact per call.",
                       {"type": "object", "properties": {"text": {"type": "string", "description": "the fact"}},
                        "required": ["text"]}, write_note),
    "read_notes": Tool("read_notes", "Read your notes file. Use it after a long task or when unsure what you found.",
                       {"type": "object", "properties": {}}, read_notes),
}
```

Add them to the agent and tell it to use them:

```python title="run_with_notes.py"
from agent import SYSTEM_PROMPT, TOOLS, set_jail
from memory import MemoryAgent
from notes_tools import NOTE_TOOLS

if __name__ == "__main__":
    set_jail("sandbox")
    system = SYSTEM_PROMPT + "\n\nAfter each important finding, save it with write_note. Before your final answer, call read_notes."
    agent = MemoryAgent(tools=dict(TOOLS, **NOTE_TOOLS), system=system, budget=3000)
    print(agent.run("Audit this folder: list every TODO, then summarize each log file."))
```

### Structured state in SQLite

For facts with a natural key, such as a host, a file:line, or an error signature, let code keep the table. The `ON CONFLICT ... DO UPDATE` upsert means recording the same fact twice updates it instead of duplicating it: deduplication is exact, not a hope.

```python title="state.py"
"""state.py: structured agent state in SQLite (facts that code, not the model, keeps track of)."""
import json
import sqlite3
from pathlib import Path

from agent import Tool

DB = sqlite3.connect(Path(__file__).resolve().parent / "state.db")
DB.execute("""CREATE TABLE IF NOT EXISTS facts (
    kind TEXT, key TEXT, value TEXT, source TEXT, updated TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (kind, key))""")


def record_fact(kind: str, key: str, value: str, source: str = "") -> str:
    DB.execute("INSERT INTO facts(kind, key, value, source) VALUES (?, ?, ?, ?) "
               "ON CONFLICT(kind, key) DO UPDATE SET value=excluded.value, source=excluded.source, "
               "updated=CURRENT_TIMESTAMP", (kind, key, value, source))
    DB.commit()  # upsert: recording the same host twice cannot create duplicates
    return f"recorded {kind}:{key}"


def query_facts(kind: str) -> str:
    rows = DB.execute("SELECT key, value, source FROM facts WHERE kind=? ORDER BY key LIMIT 200", (kind,)).fetchall()
    return json.dumps([{"key": k, "value": v, "source": s} for k, v, s in rows]) if rows else "[]"


STATE_TOOLS = {
    "record_fact": Tool("record_fact", "Store or update one structured fact, e.g. kind='todo', key='src/app.py:2'.",
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "description": "category, e.g. 'todo', 'host', 'error'"},
                            "key": {"type": "string", "description": "unique id within the kind"},
                            "value": {"type": "string", "description": "the fact itself"},
                            "source": {"type": "string", "description": "where you saw it (file:line, URL)"}},
                         "required": ["kind", "key", "value"]}, record_fact),
    "query_facts": Tool("query_facts", "List every stored fact of one kind (max 200), as JSON.",
                        {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]},
                        query_facts),
}
```

This is the pattern the capstone uses for hosts and findings. The model records and queries; code guarantees uniqueness and survives crashes, restarts and compaction. `sqlite3` ships with Python, so there's nothing to install.

### Episodic memory with embeddings

To let the agent learn from previous runs, store a short summary of each run with an embedding of the task, and at the start of a new run recall the most similar past episodes. LM Studio serves embeddings at the same `/v1` endpoint; load an embedding model as in the RAG lesson and use its name.

```python title="episodes.py"
"""episodes.py: long-term episodic memory: past run summaries, recalled by embedding similarity."""
import json
import math
from pathlib import Path

from openai import OpenAI

from agent import BASE_URL

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"  # use the embedding model you loaded (lms ls)
STORE = Path(__file__).resolve().parent / "episodes.jsonl"
client = OpenAI(base_url=BASE_URL, api_key="lm-studio")


def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


def remember(task: str, outcome: str) -> None:
    """Store one episode: what was asked and what happened."""
    with STORE.open("a") as f:
        f.write(json.dumps({"task": task, "outcome": outcome[:1500], "vec": embed(task)}) + "\n")


def recall(task: str, k: int = 3, min_score: float = 0.6) -> str:
    """Return the k most similar past episodes as text for the prompt."""
    if not STORE.exists():
        return ""
    q = embed(task)
    eps = [json.loads(line) for line in STORE.read_text().splitlines()]
    scored = sorted(((cosine(q, e["vec"]), e) for e in eps), key=lambda t: t[0], reverse=True)[:k]
    return "\n".join(f"- (similarity {s:.2f}) Task: {e['task']}\n  Outcome: {e['outcome'][:400]}"
                     for s, e in scored if s >= min_score)
```

Wire it in at the edges of a run, not inside the loop:

```python
from agent import set_jail
from episodes import recall, remember
from memory import MemoryAgent

if __name__ == "__main__":
    set_jail("sandbox")
    task = "Summarize the errors in logs/app.log"
    past = recall(task)
    agent = MemoryAgent()
    prompt = task + (f"\n\nNotes from similar past runs (may be outdated):\n{past}" if past else "")
    answer = agent.run(prompt)
    remember(task, answer)
    print(answer)
```

A pure-Python cosine over a JSONL file is fine for hundreds of episodes. For thousands, use a real vector store (the RAG lesson covers options). Treat recalled episodes as hints, label them as possibly outdated, and never as instructions: if a past run ingested an injection, recall can replay it.

> [!LAB] Make memory visible
> 1. Run `memory.py` on the audit task with `--budget 3000` and again with the default budget. Compare the number of steps, the `prompt_tokens` curve (`jq -c 'select(.type=="llm") | .usage.prompt_tokens'`) and the final answers.
> 2. Find the `compact` events. Check the summary line by line against the tool results that were summarized. Note every fact that got lost or changed.
> 3. Run `run_with_notes.py`. Open `sandbox/AGENT_NOTES.md`. Did the notes survive compaction better than the summary did?
> 4. Add `STATE_TOOLS` from `state.py` to the agent and ask it to record every TODO as `kind='todo'`, `key='<file>:<line>'`. Run it twice. Then check `sqlite3 state.db "select * from facts"`: are there duplicates?
> 5. Look at the `cpt` value the `Memory` class converges to on log-heavy runs versus code-heavy runs (add a `print(self.memory.cpt)` at the end). How far off was 3.5?

## Recap

- The model is stateless; every request carries everything it knows. Memory is what you choose to send.
- On a 16 GB Air, a managed 32K window beats a huge one: faster prefill, better attention, about 1 GiB of KV cache for this model.
- Budget the window explicitly: fixed overhead (system prompt, tool schemas, template text) of about 1,000 tokens, a reply reserve, a margin, and the rest for history. Calibrate estimates with `usage.prompt_tokens`.
- Trim in order of cost: elide old tool results first, summarize only when needed, keep the task verbatim, never split a tool call from its result, never add a second system message.
- Put durable facts outside the window: a notes file for the agent's own working notes, SQLite for structured facts, embeddings for long-term episodic recall.
- Episodic (what happened), semantic (facts) and procedural (how to) memory belong in different places.

## Check yourself

1. Your agent's history is `[system, user, assistant(tool_calls), tool, tool, assistant(tool_calls), tool, assistant]` and a naive window keeps the last 5 messages. What goes wrong, and how does `_cut_index` fix it?
   <details><summary>Answer</summary>

   The last 5 messages start at the second `tool` message, so the window begins with a tool result whose `tool_call` was cut off. The fix moves the cut point backwards while it sits on a `tool` message, until it reaches the assistant message that made the calls, so the recent window starts at a turn boundary.

   </details>

2. You load the model at 32,768 tokens, `max_tokens` is 2,048, and thinking is on. Why is a `Memory` budget of 30,000 a bad idea?
   <details><summary>Answer</summary>

   The context must hold the prompt and the reply. 30,000 + 2,048 is already within a few hundred tokens of 32,768, thinking can push the reply past 2,048 if you raise `max_tokens`, and the budget is checked against an estimate that can be off by 10% or more. One underestimate and the request overflows. Leave room for the reply plus a margin: around 24,000 is the safer number here.

   </details>

3. Why does `MemoryAgent` merge the summary into a user message instead of adding a new system message?
   <details><summary>Answer</summary>

   Qwen3.5's chat template only allows a system message at position 0 and raises "System message must be at the beginning." otherwise. Putting the summary in the first user message, after the verbatim task, keeps the conversation valid and keeps the goal at the top.

   </details>

4. Compaction rewrites the start of the prompt. What does that cost on LM Studio, and how should that affect when you compact?
   <details><summary>Answer</summary>

   LM Studio can reuse cached work only for a prompt that starts with the same tokens as the previous request. Rewriting the head invalidates that, so the next step re-processes the entire prompt. Compact rarely, when you actually cross the budget, and cut a big chunk each time, instead of trimming a little every step.

   </details>

5. During recon your agent discovers 300 subdomains. Where should they live: in the context, in a summary, in a notes file, or in SQLite? Why?
   <details><summary>Answer</summary>

   SQLite (structured state). 300 hosts would bloat the context, a model-written summary could drop or invent entries, and a free-text notes file can't be deduplicated or queried reliably. A table keyed by hostname gives exact deduplication, survives restarts and compaction, and lets code filter by scope before anything reaches the model, which then only sees the small slice it needs.

   </details>

## Go deeper

- [Effective context engineering for AI agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — context rot, compaction, structured note-taking and sub-agents, in one practical post
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — why more context is not the same as better recall
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — the context-as-RAM, storage-as-disk idea taken to its logical end
- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) — the episodic, semantic and procedural memory framing
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — memory streams, retrieval by relevance and recency, and reflection
