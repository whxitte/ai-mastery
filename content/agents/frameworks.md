# Agent frameworks compared

You've built an agent loop by hand, so you know exactly what a framework would be doing for you. This lesson compares the main Python agent frameworks honestly, from the point of view of someone running a 9B model on a 16 GB Mac: what each one is, where it shines, where it fights a small local model, and how to point it at LM Studio. You'll run three of them against your local server and finish with a clear rule for when to keep your own loop.

> [!GOALS]
> - Describe what an agent framework adds on top of the loop you wrote
> - Compare seven frameworks plus MCP on strengths, small-model weaknesses and learning curve
> - Connect smolagents, PydanticAI and LangGraph to LM Studio and run a working example of each
> - Decide when to stay with your own loop and when to adopt a framework

## What a framework actually gives you

Every framework below wraps the same protocol you used in [Build an agent from scratch](#/agents/agent-from-scratch): send `messages` and `tools`, receive `tool_calls`, run them, append `tool` messages, repeat. What they add:

- **Tool schemas from type hints and docstrings**, so you don't write JSON Schema by hand
- **The loop**, with step limits, error handling and retries
- **Typed outputs**: validate the final answer against a schema and retry on failure
- **State and persistence**: checkpoints you can resume after a crash or a human decision
- **Human-in-the-loop**: pause before a risky action, resume with an answer
- **Multi-agent plumbing**: hand-offs, sub-agents, orchestrators
- **Tracing and integrations**: OpenTelemetry, dashboards, MCP clients, vector stores

What they cost: prompts you didn't write (often long and tuned for frontier models), abstractions between you and the wire format, version churn, and security boundaries that live in someone else's code. With a 9B model, the first cost is the big one: a framework's 2,000-token built-in system prompt is a large slice of a 32K window and may be written for a model much stronger than yours.

> [!NOTE]
> Versions checked for this lesson (November 2026): smolagents 1.26, PydanticAI 2.65, LangGraph 1.2, OpenAI Agents SDK 0.24, LM Studio Python SDK 1.5. These projects move fast; PydanticAI in particular ships new releases constantly. If an import fails, check the project's docs for the current name before anything else.

## The comparison

| Framework | What it is | Strengths | Weaknesses with a small local model | Learning curve | Pointing it at LM Studio |
|---|---|---|---|---|---|
| **Your own loop** | ~300 lines you wrote | Total control of prompts, tools and guardrails; nothing hidden | You build every feature yourself | You already did it | Native |
| **LM Studio SDK `.act()`** | LM Studio's own Python SDK (`lmstudio`), with an automatic multi-round tool loop | Plain functions as tools, round limit, callbacks for every stage, tool exceptions sent back to the model | Only works with LM Studio; less control over the loop's internals; the SDK source notes tool calling can't yet be combined with structured responses in `.act()` | Low | Connects to LM Studio directly (not via the OpenAI endpoint) |
| **smolagents** (Hugging Face) | Minimal agent library with two agent types: `CodeAgent` (actions written as Python code) and `ToolCallingAgent` (JSON tool calls) | Small and readable; code actions can compose several tools in one step; built-in sandbox options | Long built-in prompts; `CodeAgent` executes model-written Python, which needs a real sandbox; 9B models make more code mistakes | Low | `OpenAIModel(model_id, api_base, api_key)` |
| **LangGraph** (LangChain) | Agents as explicit graphs of nodes over shared state, with checkpointing and interrupts | Durable state, resumable runs, first-class human-in-the-loop, any control flow you can draw | Verbose; pulls in the LangChain ecosystem and its churn; you design the graph | Medium to high | `ChatOpenAI(model, base_url, api_key)` from `langchain-openai` |
| **PydanticAI** | Type-safe agent framework from the Pydantic team | Typed, validated outputs with automatic retries; dependency injection; good testing story; MCP support | Fast release cadence; for non-OpenAI servers you may need to tune a model "profile"; the default output mode relies on tool calling | Low to medium | `OpenAIChatModel(model, provider=OpenAICompatibleProvider(...))` |
| **CrewAI** | Role-based "crews" of agents with tasks and processes | Quick to express multi-role workflows; popular, many examples | Many LLM calls and long role prompts per task: slow on one fanless laptop; non-native providers go through LiteLLM | Low to start, medium to control | `LLM(model="openai/<model>", base_url=..., api_key=...)` per its docs' OpenAI-compatible examples |
| **OpenAI Agents SDK** | OpenAI's lightweight framework: agents, tools, hand-offs, guardrails, tracing | Clean primitives; hand-offs and guardrails built in | Designed around OpenAI models and the Responses API; tracing tries to upload to OpenAI unless disabled; structured output needs JSON-schema support | Low to medium | `OpenAIChatCompletionsModel(model, openai_client=AsyncOpenAI(base_url=...))` |
| **LlamaIndex** | Data framework with agent workflows (`FunctionAgent`, `AgentWorkflow`) | Best-in-class connectors, indexing and RAG; agents that live next to your data | Agent layer is secondary to the data layer; its own docs warn smaller models are less reliable as agents | Medium | `llama-index-llms-lmstudio` or `llama-index-llms-openai-like` integrations |
| **MCP** (not a framework) | Model Context Protocol: a standard way to expose tools to any agent or host | Write a tool server once, use it from LM Studio, Claude, and most frameworks above | Every MCP server you connect is code with access; tool descriptions from servers are prompts you didn't write | Low to use, medium to write | LM Studio is itself an MCP host; see [MCP: plugging tools into LM Studio](#/using/mcp) |

### MCP as the tool layer

Think of MCP as the "driver model" for tools. Instead of writing a `Tool(...)` for every framework, you run an MCP server that exposes tools, and any MCP-capable client consumes them: LM Studio's chat, smolagents (`MCPClient`, `ToolCollection.from_mcp`), PydanticAI (install `pydantic-ai-slim[mcp]`), the OpenAI Agents SDK (`agents.mcp`), and LangChain via `langchain-mcp-adapters`. For the capstone that means a scope-guarded recon tool server could be reused across frameworks. The security rule doesn't change: an MCP server is software you install and grant access to, and its tool descriptions go straight into your model's context.

## Example 1: smolagents

```bash
cd ~/ai-lab/agents && source .venv/bin/activate
pip install "smolagents[openai]"
```

```python title="smol_agent.py"
"""smol_agent.py: a smolagents ToolCallingAgent on LM Studio."""
from pathlib import Path

from smolagents import OpenAIModel, ToolCallingAgent, tool

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # check yours: lms ls
WORKDIR = Path("sandbox").resolve()


@tool
def count_todos(folder: str) -> str:
    """Count TODO markers in text files under a folder of the sandbox.

    Args:
        folder: Folder relative to the sandbox, for example "." or "src".
    """
    root = (WORKDIR / folder).resolve()
    if not root.is_relative_to(WORKDIR):
        return "ERROR: outside the sandbox"
    hits = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".txt", ".js"}:
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if "TODO" in line:
                    hits.append(f"{f.relative_to(WORKDIR)}:{n}: {line.strip()}")
    return "\n".join(hits) or "no TODOs found"


model = OpenAIModel(
    model_id=MODEL,
    api_base=BASE_URL,
    api_key="lm-studio",
    temperature=0.3,
)

agent = ToolCallingAgent(tools=[count_todos], model=model, max_steps=6)

if __name__ == "__main__":
    print(agent.run("List every TODO in the sandbox and group them by file."))
```

```bash
python smol_agent.py
```

smolagents prints a box for every step, with the tool call, the observation and token counts. Things to know:

- **`@tool` reads your docstring.** The `Args:` section is required: every argument must be documented there, or smolagents refuses to build the tool. That's a good habit forced on you.
- **The agent ends by calling a built-in `final_answer` tool.** A step that just writes text isn't treated as done the way your own loop treats it.
- **`OpenAIModel`** is the current class name; older tutorials use `OpenAIServerModel`, which still works as an alias. Extra keyword arguments such as `temperature` are passed through to every completion call.

To try the code-writing variant, swap in `CodeAgent(tools=[count_todos], model=model)`. It asks the model to write Python that calls your tools, which can do in one step what takes JSON calls three. Two cautions:

> [!WARNING]
> `CodeAgent` executes model-written Python. smolagents' default `LocalPythonExecutor` is a restricted interpreter (imports allowlisted, operation counts capped), but its own docs state that no local Python sandbox can be completely secure, and recommend remote or container execution for real isolation (`executor_type="docker"`, `"e2b"`, `"modal"` or `"blaxel"`). With an uncensored model and untrusted inputs such as web pages, use Docker or don't use `CodeAgent`.

## Example 2: PydanticAI with typed output

PydanticAI's headline feature is that the agent's result is a validated Pydantic object, not a string.

```bash
pip install "pydantic-ai-slim[openai]"
```

```python title="pai_agent.py"
"""pai_agent.py: a PydanticAI agent with typed output on LM Studio."""
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # check yours: lms ls


class Triage(BaseModel):
    """Triage of one log line."""
    severity: Literal["info", "low", "medium", "high"]
    category: str = Field(description="short label, e.g. 'auth', 'db', 'network'")
    summary: str = Field(max_length=200)


model = OpenAIChatModel(
    MODEL,
    provider=OpenAICompatibleProvider("lmstudio", base_url=BASE_URL, api_key="lm-studio"),
)

agent = Agent(
    model,
    output_type=Triage,
    instructions="You triage log lines for an on-call engineer. Use lookup_service when a service name appears.",
    model_settings={"temperature": 0.3},
    retries=2,
)


@agent.tool_plain
def lookup_service(name: str) -> str:
    """Return the owner and criticality of an internal service."""
    services = {"auth-api": "owner=identity-team criticality=high", "thumbs": "owner=media criticality=low"}
    return services.get(name, "unknown service")


if __name__ == "__main__":
    result = agent.run_sync("2026-11-01T10:00:01Z auth-api ERROR 37 failed logins for admin from 203.0.113.9 in 60s")
    print(repr(result.output))
    print(result.usage)
```

```bash
PYDANTIC_AI_NO_BANNER=1 python pai_agent.py
```

What's going on:

- **`OpenAICompatibleProvider("lmstudio", base_url=..., api_key=...)`** is the provider for any OpenAI-compatible server PydanticAI has no dedicated provider for. Its docs reserve `OpenAIProvider` for OpenAI itself or a transparent proxy to it. `OpenAIChatModel` speaks Chat Completions, which LM Studio serves.
- **`output_type=Triage`**: by default PydanticAI turns your model into an extra tool (named `final_result`) and the run ends when the model calls it with valid arguments. If validation fails (say, `severity="critical"`, which isn't in the `Literal`), the error goes back to the model and it retries, up to `retries` times.
- **Output modes.** Besides that default tool mode, `NativeOutput(Triage)` uses the server's JSON-schema response format (LM Studio constrains decoding with a grammar for GGUF models), and `PromptedOutput(Triage)` asks for JSON in the prompt and parses it. If tool calling is flaky for a particular task, try `NativeOutput`.
- **`@agent.tool_plain`** registers a tool from a plain function; `@agent.tool` is the variant that receives a run context for dependency injection.
- If a feature like thinking or structured output is ignored or rejected by your server, PydanticAI's docs point you to a **model profile** (`OpenAIModelProfile`) to tell it what the endpoint supports.

## Example 3: LangGraph with a human approval gate

LangGraph's strength is control flow you can see and pause. This example routes any call to a risky tool through an `approve` node that stops the graph and waits for a human, then resumes from a checkpoint. That's precisely the shape of the capstone's human-in-the-loop step.

```bash
pip install langgraph langchain-openai
```

```python title="lg_agent.py"
"""lg_agent.py: a LangGraph agent with a human approval gate on risky tools."""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # check yours: lms ls
RISKY = {"delete_file"}


@tool
def list_files() -> str:
    """List files in the demo folder."""
    return "notes.txt\nold.log\nreport.md"


@tool
def delete_file(name: str) -> str:
    """Delete one file from the demo folder (pretend)."""
    return f"deleted {name} (simulated)"


tools = [list_files, delete_file]
llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key="lm-studio", temperature=0.3).bind_tools(tools)


def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


def route(state: MessagesState):
    calls = state["messages"][-1].tool_calls
    if not calls:
        return END
    return "approve" if any(c["name"] in RISKY for c in calls) else "tools"


def approve(state: MessagesState):
    calls = state["messages"][-1].tool_calls
    answer = interrupt({"question": "Approve these tool calls?", "calls": calls})  # pauses the graph
    if answer == "yes":
        return Command(goto="tools")
    refusals = [{"role": "tool", "tool_call_id": c["id"], "content": "DENIED by operator"} for c in calls]
    return Command(goto="model", update={"messages": refusals})


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("approve", approve)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, ["tools", "approve", END])
builder.add_edge("tools", "model")
graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "demo-1"}}
    out = graph.invoke({"messages": [("user", "Clean up: delete old.log")]}, cfg)
    while "__interrupt__" in out:
        print(out["__interrupt__"][0].value)
        out = graph.invoke(Command(resume=input("yes/no? ").strip()), cfg)
    print(out["messages"][-1].content)
```

```bash
python lg_agent.py
```

The pieces:

- **State.** `MessagesState` is a shared dict holding the message list; every node reads it and returns updates.
- **Nodes and edges.** `model` calls the LLM; `route` inspects its tool calls and sends risky ones to `approve`, safe ones to `tools` (LangGraph's prebuilt `ToolNode`), and a text reply to `END`.
- **`interrupt(...)`** pauses the graph and surfaces its argument (which must be JSON-serializable) to the caller as `__interrupt__`. **`Command(resume=...)`** continues it; the value becomes `interrupt()`'s return value.
- **Checkpointer and `thread_id`.** `InMemorySaver` stores the paused state; the same `thread_id` is required to resume. Swap in a durable checkpointer (SQLite, Postgres) and you can resume after a reboot.
- **Gotcha:** on resume, the interrupted node runs again from its beginning, so code before `interrupt()` executes twice. Keep side effects after it.

LangChain also offers a prebuilt agent (`create_agent` in LangChain 1.x) with human-in-the-loop middleware, if you'd rather not draw the graph yourself.

## Two quick hookups

**LM Studio's own SDK.** `.act()` runs the whole tool loop for you. Exceptions raised in a tool are converted to text and sent back to the model by default (SDK 1.3.0 and later); `max_prediction_rounds` is the step budget.

```python title="lms_act.py"
from pathlib import Path

import lmstudio as lms  # pip install lmstudio

MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # check yours: lms ls
SANDBOX = Path("sandbox").resolve()


def count_errors(filename: str) -> int:
    """Count lines containing ERROR in one file inside the sandbox's logs folder."""
    path = (SANDBOX / "logs" / filename).resolve()
    if not path.is_relative_to(SANDBOX):
        raise ValueError("outside the sandbox")
    return sum("ERROR" in line for line in path.read_text().splitlines())


if __name__ == "__main__":
    model = lms.llm(MODEL)
    result = model.act("How many ERROR lines are in app.log?", [count_errors],
                       on_message=print, max_prediction_rounds=5)
    print(f"{result.rounds} rounds in {result.total_time_seconds:.1f}s")
```

**OpenAI Agents SDK.** Use the Chat Completions model class (most non-OpenAI servers don't implement the full Responses API) and disable tracing, which otherwise expects an OpenAI platform key.

```python title="oa_agent.py"
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # check yours: lms ls
set_tracing_disabled(True)


@function_tool
def service_owner(name: str) -> str:
    """Return the owning team of an internal service."""
    return {"auth-api": "identity-team"}.get(name, "unknown")


if __name__ == "__main__":
    model = OpenAIChatCompletionsModel(model=MODEL, openai_client=AsyncOpenAI(base_url=BASE_URL, api_key="lm-studio"))
    agent = Agent(name="ops", instructions="Answer briefly. Use tools for facts.", model=model, tools=[service_owner])
    print(Runner.run_sync(agent, "Who owns auth-api?").final_output)
```

(`pip install openai-agents`.) CrewAI and LlamaIndex follow the same idea; use the connection settings in the table and their docs' examples.

> [!LAB] Same task, three frameworks
> 1. Run `smol_agent.py`, `pai_agent.py` and `lg_agent.py` against LM Studio. In `lg_agent.py`, answer `no` once and watch the denial reach the model.
> 2. Run `lms log stream` in a second terminal during each run and note the size of the first request each framework sends. Compare it with your own `agent.py` on the TODO task (`usage.prompt_tokens` of the first `llm` event in its trace).
> 3. In `pai_agent.py`, tighten `summary` to `Field(max_length=15)` so most summaries fail validation, and re-run. The printed usage shows more `requests` as PydanticAI feeds the errors back and retries; with luck it eventually fails with a validation error after `retries` attempts.
> 4. Replace `output_type=Triage` with `output_type=NativeOutput(Triage)` (import `NativeOutput` from `pydantic_ai`). Does it behave differently on your model?
> 5. Write down, for each framework, one thing it did that you'd have to build yourself, and one thing it hid that you'd want to see.

## When to keep your own loop, and when to adopt a framework

Here's the recommendation for your setup, labeled as such: it's a judgment, not a benchmark.

**Stay with your own loop when:**

- You're learning, or the agent has a handful of tools. 300 lines you understand beat 30,000 you don't.
- Security boundaries are the point. For the capstone, the scope guard, approval gate and rate limiter must sit exactly where you can audit them.
- You need to tune every token of the prompt for a 9B model, including the tool instructions.
- You want stable code that won't break on a dependency's next minor release.

**Adopt a framework when it buys a specific capability you'd otherwise build:**

| You need | Reach for |
|---|---|
| Validated, typed outputs from many different steps | PydanticAI (or this module's own `call_with_schema` from [Making a 9B agent reliable](#/agents/reliability)) |
| Durable, resumable runs with human approval in the middle, or complex branching | LangGraph |
| Quick experiments, or code-writing agents inside a container sandbox | smolagents |
| An agent that mostly lives on top of your documents and indexes | LlamaIndex |
| Tools shared across LM Studio chat, Claude and several frameworks | MCP servers |
| The simplest possible tool loop, LM Studio only | LM Studio SDK `.act()` |

Multi-role frameworks such as CrewAI are easy to start with, but on one fanless 16 GB machine every extra role is another long prompt on the same model, run sequentially. Measure before committing.

The capstone in module 6 builds on your own loop from this module, plus the validation helpers from the next lesson. Any piece of it can be swapped for a framework later, and having built it by hand, you'll know exactly what you're swapping.

## Recap

- Frameworks wrap the same tool-calling protocol you built; they add schemas from type hints, loops, typed outputs, state, human-in-the-loop, multi-agent plumbing and tracing.
- Their main cost on a 9B model is prompts and abstractions you didn't write, plus version churn.
- smolagents: `OpenAIModel(model_id, api_base, api_key)`, `ToolCallingAgent` for JSON calls, `CodeAgent` only with a real sandbox.
- PydanticAI: `OpenAIChatModel` + `OpenAICompatibleProvider`, typed `output_type`, automatic validation retries, and tool, native or prompted output modes.
- LangGraph: explicit graphs over state, `interrupt()` and `Command(resume=...)` with a checkpointer for human approval.
- OpenAI Agents SDK needs `OpenAIChatCompletionsModel` and disabled tracing; LM Studio's `.act()` is the simplest loop; MCP is the portable tool layer.
- Keep your own loop when control and auditability matter most; adopt a framework for a specific capability.

## Check yourself

1. Why is `CodeAgent` riskier than `ToolCallingAgent`, and what does smolagents itself recommend?
   <details><summary>Answer</summary>

   `CodeAgent` executes Python the model writes, so a mistake or a prompt injection becomes code running on your machine, whereas `ToolCallingAgent` can only call the specific tools you exposed. smolagents' local executor restricts imports and operation counts, but its docs say no local Python sandbox can be completely secure and recommend remote or container executors (Docker, E2B, Modal, Blaxel) for real isolation.

   </details>

2. In PydanticAI, what happens by default when the model returns output that fails your Pydantic model's validation?
   <details><summary>Answer</summary>

   The default output mode exposes your output type as a tool (`final_result`). When the model calls it with arguments that fail validation, PydanticAI sends the validation error back to the model and lets it try again, up to the agent's `retries` setting, then raises an error if it still fails.

   </details>

3. In the LangGraph example, why does the approval logic live in its own node, and why must side effects in that node come after `interrupt()`?
   <details><summary>Answer</summary>

   A separate node makes the pause point explicit in the graph and checkpointable: the run stops there, state is saved, and it can be resumed later with the human's answer. On resume, LangGraph re-runs the interrupted node from the start, so any code before `interrupt()` executes twice; side effects placed there would happen twice.

   </details>

4. Why does the OpenAI Agents SDK example call `set_tracing_disabled(True)` and use `OpenAIChatCompletionsModel`?
   <details><summary>Answer</summary>

   Tracing defaults to exporting to OpenAI's platform, which needs an OpenAI API key and would try to send your run data off the machine. `OpenAIChatCompletionsModel` uses the Chat Completions API, which LM Studio serves; the SDK's default path is OpenAI's Responses API, which many non-OpenAI servers don't fully support.

   </details>

5. Your capstone needs a scope guard, a per-host rate limiter and a human approval step before any active test. Own loop or framework?
   <details><summary>Answer</summary>

   Either can work, but the guardrails must be code you control and can audit, in the tool path, not inside a framework's prompt. The recommended start is your own loop with those controls in the tools, plus validated outputs. If you later need durable pause-and-resume around approvals, LangGraph's interrupts and checkpointers are a good fit, with the same guard functions called inside its tool nodes.

   </details>

## Go deeper

- [smolagents documentation](https://huggingface.co/docs/smolagents/index) — guided tour, model classes, and the secure code execution guide
- [PydanticAI: output](https://pydantic.dev/docs/ai/core-concepts/output/) — tool, native and prompted output modes and validators
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graphs, persistence and interrupts
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — agents, tools, hand-offs, guardrails, and the models page for non-OpenAI providers
- [LM Studio Python SDK](https://lmstudio.ai/docs/python) — `.act()`, tool definitions and callbacks
- [Model Context Protocol](https://modelcontextprotocol.io) — the specification and server ecosystem
