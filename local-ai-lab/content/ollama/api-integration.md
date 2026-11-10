# API, integrations and securing Ollama

Everything you've done with Ollama so far went through its HTTP API, even when you typed into `ollama run`. This lesson makes that API yours: the native endpoints with curl, the official Python library, the OpenAI-compatible endpoint that lets every script in this course switch from LM Studio with two config changes, structured outputs, tool calls, embeddings, and Open WebUI. Then you'll look at it the way an attacker would, because an Ollama port without authentication is one of the most commonly exposed AI services on the internet.

> [!GOALS]
> - Call Ollama's native API (`/api/chat`, `/api/generate`, `/api/embed`, `/api/ps`, `/api/show`, `/api/pull`) and read its responses
> - Use the official `ollama` Python library for streaming, thinking, structured output, tools and embeddings
> - Port the course's OpenAI-client scripts to Ollama, knowing exactly which parameters don't carry over
> - Connect Open WebUI to Ollama
> - Explain Ollama's security model, real CVEs and real-world exposure, and harden your own setup

## Two APIs on one port

Ollama serves several API styles on `http://localhost:11434`:

| API | Paths | Use it when |
|---|---|---|
| **Native** | `/api/...` | You want every Ollama feature: `options` (any parameter, including `num_ctx`, `top_k`, `min_p`), `think`, `keep_alive`, model management |
| **OpenAI-compatible** | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses` | Existing code and tools that speak OpenAI, including every script in this course |
| **Anthropic-compatible** | `/v1/messages` | Tools built for the Anthropic Messages API (see the docs if you need it) |

The native API predates the others and is the more complete one. The compatible endpoints are translation layers on top of it.

> [!CONCEPT] NDJSON streaming
> Ollama's native endpoints stream by default as **NDJSON** (newline-delimited JSON): one complete JSON object per line, each carrying a piece of the answer, with a final object (`"done": true`) that carries statistics. The OpenAI-compatible endpoints use the OpenAI style instead (Server-Sent Events, `data: {...}` lines). Add `"stream": false` to any native call to get a single JSON response.

## The native API with curl

### Endpoint map

| Method and path | Purpose |
|---|---|
| `POST /api/chat` | Chat completion from a list of messages (supports `tools`, `think`, `format`) |
| `POST /api/generate` | Completion from a single prompt (plus `system`, `suffix`, `raw`) |
| `POST /api/embed` | Embeddings for one string or a list (replaces the older `/api/embeddings`) |
| `GET /api/tags` | Models on disk (what `ollama list` shows) |
| `GET /api/ps` | Models in memory (what `ollama ps` shows), with `size_vram`, `context_length`, `expires_at` |
| `POST /api/show` | Model details: parameters, template, capabilities, `thinking` controls, renderer and template source |
| `POST /api/pull`, `POST /api/push` | Download from or upload to a registry |
| `POST /api/create`, `POST /api/copy`, `DELETE /api/delete` | Build, copy and delete models |
| `GET /api/version` | Server version |

### A chat call, explained

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen-sec",
  "messages": [{"role": "user", "content": "One-line definition of SSRF."}],
  "stream": false,
  "think": false,
  "keep_alive": "10m",
  "options": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "num_ctx": 16384}
}' | python3 -m json.tool
```

| Field | Meaning |
|---|---|
| `model` | Any local model name, including your [Modelfile](#/ollama/modelfile) variants |
| `messages` | Roles `system`, `user`, `assistant`, `tool`. A `system` message here replaces the Modelfile's `SYSTEM` |
| `think` | `true`, `false`, or a level string for models that define levels; omit it for the model's default |
| `keep_alive` | How long to keep the model loaded after this request (`"10m"`, seconds, `-1` forever, `0` unload now) |
| `options` | Any Modelfile parameter for this request only; it beats the Modelfile value |
| `format` | `"json"` or a JSON schema (structured outputs, below) |

The response contains `message.content`, a separate `message.thinking` when thinking was on, and the same statistics `--verbose` prints, in **nanoseconds**: `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`. Decode speed is `eval_count / eval_duration × 1e9`.

Two management calls you'll use from scripts:

```bash
curl -s http://localhost:11434/api/ps | python3 -m json.tool        # what's loaded, how big, until when
curl -s http://localhost:11434/api/chat -d '{"model": "qwen-sec", "keep_alive": 0}'   # unload now
```

A request with a model and no messages just loads (or, with `keep_alive: 0`, unloads) the model. That's how agents preload before a session.

```bash
curl -s http://localhost:11434/api/pull -d '{"model": "nomic-embed-text:v1.5", "stream": false}'
```

## The official Python library

```bash
mkdir -p ~/ai-lab/ollama/api && cd ~/ai-lab/ollama/api
python3 -m venv .venv && source .venv/bin/activate
pip install ollama openai pydantic
```

The `ollama` package (version 0.6.x at the time of writing) mirrors the native API: `chat`, `generate`, `embed`, `list`, `show`, `ps`, `pull`, `create`, `copy`, `delete`, plus `Client` and `AsyncClient` for a non-default host or extra headers.

### Streaming with thinking separated

```python title="stream_think.py"
#!/usr/bin/env python3
"""Stream a reply from Ollama, printing the reasoning and the answer separately."""
from ollama import Client

client = Client(host="http://localhost:11434")


def main() -> None:
    stream = client.chat(
        model="qwen-think",
        messages=[{"role": "user", "content": "Is 'admin'--' a SQL injection payload? Explain briefly."}],
        think=True,
        stream=True,
        options={"num_predict": 1024},
        keep_alive="10m",
    )
    in_thinking = False
    for chunk in stream:
        if chunk.message.thinking:
            if not in_thinking:
                print("--- thinking ---")
                in_thinking = True
            print(chunk.message.thinking, end="", flush=True)
        if chunk.message.content:
            if in_thinking:
                print("\n--- answer ---")
                in_thinking = False
            print(chunk.message.content, end="", flush=True)
        if chunk.done:
            rate = chunk.eval_count / chunk.eval_duration * 1e9
            print(f"\n[{chunk.eval_count} tokens at {rate:.1f} tok/s]")


if __name__ == "__main__":
    main()
```

Notice there's no `<think>` regex anywhere: Ollama has already split reasoning from the answer, as long as the model's renderer and parser handle thinking (check `ollama show`, as in [the Modelfile lesson](#/ollama/modelfile)).

### Structured outputs

The native `format` field takes a JSON schema and constrains generation to it, the same grammar-constrained decoding idea taught in [Structured output and tool calling](#/using/structured-output-tools). Ollama's docs recommend also describing the schema in the prompt.

```python title="structured.py"
#!/usr/bin/env python3
"""Extract a validated finding from free text with Ollama's native format= schema."""
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field


class Finding(BaseModel):
    title: str
    severity: Literal["info", "low", "medium", "high", "critical"]
    cwe: str = Field(description="CWE id like CWE-79, or 'unknown'")
    evidence: str = Field(description="Exact text from the note that supports the finding")


NOTE = "Lab note: /search?q= reflects input unencoded inside a <script> block; alert(1) fired in Firefox."


def main() -> None:
    schema = Finding.model_json_schema()
    resp = chat(
        model="qwen-hauhau",
        messages=[{"role": "user",
                   "content": f"Turn this note into a finding. Reply as JSON matching: {schema}\n\n{NOTE}"}],
        format=schema,          # constrained decoding to this schema
        think=False,
        options={"temperature": 0.2},
    )
    finding = Finding.model_validate_json(resp.message.content)
    assert finding.evidence in NOTE, "evidence must be copied from the input"
    print(finding.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

The schema guarantees shape, Pydantic guarantees types, and your `assert` checks truth against the input. Same layering as the LM Studio version.

### Tool calling

The library can take plain Python functions as tools: it builds the JSON schema from the signature and docstring.

```python title="tools.py"
#!/usr/bin/env python3
"""Minimal tool-calling loop with the ollama library. The tool is local and read-only."""
import socket

from ollama import chat


def resolve_host(hostname: str) -> str:
    """Resolve a hostname to its IPv4 addresses.

    Args:
      hostname: The DNS name to resolve, e.g. example.com
    """
    try:
        return ", ".join(sorted({ai[4][0] for ai in socket.getaddrinfo(hostname, None, socket.AF_INET)}))
    except socket.gaierror as exc:
        return f"error: {exc}"


TOOLS = {"resolve_host": resolve_host}


def main() -> None:
    messages = [{"role": "user", "content": "What IPv4 addresses does example.com resolve to?"}]
    resp = chat(model="qwen-hauhau", messages=messages, tools=[resolve_host], think=False)
    messages.append(resp.message)
    for call in resp.message.tool_calls or []:
        result = TOOLS[call.function.name](**call.function.arguments)
        print(f"tool {call.function.name}({call.function.arguments}) -> {result}")
        messages.append({"role": "tool", "tool_name": call.function.name, "content": result})
    final = chat(model="qwen-hauhau", messages=messages, tools=[resolve_host], think=False)
    print(final.message.content)


if __name__ == "__main__":
    main()
```

If `resp.message.tool_calls` is always empty, check `ollama show qwen-hauhau` for the `tools` capability; tool calling depends on the template (or renderer and parser) understanding tools. Everything from [the agent lessons](#/agents/agent-from-scratch) about validating arguments and allow-listing tools applies unchanged.

## Embeddings for RAG

Pull a dedicated embedding model; your chat model is the wrong tool for this ([RAG](#/using/rag)). Options in the Ollama library, with listed download sizes:

| Model tag | Size | Notes |
|---|---|---|
| `nomic-embed-text:v1.5` (= `latest`) | 274 MB | Same model LM Studio ships for RAG; 768 dimensions; needs the `search_document: ` / `search_query: ` prefixes. Ollama lists a 2K context window |
| `embeddinggemma:300m` | 622 MB (QAT q8_0 and q4_0 variants are smaller) | Google's 300M embedding model; Ollama's docs use it in examples |
| `qwen3-embedding:0.6b` | 639 MB | Multilingual, long context (32K). Careful: `qwen3-embedding:latest` is the **8B** (4.7 GB) |

```bash
ollama pull nomic-embed-text:v1.5
curl -s http://localhost:11434/api/embed -d '{"model": "nomic-embed-text:v1.5", "input": ["search_query: what is SSRF?"]}' | head -c 300
```

`/api/embed` returns L2-normalized vectors (so cosine similarity is a dot product) and truncates over-long inputs by default (`"truncate": false` makes it an error instead). The OpenAI-style `/v1/embeddings` works too, which is what the course's RAG scripts use.

> [!WARNING]
> Vectors from Ollama's `nomic-embed-text` and LM Studio's build of the same model may differ slightly (different conversion, quantization or engine). Don't mix them in one index: when you switch backends, rebuild the index (the course's `ingest.py --rebuild`).

## Porting the course's scripts to Ollama

Every script in this course uses the official `openai` client pointed at LM Studio. Moving one to Ollama is mostly two lines:

```python
BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen-hauhau"     # check yours: ollama list, or curl http://localhost:11434/v1/models
```

The API key is required by the `openai` client but ignored by Ollama; any string works (`"ollama"` by convention).

What does not carry over, based on Ollama's OpenAI-compatibility docs for `/v1/chat/completions`:

| In the LM Studio version | On Ollama `/v1` | What to do |
|---|---|---|
| `extra_body={"top_k": 20}`, `repeat_penalty`, `min_p` | Not accepted | Bake into a Modelfile `PARAMETER` |
| Context length from LM Studio's model settings | `/v1` has no way to set `num_ctx` | Bake `PARAMETER num_ctx` (the docs say exactly this) |
| `tool_choice`, `logit_bias`, `n` | Not supported | Drop them; enforce tool use in your loop |
| Reasoning in `delta.reasoning_content` | Arrives in `reasoning` (non-standard field) | Read both, like the [benchmark script](#/ollama/tuning) |
| `<think>` stripping regex | Usually unnecessary (reasoning is separated) | Harmless to keep |
| `ttl` (LM Studio idle unload) | Use `keep_alive` | The source accepts `keep_alive` on `/v1` as an Ollama extension; the native API documents it |
| Thinking toggle | `reasoning_effort` (or `reasoning.effort`) | `"none"` requests no thinking on models with an on/off switch; `/api/show` lists what a model supports |
| Model id `qwen3.5-9b-uncensored-hauhaucs-aggressive` | Ollama names: `qwen-hauhau`, `qwen-sec:latest` | Use `ollama list` names |

Supported on `/v1/chat/completions`: `messages` (text and base64 images), `temperature`, `top_p`, `max_tokens`, `seed`, `stop`, `stream`, `stream_options.include_usage`, `presence_penalty`, `frequency_penalty`, `response_format` (JSON mode and JSON schema), `tools`, and the reasoning controls. That covers the course's structured-output and RAG scripts.

The cleanest pattern is to read the backend from the environment, so one file runs against either server:

```python
import os

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b-uncensored-hauhaucs-aggressive")
API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
```

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen-hauhau LLM_API_KEY=ollama python3 scan_to_json.py scan.txt
```

> [!TIP]
> Many agent frameworks and tools also have a native Ollama integration (they talk to `/api/chat` and accept `num_ctx` and `keep_alive`). When given a choice, the native provider usually exposes more knobs; the OpenAI-compatible one is more portable. The [agent frameworks lesson](#/agents/frameworks) applies either way.

## Open WebUI with Ollama

Open WebUI speaks the native Ollama API directly, so it can also list, pull and delete Ollama models from its admin panel. The [offline stack lesson](#/infra/offline-stack) runs it against LM Studio; against Ollama the key variables (from Open WebUI's environment reference) are:

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Reach Ollama on the Mac from the container (this is also Open WebUI's Docker default) |
| `ENABLE_OLLAMA_API` | `true` (default) | Use the native Ollama connection |
| `RAG_EMBEDDING_ENGINE` | `ollama` | Embeddings from Ollama |
| `RAG_OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Where to send embedding requests |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text:v1.5` | Must be pulled in Ollama first |

```bash
docker run -d --name open-webui -p 127.0.0.1:3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e RAG_EMBEDDING_ENGINE=ollama \
  -e RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e RAG_EMBEDDING_MODEL=nomic-embed-text:v1.5 \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

Note `127.0.0.1:3000:8080`: the UI is published on your Mac's loopback only. Ollama itself stays on its default `127.0.0.1` binding; Docker Desktop's `host.docker.internal` reaches it without exposing it to your network.

> [!WARNING] Open WebUI can override your context size
> Open WebUI's docs warn that its per-model `num_ctx` setting, when set, is sent with every request and overrides `OLLAMA_CONTEXT_LENGTH`, and that the control pre-fills a small 2048. Leave it unset if your models have `num_ctx` baked in, or set it deliberately. Also remember the `RAG_*` variables are persistent config: change them in the admin panel after first launch.

## Securing Ollama

### The security model in one sentence

The docs state it plainly: "The local API at `http://localhost:11434` does not require authentication." Anyone who can reach the port can use every endpoint. The only built-in protection is the default bind address, `127.0.0.1`.

That's a reasonable design for a single-user laptop, the same model as many local developer services. It fails the moment the port becomes reachable by someone else, and there are several ways that happens:

- Setting `OLLAMA_HOST=0.0.0.0` (or flipping the app's **Expose Ollama to the network** toggle, which does exactly that) on a laptop that joins hotel, conference or client Wi-Fi.
- Running the `ollama/ollama` Docker image with `-p 11434:11434`, which publishes on all interfaces.
- Exposing it through a tunnel (the FAQ documents ngrok and Cloudflare Tunnel) without adding authentication.
- Even on localhost: any local process can call it (a malicious npm postinstall script, a compromised editor extension, a rogue MCP server), and a web page can reach it if CORS is opened with `OLLAMA_ORIGINS=*`.

### What an attacker gets from an open port

Think of 11434 like an unauthenticated Docker socket for models: less dangerous than root on the host, much more dangerous than "someone can chat with my bot".

| Endpoint | Abuse |
|---|---|
| `/api/chat`, `/api/generate`, `/v1/*` | Free compute on your machine; reading your system prompts (`/api/show`); prompt-injecting any agent that trusts the model's output |
| `/api/tags`, `/api/show` | Inventory: which models, which custom personas, which `SYSTEM` prompts (possibly with client names or internal details) |
| `/api/pull` | Fill your disk; pull a model from a malicious registry (this was the entry point of CVE-2024-37032) |
| `/api/create`, `/api/push`, `/api/delete` | Plant or replace models, exfiltrate custom models, delete your work |
| Cloud models | If you ran `ollama signin`, the docs show cloud models working "through the local API", so an attacker could spend your cloud usage. `OLLAMA_NO_CLOUD=1` closes this |

### It happens at scale

This isn't hypothetical. Independent internet-wide measurements:

- **Wiz (June 2024)** found over 1,000 exposed Ollama instances while researching Probllama, including private models not in the public library.
- **Cisco Talos (reported September 2025)** used Shodan to find more than 1,100 exposed Ollama servers, roughly 20% of them actively serving models to anyone.
- **SentinelOne and Censys (reported in early 2026)** counted about 175,000 publicly reachable Ollama hosts across 130 countries in a longer study; secondary coverage differs on method details, so treat the exact figure with care.

Different methods, different numbers, one conclusion: "I'll just bind it to 0.0.0.0 for a minute" is how most of these happen.

### Real CVEs worth knowing

All verified against NVD (scores as listed there). Each is fixed; the point is the pattern.

| CVE | Fixed in | What it was |
|---|---|---|
| **CVE-2024-37032** "Probllama" (CVSS 8.8) | 0.1.34 | `/api/pull` did not validate a manifest's `digest` format, so a malicious registry could use path traversal (`../`) for **arbitrary file write**. Wiz showed **remote code execution** on Docker deployments, where the server ran as root and listened on 0.0.0.0, by overwriting `/etc/ld.so.preload`. Reported May 5, 2024, fixed within hours, released May 8 |
| **CVE-2024-28224** (6.6) | 0.1.29 | **DNS rebinding**: a malicious web page could reach the API on localhost through the victim's browser, allowing chat, model deletion or resource exhaustion. "Bound to localhost" was not enough |
| **CVE-2024-39720** (8.2) | 0.1.46 | A 4-byte malformed GGUF uploaded as a blob and referenced from a Modelfile `FROM` crashed the server via `/api/create` (denial of service) |
| **CVE-2024-39722** (7.5) | 0.1.46 | Path traversal in `/api/push` revealed which files exist on the server |
| **CVE-2025-51471** (6.9) | see advisory | Ollama 0.6.7: a malicious `realm` value in a registry's `WWW-Authenticate` header during `/api/pull` could leak authentication tokens to an attacker-controlled host |

The lessons map directly onto your own [AI threat model](#/security/ai-threat-model): the model registry is a supply chain, model files are attacker-controllable input to a parser, and the API is an unauthenticated control plane.

### Hardening checklist

- [ ] **Keep the default bind.** `OLLAMA_HOST` unset (127.0.0.1) and the app's "Expose Ollama to the network" off. Check what's listening:

```bash
lsof -nP -iTCP:11434 -sTCP:LISTEN
```

```text
COMMAND   PID  USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
ollama   4242  sethu  3u   IPv4  ...         0t0  TCP 127.0.0.1:11434 (LISTEN)
```

`127.0.0.1:11434` is good; `*:11434` means every interface.

- [ ] **Remote access via SSH, not exposure.** From another machine you own: `ssh -N -L 11434:127.0.0.1:11434 you@your-mac`, then use `http://localhost:11434` there. Zero ports opened.
- [ ] **If several people need it, put an authenticating reverse proxy in front** and keep Ollama on loopback. The FAQ's nginx example rewrites the `Host` header; add a bearer-token check so OpenAI clients authenticate with their normal `api_key`:

```text title="ollama-proxy.conf (nginx, inside the http block)"
map $http_authorization $ollama_authorized {
    default                                  0;
    "Bearer REPLACE_WITH_A_LONG_RANDOM_TOKEN" 1;
}
server {
    listen 8443 ssl;
    server_name ollama.lab.internal;
    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    location / {
        if ($ollama_authorized = 0) { return 401; }
        proxy_pass http://127.0.0.1:11434;
        proxy_set_header Host localhost:11434;
        proxy_buffering off;   # let streamed tokens through immediately
    }
}
```

Generate the token with `openssl rand -hex 32`, keep it out of git, and consider allow-listing only `/v1/` and `/api/chat` paths so clients can't reach `/api/pull`, `/api/create` or `/api/delete`.

- [ ] **Firewall as a backstop.** Turn on the macOS firewall (System Settings, Network, Firewall). Optional: block incoming connections for the `ollama` binary. Hardening at the network layer is covered in [Hardening and air-gapping](#/infra/hardening).
- [ ] **Docker: publish to loopback.** If you ever run the container on Linux, use `-p 127.0.0.1:11434:11434`, never `-p 11434:11434`.
- [ ] **Keep Ollama updated.** Every CVE above was fixed quickly; the exposed servers in the studies were mostly old versions. The Mac app auto-downloads updates; restart to apply.
- [ ] **Disable cloud if you don't use it.** `OLLAMA_NO_CLOUD=1`, or `"disable_ollama_cloud": true` in `~/.ollama/server.json`.
- [ ] **Keep CORS tight.** Leave `OLLAMA_ORIGINS` at its default; add specific origins only, never `*`.
- [ ] **Treat models as untrusted input.** Pull from publishers you trust, review `ollama show --modelfile` for `SYSTEM` and `TEMPLATE` content, and remember GGUF parsing bugs are a real class.
- [ ] **Don't log prompts by accident.** `OLLAMA_DEBUG_LOG_REQUESTS` writes request bodies to disk: debug only, then delete.

> [!NOTE]
> Ollama's source contains an undocumented `OLLAMA_AUTH` variable described as "authentication between the Ollama client and server". It is not part of the documented configuration and is not an access-control feature you can rely on for exposing the API. Use the proxy pattern above.

> [!DANGER]
> Scanning for exposed Ollama servers is easy (it's one HTTP GET), and using one you find is unauthorized access to someone else's system, whatever the lack of a password suggests. Test exposure only on hosts you own or are authorized to assess. If you find one in a bug bounty target, check the program's scope and rules on infrastructure findings, stop at proof of exposure (for example `/api/version` or `/api/tags`), and don't pull, create, delete or run models on it.

## Lab: port the structured-output script and compare

> [!LAB] Same script, two backends, one eval
> Goal: run the course's `scan_to_json.py` ([Structured output and tool calling](#/using/structured-output-tools)) against Ollama with minimal changes, then measure both backends with the same small eval.
>
> 1. Copy the script and the lab scan:
>
> ```bash
> mkdir -p ~/ai-lab/ollama/port && cd ~/ai-lab/ollama/port
> cp ~/ai-lab/tools/scan_to_json.py .
> cp ~/ai-lab/prompts/tasks/nmap-summary/cases/lab-web.txt scan.txt
> python3 -m venv .venv && source .venv/bin/activate && pip install openai pydantic
> ```
>
> 2. Apply the environment-driven config (the diff below): base URL, model and key from the environment, and an optional switch to disable thinking on Ollama.
> 3. Save `eval_scan.py` (below) next to it.
> 4. **LM Studio first.** Load your model in LM Studio, start its server, quit Ollama, and run:
>
> ```bash
> python3 eval_scan.py scan.txt --runs 10
> ```
>
> 5. **Then Ollama.** Quit LM Studio, start Ollama, and run the same eval against the base model you imported (no `SYSTEM` prompt, so the comparison is fair):
>
> ```bash
> LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen-hauhau LLM_API_KEY=ollama LLM_NO_THINK=1 \
>   python3 eval_scan.py scan.txt --runs 10
> ```
>
> 6. Compare valid rate, first-try rate, number of distinct port sets (consistency) and median time. Record them in `~/ai-lab/ollama/NOTES.md` next to your benchmark.
> 7. **Stretch: RAG.** Port `~/ai-lab/rag/ingest.py` and `ask.py` the same way with `EMBED_MODEL = "nomic-embed-text:v1.5"` (keep the `search_document: ` / `search_query: ` prefixes), rebuild the index with `python3 ingest.py --rebuild`, and run `eval_retrieval.py`. Did hit rate change? If it did, is that the embedding build, or the different 2K context window truncating long chunks?

```diff title="scan_to_json.py (changes only)"
 import json
+import os
 import re
 import sys
@@
-BASE_URL = "http://localhost:1234/v1"
-MODEL = "qwen3.5-9b-uncensored-hauhaucs-aggressive"  # check yours: `lms ls` or curl http://localhost:1234/v1/models
+BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
+MODEL = os.environ.get("LLM_MODEL", "qwen3.5-9b-uncensored-hauhaucs-aggressive")
+API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
+# Ollama: reasoning_effort "none" asks a thinking model not to think (see /api/show for support)
+EXTRA_BODY = {"reasoning_effort": "none"} if os.environ.get("LLM_NO_THINK") == "1" else None
@@
-    client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
+    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
@@
             response_format={"type": "json_schema",
                              "json_schema": {"name": "scan_summary", "strict": True, "schema": schema}},
+            extra_body=EXTRA_BODY,
         )
```

```python title="eval_scan.py"
#!/usr/bin/env python3
"""Run scan_to_json.summarize() N times against the backend the LLM_* variables point at."""
import argparse
import contextlib
import io
import statistics
import time

import scan_to_json  # reads LLM_BASE_URL / LLM_MODEL / LLM_API_KEY at import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="nmap output file")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    scan = open(args.scan, encoding="utf-8").read()

    valid = first_try = 0
    times: list[float] = []
    port_sets: set[tuple[int, ...]] = set()
    for i in range(1, args.runs + 1):
        log = io.StringIO()
        start = time.perf_counter()
        try:
            with contextlib.redirect_stderr(log):  # summarize() reports rejected attempts on stderr
                summary = scan_to_json.summarize(scan)
            valid += 1
            first_try += "rejected" not in log.getvalue()
            port_sets.add(tuple(sorted(s.port for s in summary.services)))
            status = "ok"
        except RuntimeError:
            status = "FAILED"
        times.append(time.perf_counter() - start)
        print(f"run {i}: {status} in {times[-1]:.1f}s")

    print(f"\nbackend {scan_to_json.BASE_URL}  model {scan_to_json.MODEL}")
    print(f"valid {valid}/{args.runs}  first-try {first_try}/{args.runs}  "
          f"distinct port sets {len(port_sets)}  median time {statistics.median(times):.1f}s")


if __name__ == "__main__":
    main()
```

What to expect: with the same weights and grammar-constrained output, valid rates should be similar. Differences usually come from the template (check `ollama show`), thinking being on in one backend and off in the other, or sampling defaults the script doesn't send (`top_k`, `min_p`: baked in Ollama, per-model settings in LM Studio). If one backend is clearly worse, find which of those differs before blaming the tool.

## Recap

- Ollama serves a native API (`/api/*`, NDJSON streaming, full `options`, `think`, `keep_alive`) and OpenAI- and Anthropic-compatible layers on the same port.
- The `ollama` Python library mirrors the native API and separates `thinking` from `content`; `format=` takes a JSON schema, and `tools=` takes Python functions.
- Porting an OpenAI-client script: change the base URL to `http://localhost:11434/v1` and the model to an Ollama name. Bake `num_ctx`, `top_k`, `min_p` and repeat penalty into a Modelfile, since `/v1` can't send them; drop `tool_choice`; read reasoning from `reasoning`.
- Embeddings: `nomic-embed-text:v1.5`, `embeddinggemma:300m` or `qwen3-embedding:0.6b` (not `latest`, which is 8B). Rebuild indexes when switching backends.
- Open WebUI connects natively via `OLLAMA_BASE_URL`; watch its `num_ctx` override.
- The local API has no authentication. Exposure (0.0.0.0, Docker `-p`, tunnels, open CORS) is common in the wild, and real CVEs (Probllama, DNS rebinding, blob and push bugs, token leakage) show the API and registry are attack surface. Bind to loopback, use SSH or an authenticating proxy, stay updated, disable cloud if unused, and treat models as untrusted input.

## Check yourself

1. Your script uses the OpenAI client against `http://localhost:11434/v1` with `extra_body={"top_k": 20, "min_p": 0}` and outputs feel unchanged. Why, and what's the fix?
   <details><summary>Answer</summary>

   Ollama's OpenAI-compatible endpoint doesn't accept `top_k` or `min_p` (they aren't in its supported request fields), so they have no effect. Bake them into the model with `PARAMETER top_k 20` and `PARAMETER min_p 0` in a Modelfile, or use the native `/api/chat` with `options`.

   </details>
2. A teammate says "Ollama only listens on localhost, so it's safe on the office network." Give two ways it can still be reached by someone else.
   <details><summary>Answer</summary>

   Any of: the app's "Expose Ollama to the network" toggle or `OLLAMA_HOST=0.0.0.0` changes the bind; a Docker container published with `-p 11434:11434` listens on all interfaces; a tunnel (ngrok, Cloudflare) forwards it; a browser-based attack like DNS rebinding (CVE-2024-28224, fixed in 0.1.29) or overly broad `OLLAMA_ORIGINS` lets web pages reach it; any local process can call it without credentials.

   </details>
3. Why did Probllama (CVE-2024-37032) lead to remote code execution on Docker deployments specifically?
   <details><summary>Answer</summary>

   The bug allowed arbitrary file writes via path traversal in a malicious manifest's digest during `/api/pull`. In the Docker image the server ran as root and listened on 0.0.0.0, so a remote attacker could reach the API and overwrite a privileged file (Wiz used `/etc/ld.so.preload`) to load their own library, turning a file write into code execution.

   </details>
4. You switch your RAG pipeline from LM Studio to Ollama, keep the same Chroma index, and retrieval quality drops. What's the most likely cause?
   <details><summary>Answer</summary>

   The index was built with LM Studio's embedding build and queries are now embedded by Ollama's. Even for the "same" model, different conversions, quantization or engines can produce slightly different vectors, and mixing them hurts similarity. Rebuild the index with the new backend. Also check prefixes and Ollama's listed 2K context for `nomic-embed-text`, which truncates longer chunks by default.

   </details>
5. You want colleagues on your lab network to use your Ollama with their OpenAI-client scripts. Describe a setup that doesn't expose the unauthenticated API.
   <details><summary>Answer</summary>

   Keep Ollama bound to `127.0.0.1`. Put a reverse proxy (for example nginx with TLS) on the network-facing port that requires a bearer token and forwards to `127.0.0.1:11434` with the `Host` header rewritten. Clients set `base_url` to the proxy and `api_key` to the token. Optionally allow only `/v1/` and `/api/chat` paths so management endpoints (`pull`, `create`, `delete`) stay unreachable. For a single person, an SSH tunnel is simpler.

   </details>

## Go deeper
- [Ollama API reference (GitHub)](https://github.com/ollama/ollama/blob/main/docs/api.md) — every native endpoint, parameter and response field, with curl examples.
- [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) — the exact list of supported and unsupported fields per endpoint.
- [ollama-python](https://github.com/ollama/ollama-python) — the official Python library, with examples for streaming, tools and structured outputs.
- [Probllama: Ollama RCE (Wiz)](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032) — the full write-up of CVE-2024-37032, a model case of supply-chain-to-RCE.
- [More Models, More ProbLLMs (Oligo)](https://www.oligo.security/blog/more-models-more-probllms) — the research behind the 2024 Ollama DoS and file-disclosure CVEs.
- [Exposed Ollama servers research (The Register)](https://www.theregister.com/2025/09/02/exposed_ollama_servers_insecure_research/) — coverage of Cisco Talos's internet-wide measurement.
