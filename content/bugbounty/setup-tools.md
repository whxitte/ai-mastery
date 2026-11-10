# Toolchain and a safe practice lab

This lesson installs everything the hunting agent drives (the ProjectDiscovery recon tools, ffuf, jq, a Python environment) and sets up a target you are allowed to hit as hard as you like: OWASP Juice Shop in Docker, bound to your own laptop. It ends with a lab that runs every tool against `localhost` so you know the whole chain works before you write any agent code.

> [!GOALS]
> - Install subfinder, dnsx, httpx, katana, nuclei, ffuf and jq on macOS, and know the three install routes (Homebrew, `go install`, pdtm)
> - Avoid the `httpx` name clash between the Go recon tool and the Python HTTP library
> - Create the Python environment for the project
> - Pick a Docker runtime and size it so Docker, Burp and a 5–7 GB model fit in 16 GB
> - Run Juice Shop locally and verify every tool against it

## The toolchain at a glance

| Tool | What it does in this project | Talks to the target? |
|---|---|---|
| `subfinder` | Passive subdomain enumeration from third-party sources (certificate logs, DNS datasets) | No, it queries the sources |
| `dnsx` | Resolves names to IPs, so the Scope Guard can check IP ranges | Only DNS resolvers |
| `httpx` (ProjectDiscovery) | Probes which hosts serve HTTP(S); title, status, tech, server | Yes, light |
| `katana` | Crawls live sites, parses JavaScript for endpoints | Yes, moderate |
| `nuclei` | Template-based scanner (misconfigurations, exposures, CVEs) | Yes, noisy: approval required |
| `ffuf` | Content discovery with a wordlist | Yes, noisy: approval required |
| `jq` | Slicing JSON output in the terminal | No |

You know most of these from manual recon. The difference here is that Python will call them, so we care about exact flags, JSON output, and making sure the right binary runs.

## Homebrew first

Check whether Homebrew is installed:

```bash
brew --version
```

If not, install it with the official one-liner from [brew.sh](https://brew.sh), then follow the "Next steps" it prints (on Apple Silicon it adds `/opt/homebrew/bin` to your `PATH` via `~/.zprofile`):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Install the recon tools: pick one route

There are three ways to install the ProjectDiscovery tools. They all work; the important thing is to **pick one**, so you do not end up with two versions of `nuclei` on your `PATH` and wonder which one ran.

| Route | Command | Updates | Best for |
|---|---|---|---|
| A. Homebrew (recommended) | `brew install subfinder dnsx httpx katana nuclei` | `brew upgrade` | Simplest; prebuilt binaries for Apple Silicon |
| B. `go install` | One command per tool (below) | Re-run the command | Latest commit-level control, building from source |
| C. pdtm | `pdtm -ia` | `pdtm -ua` | Managing the whole ProjectDiscovery suite |

### Route A: Homebrew

All five tools, ffuf and jq have Homebrew formulae:

```bash
brew install subfinder dnsx httpx katana nuclei ffuf jq
```

### Route B: Go and `go install`

Install Go and put Go-installed binaries on your `PATH`:

```bash
brew install go
echo 'export PATH="$PATH:$(go env GOPATH)/bin"' >> ~/.zshrc
source ~/.zshrc
```

Then install each tool from its official module path:

```bash
go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
go install -v github.com/projectdiscovery/dnsx/cmd/dnsx@latest
go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest
CGO_ENABLED=1 go install github.com/projectdiscovery/katana/cmd/katana@latest
go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
go install github.com/ffuf/ffuf/v2@latest
brew install jq
```

> [!WARNING] Go version requirements move
> Each README states a minimum Go version (the line "requires go >= ..."), and ProjectDiscovery raises it regularly. Homebrew's `go` is current, so it normally satisfies them. If a build fails with a version error, run `brew upgrade go` and retry. katana's README asks for `CGO_ENABLED=1`, which is why that line differs.

### Route C: pdtm

pdtm is ProjectDiscovery's tool manager. It downloads release binaries into `~/.pdtm/go/bin` and adds that folder to your `PATH`:

```bash
brew install pdtm            # or: go install -v github.com/projectdiscovery/pdtm/cmd/pdtm@latest
pdtm -ia                     # install all ProjectDiscovery tools
pdtm -sp                     # show where it put them
brew install ffuf jq         # not ProjectDiscovery tools
```

Open a new terminal afterwards so the `PATH` change takes effect. Later, `pdtm -ua` updates everything.

### Confirm what is on your PATH

```bash
which -a subfinder dnsx httpx katana nuclei ffuf jq
```

Each tool should appear once. If one appears twice (for example `/opt/homebrew/bin/nuclei` and `~/go/bin/nuclei`), remove one install.

### First-run housekeeping

nuclei needs its community template library. It fetches it automatically on first use, but do it explicitly now so the first scan is not slowed down:

```bash
nuclei -ut
```

subfinder works without configuration, but many passive sources only answer with an API key. Keys go in `~/.config/subfinder/provider-config.yaml` (the default path; see `subfinder -h` for `-pc`). You do not need any keys for the Juice Shop lab.

> [!NOTE] Tools that phone home
> ProjectDiscovery tools check for updates on each run; every tool accepts `-duc` (`-disable-update-check`) to skip that, and our wrappers pass it. nuclei also uses public interactsh servers for out-of-band (OAST) templates; `-ni` (`-no-interactsh`) turns that off. For a private setup, both matter.

## The `httpx` name clash

There are two different programs called `httpx`:

- **ProjectDiscovery httpx**: the Go HTTP prober you just installed.
- **Python httpx**: an HTTP client library. Installing it also installs a small command-line script called `httpx`.

You will never install Python httpx on purpose, but the `openai` Python package depends on it. So the moment you create the project's virtual environment and `pip install openai`, a file `.venv/bin/httpx` appears. When the venv is active, `.venv/bin` is first on your `PATH`, and typing `httpx` runs the Python one, which does not understand `-sc` or `-title`.

```bash
cd ~/ai-lab/bugbounty && source .venv/bin/activate
which -a httpx
```

```text
/Users/you/ai-lab/bugbounty/.venv/bin/httpx
/opt/homebrew/bin/httpx
```

Three ways to deal with it:

1. In the terminal, call the Go binary by full path (`/opt/homebrew/bin/httpx`, `~/go/bin/httpx` or `~/.pdtm/go/bin/httpx`) or define an alias such as `alias pdhttpx=/opt/homebrew/bin/httpx`.
2. In code, never trust `PATH` order. The `tools.py` module below skips the venv and any file that is a Python script.
3. Pin explicit paths with environment variables (`BB_HTTPX_PATH=/opt/homebrew/bin/httpx`), which `tools.py` honours.

## The Python project

Create the folder, a virtual environment and the dependencies the whole capstone uses:

```bash
mkdir -p ~/ai-lab/bugbounty && cd ~/ai-lab/bugbounty
python3 -m venv .venv
source .venv/bin/activate
```

```text title="requirements.txt"
openai>=1.40
pydantic>=2.7
pyyaml>=6.0
requests>=2.32
jsbeautifier>=1.15
cvss>=3.0
pytest>=8.0
```

```bash
pip install -r requirements.txt
```

> [!TIP] uv is optional
> If you use [uv](https://docs.astral.sh/uv/), `uv venv && uv pip install -r requirements.txt` does the same thing faster. Nothing in this module requires it.

What each dependency is for:

| Package | Used for |
|---|---|
| `openai` | Talking to LM Studio's OpenAI-compatible server |
| `pydantic` | Defining output schemas and validating the model's JSON |
| `pyyaml` | Reading `scope.yaml` |
| `requests` | Our own scope-checked HTTP fetches (downloading JS files) |
| `jsbeautifier` | Reformatting minified JS snippets before the model reads them |
| `cvss` | Computing CVSS 3.1 and 4.0 scores from vectors in the reporting lesson |
| `pytest` | Tests for the Scope Guard, redaction and regressions |

Now the shared settings file and the tool resolver. `config.py` is read by every other module:

```python title="config.py"
"""Shared settings for the bug bounty lab."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_URL = os.environ.get("BB_BASE_URL", "http://localhost:1234/v1")
# Check yours with `lms ls` or `curl http://localhost:1234/v1/models`
MODEL = os.environ.get("BB_MODEL", "qwen3.5-9b-uncensored-hauhaucs-aggressive")

DATA_DIR = ROOT / "data"        # one subfolder per program
TRACE_DIR = ROOT / "traces"     # JSONL logs of agent runs
BRIEF_DIR = ROOT / "briefs"     # Markdown hunt briefs
REPORT_DIR = ROOT / "reports"   # your report drafts
for d in (DATA_DIR, TRACE_DIR, BRIEF_DIR, REPORT_DIR):
    d.mkdir(exist_ok=True)
```

`tools.py` finds the real binaries and runs them **without a shell**. Passing a list of arguments to `subprocess.run` means a hostile string like `; rm -rf ~` is just an argument, never a command:

```python title="tools.py"
"""Find external tools by absolute path and run them without a shell."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXTRA_DIRS = [Path("/opt/homebrew/bin"), Path.home() / "go" / "bin",
              Path.home() / ".pdtm" / "go" / "bin", Path("/usr/local/bin")]


def _is_python_script(p: Path) -> bool:
    """Console scripts installed by pip start with a python shebang; Go binaries do not."""
    try:
        with p.open("rb") as f:
            head = f.read(256)
    except OSError:
        return False
    return head.startswith(b"#!") and b"python" in head


def resolve(name: str) -> str:
    """Absolute path of a tool. Never returns the venv's Python 'httpx' script."""
    override = os.environ.get(f"BB_{name.upper()}_PATH")
    if override:
        return override
    venv_bin = Path(sys.prefix, "bin").resolve() if sys.prefix != sys.base_prefix else None
    dirs = [Path(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d] + EXTRA_DIRS
    for d in dirs:
        cand = d / name
        if not (cand.is_file() and os.access(cand, os.X_OK)):
            continue
        if venv_bin is not None and d.resolve() == venv_bin:
            continue
        if _is_python_script(cand):
            continue
        return str(cand)
    raise FileNotFoundError(f"{name}: not found. Install it or set BB_{name.upper()}_PATH")


def run(argv: list[str], *, stdin: str | None = None,
        timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a tool with an argument list (no shell). argv[0] is a tool name."""
    full = [resolve(argv[0]), *argv[1:]]
    return subprocess.run(full, input=stdin, capture_output=True, text=True,
                          timeout=timeout, check=False)
```

And a checker you can rerun any time the toolchain changes:

```python title="check_tools.py"
"""Print the resolved path and version of every external tool."""
from tools import resolve, run

VERSION_FLAG = {"ffuf": "-V", "jq": "--version"}

if __name__ == "__main__":
    for name in ["subfinder", "dnsx", "httpx", "katana", "nuclei", "ffuf", "jq"]:
        try:
            path = resolve(name)
        except FileNotFoundError as exc:
            print(f"MISSING  {exc}")
            continue
        p = run([name, VERSION_FLAG.get(name, "-version")], timeout=60)
        lines = (p.stdout + p.stderr).strip().splitlines()
        print(f"OK  {name:9} {path}  {lines[-1] if lines else ''}")
```

```bash
python check_tools.py
```

You should see seven `OK` lines, and the `httpx` line must point at the Go binary, not `.venv/bin/httpx`. ProjectDiscovery tools print their version on stderr with an `[INF]` prefix; that is normal.

## Docker on a 16 GB Mac

Juice Shop runs as a Linux container, and on macOS every container runtime runs a small Linux virtual machine. That VM takes memory from the same 16 GB unified pool your model needs, so this is a budgeting exercise.

| Runtime | Install | Memory behaviour | How to cap it |
|---|---|---|---|
| Docker Desktop | `brew install --cask docker-desktop` | VM memory limit defaults to 50% of host RAM (8 GB on your Mac) | Settings → Resources → Advanced → **Memory limit**; also enable **Resource Saver** |
| OrbStack | `brew install orbstack` | Limit defaults to at most 8 GB; memory is released when unused | Settings, or the `memory_mib` setting via `orb config` (check `orb config --help`) |
| Colima | `brew install colima docker` | Fixed VM size, default 2 CPUs and 2 GiB | `colima start --cpu 2 --memory 2 --disk 20` |

Any of them is fine. If you have none installed, Colima is the lightest and most explicit (you choose the VM size on the command line); OrbStack is the most pleasant on a laptop; Docker Desktop is what most documentation assumes.

> [!MAC] The memory budget
> Your model, Docker, Burp and macOS all share 16 GB. A realistic plan while hunting:
>
> | Consumer | Approximate memory |
> |---|---|
> | macOS, browser, editor | 4–5 GB |
> | Qwen3.5-9B Q4_K_M weights | 5.3 GB |
> | KV cache at 32K context (32 KiB per token, see [Memory math](#/foundations/memory-math)) | 1 GiB |
> | Docker VM for Juice Shop | 2 GB cap |
> | Burp Suite (Java) | 1–2 GB |
> | Recon tools (katana, nuclei while running) | a few hundred MB, more for nuclei with many templates |
>
> That lands around 14–15 GB: workable, but only if Docker is capped. Left at Docker Desktop's default (8 GB limit), the VM can grow and push macOS into swap, and generation speed collapses. Cap the VM at 2–3 GB, and set LM Studio's context to 32K for this project rather than the 128K+ the model card suggests for long thinking tasks. Our prompts are small chunks, so 32K is plenty. If memory is still tight, Q4_K_M plus 16K context or the 4B model are the fallbacks; the [evaluation lesson](#/bugbounty/evaluate) shows how to measure what you lose.

Start your runtime (for Colima: `colima start --cpu 2 --memory 2 --disk 20`) and check:

```bash
docker run --rm hello-world
```

## Run OWASP Juice Shop

The official image is `bkimminich/juice-shop`. It is published for `linux/amd64` and `linux/arm64`, so it runs natively on Apple Silicon. The app listens on port 3000 inside the container.

```bash
docker pull bkimminich/juice-shop
docker run -d --name juice-shop --rm -p 127.0.0.1:3000:3000 bkimminich/juice-shop
```

The `-p 127.0.0.1:3000:3000` part matters: it publishes the port on your loopback interface only. A plain `-p 3000:3000` would expose a deliberately vulnerable app to whatever network your laptop is on, café Wi-Fi included.

> [!DANGER] Keep the practice target private
> Never publish Juice Shop on `0.0.0.0` or forward it through a tunnel. It is built to be broken into; anyone on the network could use it against you.

Check it is up and how much memory it uses:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000/
docker stats juice-shop --no-stream
```

```text
200
```

Open `http://localhost:3000` in your browser (through Burp if you like). Restarting the container resets the app's data and challenge progress, which is useful when you want a clean baseline for an evaluation run.

## Load the model for this project

In LM Studio, load your Qwen3.5-9B model with a 32K context, then start the server. With the CLI:

```bash
lms ls
lms load qwen3.5-9b-uncensored-hauhaucs-aggressive --context-length 32768
lms server start
curl -s http://localhost:1234/v1/models | jq -r '.data[].id'
```

Use the id that `lms ls` shows for your copy of the model; if it differs from the default in `config.py`, export it: `export BB_MODEL=<your-id>`. [Local server, API and the lms CLI](#/lmstudio/server-api) covers the server in detail.

> [!TIP] Thinking off for analysis calls
> The analysis modules ask for short JSON answers under a schema. Thinking mode spends tokens before the JSON and, with schema-constrained output, can produce empty or cut-off replies. Start with thinking disabled for this project (check whether your model emits `<think>` blocks; see [System prompts, templates, thinking and presets](#/lmstudio/prompts-templates)). The [evaluation lesson](#/bugbounty/evaluate) measures whether turning it on helps.

## Verification lab

> [!LAB] Every tool against your own lab
> Run these from `~/ai-lab/bugbounty` with **the venv deactivated** (`deactivate`), so `httpx` means the Go tool. Each step touches only your laptop or, for steps 1–2, public DNS data about `example.com`, a domain reserved for documentation.
>
> 1. **subfinder** queries third-party sources, not a target. Juice Shop has no public subdomains, so verify it with the reserved domain:
>    ```bash
>    subfinder -d example.com -silent -duc | head
>    ```
>    A few names or none is fine; you are checking that it runs.
> 2. **dnsx** resolves names through public resolvers, so `localhost` is not a meaningful test; use the reserved domain again:
>    ```bash
>    echo example.com | dnsx -silent -a -resp -duc
>    ```
>    Expect one line with the name, `[A]`, and an address.
> 3. **httpx** probes the lab (`-p 3000` tells it which port to try):
>    ```bash
>    echo localhost | httpx -p 3000 -sc -title -td -server -silent -duc
>    ```
>    Expect a line starting `http://localhost:3000 [200] [OWASP Juice Shop]` followed by detected technologies.
> 4. **httpx JSON**, the format our code will read:
>    ```bash
>    echo localhost | httpx -p 3000 -sc -title -td -json -silent -duc | jq '{url, status_code, title, tech, webserver}'
>    ```
> 5. **katana** crawls with JavaScript parsing, rate-limited:
>    ```bash
>    katana -u http://localhost:3000 -d 2 -jc -rl 10 -silent -duc | head -30
>    ```
>    Look for API paths such as `/rest/...` and `/api/...` that katana pulled out of the JS bundles.
> 6. **katana JSONL** piped through jq:
>    ```bash
>    katana -u http://localhost:3000 -d 2 -jc -rl 10 -jsonl -or -ob -silent -duc | jq -r '.request.endpoint' | sort -u | head -30
>    ```
>    `-or` and `-ob` drop raw requests/responses and bodies from the JSON so the output stays small.
> 7. **nuclei** with one harmless template (technology detection), interactsh off:
>    ```bash
>    nuclei -u http://localhost:3000 -id tech-detect -rl 10 -ni -duc -silent
>    ```
> 8. **ffuf** with a tiny wordlist you create now (the agent reuses it later):
>    ```bash
>    mkdir -p wordlists
>    printf "%s\n" admin api api-docs assets backup ftp metrics robots.txt .git/HEAD .env \
>      server-status swagger.json graphql debug console uploads .well-known/security.txt > wordlists/small.txt
>    ffuf -u http://localhost:3000/FUZZ -w wordlists/small.txt -rate 10 -t 5 -ac -s
>    ```
>    Juice Shop is a single-page app, so many unknown paths come back as the same index page with status 200. `-ac` (auto-calibration) learns that baseline and filters it; try once without `-ac` to see the difference.
> 9. **The Python side**: activate the venv and run `python check_tools.py`. Confirm the `httpx` line points at the Go binary.
> 10. Note how long steps 3–8 took and how much memory LM Studio, Docker and Burp use together in Activity Monitor (Memory tab, "Memory Pressure" graph). Yellow is acceptable; red means shrink something.

> [!WARNING] Real targets are not your laptop
> The rates above (`-rl 10`) are fine for a local container. On a real program, use the program's stated limit; if none is stated, start low (a few requests per second) and watch for errors. Our wrappers will take the number from `scope.yaml` automatically.

## Recap

- Install the ProjectDiscovery tools one way only: Homebrew (simplest), `go install` (from source) or pdtm (suite manager). Check with `which -a`.
- The `openai` package pulls in Python `httpx`, whose script shadows the Go `httpx` inside the venv. `tools.py` resolves real binaries and skips Python scripts.
- Tools are run with argument lists, never through a shell.
- Pass `-duc` to skip update checks and `-ni` to keep nuclei's out-of-band callbacks off public servers.
- Cap the Docker VM at 2–3 GB and use 32K context for the model so everything fits in 16 GB.
- Juice Shop runs from `bkimminich/juice-shop` on port 3000, bound to `127.0.0.1` only.

## Check yourself

1. You activate the venv, run `httpx -l hosts.txt -sc` and get an error about missing CLI dependencies. What happened and what are two fixes?
   <details><summary>Answer</summary>

   The venv's `bin` folder is first on `PATH`, and it contains the Python `httpx` script installed as a dependency of `openai`. Fix it by calling the Go binary by full path (or an alias), by deactivating the venv in that terminal, or, in code, by resolving the binary with `tools.resolve()`, which skips Python scripts.

   </details>
2. Why bind Juice Shop with `-p 127.0.0.1:3000:3000` instead of `-p 3000:3000`?
   <details><summary>Answer</summary>

   The short form publishes the port on all interfaces, so anyone on the same network could reach a deliberately vulnerable app on your laptop. Binding to loopback keeps it reachable only from your own machine.

   </details>
3. With Docker Desktop at its default settings on your 16 GB Mac, what is the VM's memory limit, and why does it matter for the model?
   <details><summary>Answer</summary>

   50% of host memory, so 8 GB. The VM can grow toward that limit and compete with the model's 5.3 GB of weights plus KV cache, Burp and macOS. If unified memory runs out, macOS swaps and token generation slows sharply. Capping the VM at 2–3 GB avoids that.

   </details>
4. Why does the verification lab test subfinder and dnsx against `example.com` instead of Juice Shop?
   <details><summary>Answer</summary>

   subfinder queries third-party passive sources and dnsx queries public DNS resolvers; neither has anything to say about `localhost`. `example.com` is reserved for documentation, so querying public data about it is harmless and proves the tools run.

   </details>
5. What do `-duc` and `-ni` do, and why do they matter for a private setup?
   <details><summary>Answer</summary>

   `-duc` disables the automatic update check each ProjectDiscovery tool performs on start. `-ni` stops nuclei from using interactsh, which by default registers out-of-band callbacks with public servers. Both reduce what leaves your machine.

   </details>

## Go deeper

- [ProjectDiscovery documentation](https://docs.projectdiscovery.io/) — official docs for subfinder, dnsx, httpx, katana and nuclei
- [httpx README](https://github.com/projectdiscovery/httpx) — the full flag list; run `httpx -h` for your installed version
- [ffuf](https://github.com/ffuf/ffuf) — usage, filters and output formats
- [OWASP Juice Shop on GitHub](https://github.com/juice-shop/juice-shop) — setup options and architecture notes
- [Colima](https://github.com/abiosoft/colima) — lightweight container runtime for macOS with explicit VM sizing
