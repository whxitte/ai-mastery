# Why local AI for security work

You already know the question every client security questionnaire asks: "Where does our data go?" This lesson is about answering it honestly when an LLM is part of your workflow, and about the real trade-offs of a 9B model on your own laptop against a frontier model in someone else's datacenter.

> [!GOALS]
> - Explain the six concrete reasons a security professional runs models locally
> - Name the limitations of a local 9B model, with numbers from your own machine
> - Classify engagement data and decide where each class may be processed
> - Build a reversible sanitizer so you can use frontier models on scrubbed data

## The short version

A local model is a tool that **never phones home with your prompt**. Everything you type, paste or pipe into LM Studio is processed by weights sitting in your unified memory, and the answer is computed on your GPU. Nothing about that requires a network connection once the model file is on disk. LM Studio's own documentation says that nothing you enter while chatting with a local model leaves your device, and that document processing for chat-with-files happens locally too ([offline docs](https://lmstudio.ai/docs/app/offline)).

That single property changes what you are allowed to do with an LLM on an engagement. The rest of this lesson is the detail.

## Six reasons to run local

### 1. Confidentiality: client data, NDAs, PII, source code

On a typical engagement you handle things you are contractually forbidden to share with third parties:

- **Client data**: HTTP traffic with session tokens, internal hostnames, API responses full of customer records.
- **NDA-covered material**: architecture diagrams, pre-release features, the scope document itself.
- **PII**: names, emails, phone numbers, national IDs that show up in responses, logs and database dumps.
- **Source code**: white-box reviews, leaked `.git` folders you found during testing, client repositories.
- **Embargoed vulnerability details**: findings that are not yet fixed or disclosed.

Pasting any of that into a hosted chatbot is, from the client's point of view, a disclosure to a sub-processor they never approved. Whether the provider trains on it, how long they retain it, and which jurisdiction holds it are details your client's lawyer cares about, and they change over time. A local model removes the question: the data stays on a device you control, under the same disk encryption and access controls as the rest of your engagement files.

> [!NOTE]
> "Local" is only as private as the machine. Chat history is written to disk in `~/.lmstudio/conversations/`, and logs can contain prompts. Treat those like any other engagement artifact. The [threat model lesson](#/security/ai-threat-model) and [hardening lesson](#/infra/hardening) cover this.

### 2. Offline and air-gapped engagements

Some work happens where there is no internet: OT/ICS sites, classified or regulated networks, a client's isolated lab, or an on-site test where the rules of engagement forbid connecting your laptop to anything but the test network. A local model works the same at 30,000 feet as it does at your desk. You download the model once, verify its hash, and carry it in. The [offline stack lesson](#/infra/offline-stack) and the [air-gap procedure](#/infra/hardening) show how.

### 3. No rate limits, no per-token bill

Security work produces large, boring inputs: 40,000 lines of httpx output, a 6 MB minified JavaScript bundle, a week of nginx logs. With a hosted API you pay per token and hit rate limits. Locally, the only costs are time, heat and battery. You can run a triage job over a full recon dataset overnight and nobody sends you an invoice.

> [!MAC]
> Your MacBook Air is fanless. Long batch jobs will thermally throttle after several minutes, and generation speed drops. Run big jobs plugged in, on a hard surface, and expect them to be slower than the first minute suggests. The [performance lesson](#/lmstudio/performance) covers this.

### 4. No refusals on legitimate offensive topics

Frontier assistants are trained to be cautious, and they sometimes refuse perfectly normal professional requests: "write a SQL injection payload list for this parameter", "explain how this deserialization gadget chain works", "draft a phishing pretext for our authorized social engineering test". Your model, `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`, has had its refusal behavior removed (the [alignment lesson](#/foundations/alignment-uncensored) explains how).

That is genuinely useful, and it shifts responsibility to you. The model will not stop you from pointing it at an out-of-scope asset or from writing something you should not. Your scope file, your rules of engagement and your judgment are the safety net now, which is how it has always worked with Burp and nmap anyway.

### 5. Reproducibility: pin the model like a dependency

Hosted models are updated, re-tuned and retired on the provider's schedule. The answer you got in March may not be reproducible in September, which matters when a finding, a detection rule or a methodology document depends on it.

A local model is a file. If you record:

| What to pin | Example | How to get it |
|---|---|---|
| Model file and its SHA-256 | `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`, `2ca636d9...` | `shasum -a 256 <file>` |
| Hugging Face repo commit | `0a41c688...` | repo "Files" tab or the HF API |
| Runtime and app version | LM Studio version, llama.cpp runtime version | LM Studio's About screen and its runtime settings |
| Load and sampling settings | context length, temperature, top_p, top_k, seed | saved preset |

...then you can rerun the same model next year and get the same behavior (with the caveat that a different runtime version or hardware can still change outputs slightly). The [hardening lesson](#/infra/hardening) builds a hash manifest for this.

### 6. You can look inside

Weights on your disk can be inspected, probed, steered, fine-tuned and edited. That is the whole of [Part 8](#/weights/mit-study) of this course. You cannot do any of that with a hosted model.

## The honest limitations

A local 9B model is not a frontier model in a smaller box. Know where it fails before it fails you on a report.

### Knowledge cutoff vs fresh CVEs

A model only knows what was in its training data. Qwen3.5-9B was released on 2026-03-02, so its training data stops some time before that. Anything published after the cutoff (most 2026 CVEs, new tool versions, new flags, a framework's latest breaking change) is simply absent. Worse, the model does not know what it does not know: ask it about a recent CVE and it will often produce a fluent, confident, invented answer. The [hallucination lesson](#/security/hallucination-verification) shows this live and fixes it with grounding.

> [!WARNING]
> Never take a CVE ID, CVSS score, affected version range or tool flag from a model without checking an authoritative source (NVD, the vendor advisory, the GitHub advisory, `--help`). This applies to frontier models too, but a 9B model invents more often.

### Hallucination and weaker reasoning

Small models make more factual errors and lose the thread on multi-step reasoning more often than frontier models. In practice, for security work:

- **Good at**: summarizing, extracting structured data, triaging long lists, explaining a pasted advisory, rewriting notes into a report, generating candidate wordlists, spotting obvious patterns in code.
- **Weak at**: long chains of reasoning (complex auth bypass logic across many files), exact recall of facts, arithmetic, anything that needs knowledge after the cutoff, judging subtle exploitability.

The fix is not "never use it", but to design workflows where the model proposes and something deterministic (a tool, a grep, a human) verifies. Every recipe in [Security workflows](#/security/workflows) is built that way.

### Small context in practice

Qwen3.5-9B advertises a native context of 262,144 tokens. On 16 GB you cannot use most of it. Here is the arithmetic, using the numbers from [Memory math](#/foundations/memory-math):

- Weights (Q4_K_M): 5.63 GB on Hugging Face, which is about **5.24 GiB** in memory.
- KV cache for this model: **32 KiB per token** (only 8 of its 32 layers keep a growing cache).
- The GPU can wire roughly two-thirds of 16 GiB by default, about **10.7 GiB**.

| Context | KV cache | Weights + KV | Fits in ~10.7 GiB GPU budget? |
|---|---|---|---|
| 16K | 0.5 GiB | 5.74 GiB | Yes, comfortable |
| 32K | 1 GiB | 6.24 GiB | Yes |
| 64K | 2 GiB | 7.24 GiB | Yes, with less headroom |
| 128K | 4 GiB | 9.24 GiB | Barely; compute buffers and other apps push you over |
| 262K | 8 GiB | 13.24 GiB | No |

And memory is only half the story. Long contexts are slow to process (prefill), and small models get noticeably worse at using information buried in the middle of a long prompt. For security work, plan on **16K to 32K tokens of useful context** and chunk anything bigger. The workflows lesson shows how.

## A hybrid strategy

The professional answer is not "local only" or "cloud only". It is: **classify the data first, then pick the processor**. Local models handle anything sensitive; frontier models handle the hard reasoning, but only on data that has been sanitized or was never sensitive.

### Data classification for engagements

| Class | Examples | Local model | Frontier model |
|---|---|---|---|
| **Public** | Published CVEs and advisories, tool docs, OWASP material, your own public blog drafts, CTF challenges after the event | Yes | Yes |
| **Internal** | Your methodology notes, generic scripts, sanitized lab output, questions about a technique with no target details | Yes | Yes |
| **Client confidential** | Target hostnames and IPs, HTTP traffic, findings, scan output, screenshots, client source code | Yes | Only after sanitization, and only if the contract allows any third-party processing |
| **Restricted** | Credentials and tokens found during testing, PII and customer records, unpatched vulnerability details under embargo, anything the contract explicitly ring-fences | Yes, on an encrypted disk, minimal retention | Never, sanitized or not |

> [!DANGER]
> Read the contract before you decide. Some agreements forbid processing client data with any third-party service at all, including "sanitized" data. Sanitization reduces risk, it does not change what you agreed to. When in doubt, keep it local and ask the client in writing.

### The decision in one picture

```mermaid
flowchart TD
    A["New input"] --> B{"Classification?"}
    B -->|"Public / internal"| C["Any model"]
    B -->|"Client confidential"| D{"Contract allows<br/>third-party processing?"}
    B -->|"Restricted"| E["Local model only"]
    D -->|"No"| E
    D -->|"Yes"| F{"Hard reasoning needed?"}
    F -->|"No"| E
    F -->|"Yes"| G["Sanitize locally"]
    G --> H["Frontier model on placeholders"]
    H --> I["Restore placeholders locally"]
```

The pattern that falls out: let the local model do the bulk work on raw data (triage, extraction, summarization), and when you hit a genuinely hard reasoning problem, distill it into a small, sanitized question for a frontier model. "Here is a JWT validation function with placeholders for hostnames; can the `kid` header be abused?" is a fine thing to ask anywhere. "Here is the client's whole auth service" is not.

## Lab: a reversible sanitizer

Sanitizing by hand is slow and error-prone. The script below replaces sensitive strings with stable placeholders like `<<HOST_1>>`, saves the mapping locally, and restores the originals in the answer. It is **pseudonymization**, the same idea as tokenizing card numbers: the hosted model sees structure but not identity.

> [!CONCEPT] Pseudonymization
> Replacing identifying values with consistent placeholders, while keeping a private mapping so you can reverse it. The same hostname always becomes the same placeholder, so the text still makes sense ("HOST_1 redirects to HOST_2"), but the real values never leave your machine.

> [!LAB] Build and test the sanitizer
> 1. Create the folder and a virtual environment (you will reuse it for the whole security module):
>    ```bash
>    mkdir -p ~/ai-lab/sec && cd ~/ai-lab/sec
>    python3 -m venv .venv && source .venv/bin/activate
>    pip install openai requests
>    ```
> 2. Save the script below as `~/ai-lab/sec/sanitize.py`.
> 3. Create `terms.txt` with one client-specific term per line (the client's apex domains, company name, product names).
> 4. Run `scrub` on some notes, read the output carefully, then run `restore` on it and check you get the original back.

```python title="sanitize.py"
#!/usr/bin/env python3
"""Reversible pseudonymization for engagement text.

usage:
  python3 sanitize.py scrub   --map eng.map.json --terms terms.txt < notes.txt > notes.clean.txt
  python3 sanitize.py restore --map eng.map.json < answer.txt > answer.real.txt
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Generic patterns, applied after client-specific terms. Over-redaction is fine; under-redaction is not.
GENERIC = [
    ("JWT", r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"),
    ("AWSKEY", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("BEARER", r"(?i)(?<=bearer )[A-Za-z0-9._~+/=-]{16,}"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    ("IPV4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
]


def term_patterns(terms_file: str | None) -> list[tuple[str, str]]:
    """Domains match with any subdomain prefix; other terms match as whole words."""
    if not terms_file:
        return []
    out = []
    terms = [t.strip() for t in Path(terms_file).read_text().splitlines() if t.strip()]
    for term in sorted(terms, key=len, reverse=True):  # longest first
        if "." in term:
            out.append(("HOST", r"(?i)\b(?:[a-z0-9-]+\.)*" + re.escape(term) + r"\b"))
        else:
            out.append(("NAME", r"(?i)\b" + re.escape(term) + r"\b"))
    return out


def scrub(text: str, patterns: list[tuple[str, str]], mapping: dict[str, str]) -> str:
    reverse = {orig: ph for ph, orig in mapping.items()}

    def placeholder(kind: str, value: str) -> str:
        if value not in reverse:
            n = sum(1 for ph in mapping if ph.startswith(f"<<{kind}_")) + 1
            ph = f"<<{kind}_{n}>>"
            mapping[ph], reverse[value] = value, ph
        return reverse[value]

    for kind, rx in patterns:
        text = re.sub(rx, lambda m, k=kind: placeholder(k, m.group(0)), text)
    return text


def restore(text: str, mapping: dict[str, str]) -> str:
    for ph, orig in mapping.items():
        text = text.replace(ph, orig)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scrub", "restore"])
    ap.add_argument("--map", required=True, help="JSON mapping file (keep it local)")
    ap.add_argument("--terms", help="file with client domains/names, one per line")
    args = ap.parse_args()

    map_path = Path(args.map)
    mapping = json.loads(map_path.read_text()) if map_path.exists() else {}
    data = sys.stdin.read()

    if args.mode == "scrub":
        sys.stdout.write(scrub(data, term_patterns(args.terms) + GENERIC, mapping))
        map_path.write_text(json.dumps(mapping, indent=2))
        os.chmod(map_path, 0o600)  # the map IS the sensitive data now
        print(f"[{len(mapping)} values pseudonymized, map in {map_path}]", file=sys.stderr)
    else:
        sys.stdout.write(restore(data, mapping))
```

Try it:

```bash
cd ~/ai-lab/sec
printf 'example.com\nAcme Corp\n' > terms.txt
cat > notes.txt <<'EOF'
Acme Corp staging API at api-staging.example.com (10.20.0.15) accepts
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoidXNlciJ9.c2lnbmF0dXJl
and returns other tenants' invoices; contact was j.doe@example.com.
EOF
python3 sanitize.py scrub --map eng.map.json --terms terms.txt < notes.txt
```

```text
<<NAME_1>> staging API at <<HOST_1>> (<<IPV4_1>>) accepts
Authorization: Bearer <<JWT_1>>
and returns other tenants' invoices; contact was j.doe@<<HOST_2>>.
[5 values pseudonymized, map in eng.map.json]
```

Notice two things. First, the email was only partly scrubbed: the domain matched the client term before the email pattern ran, so the local part `j.doe` survived. That is exactly the kind of leak you must catch by reading the output. Fix it by moving `EMAIL` above the term patterns, or by adding the email to `terms.txt`. Second, the facts that make the finding interesting (staging, bearer token, cross-tenant invoices) survived, which is what the frontier model needs.

> [!TIP]
> Use your local model as a second pass: ask it "List any remaining names, hostnames, IDs or secrets in this text" on the scrubbed output. Regex catches the patterns you thought of; the model catches the ones you did not. Then read it yourself anyway.

> [!WARNING]
> Sanitized text can still identify a client. A unique stack trace, an unusual product name, or a distinctive vulnerability can be enough. Sanitization lowers risk; the classification table above still decides.

## Recap
- Local inference keeps prompts and documents on your device, which is what makes LLMs usable on confidential engagements.
- The six reasons: confidentiality, offline work, no rate limits or per-token cost, no refusals on legitimate offensive work, reproducibility through pinned files and hashes, and the ability to inspect and modify the model.
- The limits: knowledge cutoff (no fresh CVEs), more hallucination and weaker multi-step reasoning than frontier models, and a practical context of roughly 16K to 32K tokens on 16 GB.
- Classify data first. Restricted data stays local; client-confidential data goes to a frontier model only sanitized and only if the contract allows it.
- Pseudonymize with a reversible mapping, keep the map local and private, and always read the scrubbed output before sending it anywhere.

## Check yourself
1. A client's MSA says "no processing of Client Data by third-party services". You have a hard reasoning question about their JWT validation code. What are your options?
   <details><summary>Answer</summary>

   Use the local model on the real code. If you need frontier-level reasoning, the safest route is to write a fresh, generic reproduction of the pattern (your own code, no client identifiers or logic copied verbatim) and ask about that, or ask the client in writing for permission. Sanitizing their code and sending it would still be processing Client Data by a third party.

   </details>
2. Why is a model's answer about a CVE published last month especially dangerous?
   <details><summary>Answer</summary>

   The CVE is after the model's knowledge cutoff, so the model has no real information about it, but it will still generate a fluent, confident answer by pattern-matching on similar CVEs. You get plausible fiction: wrong versions, wrong scores, wrong root cause.

   </details>
3. Using 32 KiB of KV cache per token and 5.24 GiB of weights, how much memory does a 48K-token context need, and does it fit the default GPU budget?
   <details><summary>Answer</summary>

   48K tokens × 32 KiB = 1.5 GiB of KV cache. Plus 5.24 GiB of weights = 6.74 GiB, before compute buffers. That fits comfortably under roughly 10.7 GiB.

   </details>
4. What four things do you record to make a local model's behavior reproducible?
   <details><summary>Answer</summary>

   The exact model file and its SHA-256 hash (plus the repository commit it came from), the app and runtime versions, and the load and sampling settings (context length, temperature, top_p, top_k, seed).

   </details>
5. In the sanitizer lab, why is the mapping file saved with `0600` permissions?
   <details><summary>Answer</summary>

   The mapping links every placeholder to the real value, so it is now the most sensitive file in the workflow. Only your user should be able to read it, and it should never be sent along with the scrubbed text.

   </details>

## Go deeper
- [LM Studio: offline operation](https://lmstudio.ai/docs/app/offline) — exactly which features work without internet and what stays on your device
- [LM Studio privacy policy](https://lmstudio.ai/app-privacy) — what the app itself sends (update checks, model search) and what it does not
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) — "Sensitive Information Disclosure" is LLM02; useful vocabulary for client conversations
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — why a long context is not the same as a usable context
