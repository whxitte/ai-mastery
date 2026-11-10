# Reporting and human-in-the-loop

The agent produces leads; you turn one into a verified finding. This lesson covers the last mile: what a strong report contains, how to reason about CVSS with the model as an assistant rather than a judge, a `report.py` that drafts a report locally from your notes and raw HTTP traffic after redacting secrets and personal data, a duplicate check, and the rules that keep AI-assisted reporting on the right side of platform policy.

> [!GOALS]
> - Know the anatomy of a report triagers can reproduce in minutes
> - Reason through CVSS 3.1 and 4.0 vectors, and compute scores in code instead of trusting the model's arithmetic
> - Build `redact.py` and `report.py`: redact, draft, score, check for duplicates, gate before submission
> - Apply human-in-the-loop rules that match current HackerOne, Bugcrowd and open-source maintainer stances

## Anatomy of a strong report

Triagers read many reports a day. The best ones let them reproduce the issue on the first try and understand why it matters without a call.

| Section | What goes in it | Common mistake |
|---|---|---|
| Title | Vulnerability class + where + impact, e.g. "IDOR in `GET /rest/basket/{id}` exposes any user's basket" | "Security issue found" |
| Summary | Two to four sentences: what, where, who can exploit it, what they get | A paragraph of generic OWASP text |
| Severity | Vector plus score in the program's system (CVSS or the platform's taxonomy), with a one-line justification per metric | Score without reasoning, or inflated impact |
| Steps to reproduce | Numbered, minimal, copy-pasteable; accounts and roles named ("attacker = account A") | Missing preconditions, steps that only work in your session |
| Proof of concept | Raw request and response (redacted), screenshot or short video if UI-driven | Huge Burp dumps with live cookies |
| Impact | Concrete consequence for this business and its users, at the level you demonstrated | Theoretical chains you did not test |
| Remediation | Short, specific fix (server-side ownership check, allowlist by parsed host) | Copy-pasted cheat sheet |
| Notes | Scope confirmation, test accounts used, what you did not do (for example, "accessed only my second test account's basket") | Nothing about data handling |

> [!NOTE] Severity systems differ by platform
> HackerOne's severity field is CVSS-based; Bugcrowd uses its own Vulnerability Rating Taxonomy (VRT, priorities P1 to P5); many programs add their own rules on top. Always use the program's stated system and read its severity guidance. The CVSS reasoning below is useful either way, because it forces you to state attack conditions and impact precisely.

## CVSS: the model assists, you decide

> [!CONCEPT] CVSS vector
> The Common Vulnerability Scoring System describes a vulnerability as a set of metrics (how it is reached, how hard it is, what privileges and user action it needs, what it affects) written as a compact string such as `CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N`. A published formula turns the vector into a 0–10 score. The vector is the argument; the score is just arithmetic.

### CVSS 3.1 base metrics

| Metric | Values | Question to ask |
|---|---|---|
| AV, Attack Vector | N, A, L, P | From where can the attacker reach it? Internet = N |
| AC, Attack Complexity | L, H | Does it depend on conditions outside the attacker's control? |
| PR, Privileges Required | N, L, H | What account does the attacker need? |
| UI, User Interaction | N, R | Must a victim do something? |
| S, Scope | U, C | Does impact cross a security boundary into another component? |
| C / I / A | N, L, H | How much confidentiality, integrity, availability is lost? |

Worked example, Juice Shop "View Basket": an authenticated user reads another user's basket by changing the id. Network reachable, no special conditions, needs a normal account, no victim action, stays within the app, limited confidentiality loss (basket contents), no integrity or availability impact:

```text
CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N
Impact         = 6.42 x (1 - (1-0.22)(1-0)(1-0)) = 6.42 x 0.22 = 1.41
Exploitability = 8.22 x 0.85 (AV:N) x 0.77 (AC:L) x 0.62 (PR:L) x 0.85 (UI:N) = 2.84
Base           = roundup(1.41 + 2.84) = 4.3  (Medium)
```

Change one judgment and the score moves: if the basket included addresses and payment details for every user, C:H gives 6.5. That is exactly the kind of decision you, not the model, should make, based on what you actually observed. For comparison, an unauthenticated SQL injection that dumps all users' credentials (Juice Shop's "User Credentials" challenge) is commonly scored `AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N` = 7.5 (High) when you only demonstrated reading.

### CVSS 4.0 in one table

CVSS 4.0 (published by FIRST in late 2023) splits things differently: a new **Attack Requirements (AT)** metric for preconditions, **User Interaction** with Passive and Active values, and separate impact on the **vulnerable system** (VC, VI, VA) and **subsequent systems** (SC, SI, SA) instead of the Scope metric. Scores are labelled by which metric groups were used: CVSS-B (base only), CVSS-BT (with Threat), CVSS-BE, CVSS-BTE.

| 3.1 idea | 4.0 equivalent |
|---|---|
| AC covers both difficulty and preconditions | AC (difficulty) + AT (preconditions such as a race or a specific configuration) |
| UI: N / R | UI: N / P (passive) / A (active) |
| Scope U/C with C/I/A | VC/VI/VA for the vulnerable system, SC/SI/SA for systems beyond it |

The basket example in 4.0 would be `CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:N/VA:N/SC:N/SI:N/SA:N`. 4.0 scores come from a lookup-based algorithm rather than a short formula, so compute them with a library or [FIRST's calculator](https://www.first.org/cvss/calculator/4.0), never by hand or by the model.

The division of labour in `report.py`:

1. The model **proposes** a vector with a one-line rationale per metric, based on your notes.
2. The `cvss` library **computes** the score (models are unreliable at this arithmetic).
3. **You** read the rationale, change any metric you disagree with, and the draft records your final vector.

## Redaction before anything leaves the machine

The model is local, so redaction is not about the LLM call itself. It is about everything downstream: the draft you paste into a platform, attachments, the `llm_runs` log, a colleague you share notes with. Your PoC traffic contains your session tokens and, in many real bugs, other people's data. Redact first, draft from redacted text, and keep the mapping only on your machine.

```python title="redact.py"
"""Redact secrets and personal data from text. Returns redacted text + a local-only mapping."""
from __future__ import annotations

import json
import re
from pathlib import Path

PATTERNS = [  # (label, regex); group 1, if present, is the part to replace
    ("AUTH_HEADER", re.compile(r"(?im)^(?:authorization|proxy-authorization|x-api-key|x-auth-token):\s*(.+)$")),
    ("COOKIE", re.compile(r"(?im)^(?:cookie|set-cookie):\s*(.+)$")),
    ("JWT", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("AWS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")),
    ("SECRET_FIELD", re.compile(r"""(?i)["']?(?:password|passwd|secret|token|api_?key|totp|otp)["']?\s*[:=]\s*["']?([^"'&\s,}]{3,})""")),
    ("SECRET_PARAM", re.compile(r"(?i)[?&](?:token|key|api_?key|code|session|sig|signature)=([^&\s]+)")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)")),
]


def redact(text: str, mapping: dict | None = None) -> tuple[str, dict]:
    """Replace each distinct secret with a stable placeholder like [REDACTED-JWT-1]."""
    mapping = {} if mapping is None else mapping
    reverse = {v: k for k, v in mapping.items()}

    for label, rx in PATTERNS:
        def sub(m: re.Match) -> str:
            secret = m.group(1) if m.groups() else m.group(0)
            if "[REDACTED-" in secret:
                return m.group(0)
            if secret not in reverse:
                n = sum(1 for k in mapping if k.startswith(f"[REDACTED-{label}-")) + 1
                token = f"[REDACTED-{label}-{n}]"
                mapping[token], reverse[secret] = secret, token
            return m.group(0).replace(secret, reverse[secret])
        text = rx.sub(sub, text)
    return text, mapping


def find_leftovers(text: str) -> list[str]:
    """Anything that still looks secret after redaction: used as a submission gate."""
    hits = []
    for label, rx in PATTERNS:
        for m in rx.finditer(text):
            value = m.group(1) if m.groups() else m.group(0)
            if "[REDACTED-" not in value:
                hits.append(f"{label}: {value[:12]}...")
    return hits


def save_mapping(mapping: dict, folder: Path) -> None:
    p = folder / "redaction-map.json"   # never attach or submit this file
    p.write_text(json.dumps(mapping, indent=2))
    p.chmod(0o600)
```

The same value always gets the same placeholder (`[REDACTED-JWT-1]` everywhere), so the report stays readable, and the mapping file lets you re-identify values locally if a triager asks.

> [!WARNING] Regexes miss things, and sometimes over-match
> Names, street addresses, order contents and free-text fields have no reliable pattern, and the card and phone patterns will also catch long numeric ids. Read the redacted request and response yourself before submitting, and un-redact harmless values (such as the basket id that proves the IDOR) by hand. When you demonstrate access to other users' data, show the minimum (one record, partially masked) and say so in the report.

```python title="tests/test_redact.py"
from redact import find_leftovers, redact

RAW = """GET /rest/basket/2 HTTP/1.1
Host: localhost:3000
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXZhbHVl
Cookie: token=abc123def456

{"email": "jim@juice-sh.op", "password": "hunter2hunter2"}"""


def test_redacts_and_is_consistent():
    out, mapping = redact(RAW)
    assert "eyJhbGci" not in out and "abc123def456" not in out
    assert "jim@juice-sh.op" not in out and "hunter2hunter2" not in out
    assert find_leftovers(out) == []
    again, _ = redact(RAW, dict(mapping))
    assert again == out          # same input and mapping, same placeholders
```

## `report.py`: draft, score, check, gate

Folder convention, one per finding:

```text
reports/juice-basket-idor/
├── notes.md            # your own words: what you did, what you saw, accounts used
├── request.txt         # raw request copied from Burp's message editor
├── response.txt        # raw response
├── draft.md            # written by report.py
└── redaction-map.json  # local only, never submitted
```

Also keep `reports/known.txt` with one title per line for issues already reported on this program: your own previous reports, plus titles from the program's public disclosures or known-issues list, copied by hand.

```python title="report.py"
"""Draft a bug report locally: redact -> LLM draft -> CVSS computed in code -> duplicate check.

Usage:
  python report.py draft reports/<slug>
  python report.py check reports/<slug>/draft.md
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

from cvss import CVSS3, CVSS4
from pydantic import BaseModel, Field

from config import REPORT_DIR
from llm import call_with_schema, data_block
from redact import find_leftovers, redact, save_mapping

PROMPT_VERSION = "report-v1"
FENCE = "`" * 3


class MetricReason(BaseModel):
    metric: str = Field(max_length=4)
    value: str = Field(max_length=2)
    why: str = Field(max_length=200)


class ReportDraft(BaseModel):
    title: str = Field(max_length=140)
    summary: str = Field(max_length=800)
    steps: list[str] = Field(min_length=2, max_length=12)
    impact: str = Field(max_length=900)
    remediation: str = Field(max_length=600)
    cvss31_vector: str = Field(max_length=60)
    cvss31_reasons: list[MetricReason] = Field(max_length=8)
    cvss40_vector: str = Field(max_length=120)
    weakness_guess: str = Field(max_length=80, description="CWE id and name, to be verified")
    open_questions: list[str] = Field(max_length=5, description="facts the hunter must confirm")


SYSTEM = """Task: draft a bug bounty report from the hunter's own notes and redacted HTTP traffic.
Use only facts stated in the notes or visible in the traffic. Do not add impact the notes do not
demonstrate; if something is uncertain, put it in open_questions instead.
Keep [REDACTED-...] placeholders exactly as they are.
Steps: numbered, minimal, reproducible by a triager with their own accounts.
CVSS: propose a CVSS 3.1 vector and a CVSS 4.0 vector with one reason per 3.1 metric.
Do not compute scores."""


def score(vector: str) -> str:
    try:
        if vector.startswith("CVSS:3."):
            c = CVSS3(vector)
        elif vector.startswith("CVSS:4.0/"):
            c = CVSS4(vector)
        else:
            return "INVALID VECTOR"
        return f"{c.scores()[0]} ({c.severities()[0]})"
    except Exception as exc:  # the library raises on malformed vectors
        return f"INVALID VECTOR: {exc}"


def duplicates(title: str, known_file: Path, threshold: float = 0.6) -> list[tuple[float, str]]:
    if not known_file.exists():
        return []
    known = [k.strip() for k in known_file.read_text().splitlines() if k.strip()]
    scored = [(difflib.SequenceMatcher(None, title.lower(), k.lower()).ratio(), k) for k in known]
    return sorted((s for s in scored if s[0] >= threshold), reverse=True)[:5]


def draft(folder: Path) -> Path:
    mapping: dict = {}
    parts = {}
    for name in ("notes.md", "request.txt", "response.txt"):
        f = folder / name
        parts[name], mapping = redact(f.read_text(errors="replace") if f.exists() else "", mapping)
    save_mapping(mapping, folder)
    user = ("Hunter's notes:\n" + data_block("notes", parts["notes.md"], 8000) +
            "\n\nRequest:\n" + data_block("request", parts["request.txt"], 6000) +
            "\n\nResponse (may be truncated):\n" + data_block("response", parts["response.txt"], 6000))
    d = call_with_schema(ReportDraft, SYSTEM, user, module="report",
                         prompt_version=PROMPT_VERSION, max_tokens=3000)
    if d is None:
        sys.exit("model did not produce a valid draft; write it by hand or retry")
    dups = duplicates(d.title, REPORT_DIR / "known.txt")
    reasons = "\n".join(f"| {r.metric} | {r.value} | {r.why} |" for r in d.cvss31_reasons)
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(d.steps, 1))
    questions = "\n".join(f"- {q}" for q in d.open_questions) or "- none"
    dup_lines = "\n".join(f"- {s:.2f} similar: {k}" for s, k in dups) or "- none found in reports/known.txt"
    md = f"""<!-- DRAFT: unverified AI-assisted text. Do not submit until every box is ticked. -->
# {d.title}

## Human review (all required)
- [ ] I reproduced every step myself, today, against an in-scope asset
- [ ] Every sentence below is true and I can explain it without this tool
- [ ] I set the final severity myself (edit the vectors below if needed)
- [ ] I checked the redacted request and response by eye for personal data
- [ ] I searched for duplicates (below) and the program's disclosed reports
- [ ] I follow the program's and platform's rules on disclosing AI assistance

## Summary
{d.summary}

## Severity
- CVSS 3.1: `{d.cvss31_vector}` = {score(d.cvss31_vector)}
- CVSS 4.0: `{d.cvss40_vector}` = {score(d.cvss40_vector)}

| Metric | Value | Reason (model's proposal, check it) |
|---|---|---|
{reasons}

## Steps to reproduce
{steps}

## Proof of concept
{FENCE}http
{parts["request.txt"].strip()[:4000]}
{FENCE}

{FENCE}http
{parts["response.txt"].strip()[:3000]}
{FENCE}

## Impact
{d.impact}

## Remediation
{d.remediation}

## Weakness (verify at cwe.mitre.org)
{d.weakness_guess}

## Open questions for the hunter
{questions}

## Possible duplicates
{dup_lines}
"""
    out = folder / "draft.md"
    out.write_text(md)
    return out


def check(path: Path) -> int:
    text = path.read_text()
    problems = []
    if "- [ ]" in text:
        problems.append("unticked review boxes")
    if left := find_leftovers(text):
        problems.append(f"possible unredacted secrets or PII: {left[:5]}")
    if "INVALID VECTOR" in text:
        problems.append("invalid CVSS vector")
    if re.search(r"(?i)\bTODO\b", text):
        problems.append("TODO left in text")
    for p in problems:
        print("BLOCK:", p)
    if not problems:
        print("OK: passes the local gate. You still submit it by hand.")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("draft", "check"):
        sys.exit(__doc__)
    target = Path(sys.argv[2])
    if sys.argv[1] == "draft":
        print("draft written:", draft(target))
    else:
        sys.exit(check(target))
```

Design points:

- **The model only sees redacted text.** Placeholders flow through to the draft, so the draft never contains your tokens.
- **Scores are computed, not generated.** An invalid vector shows up as `INVALID VECTOR` and blocks the check.
- **`open_questions`** gives the model a sanctioned place to put uncertainty instead of inventing facts.
- **The CWE is a guess** labelled "verify": models often pick a plausible but wrong CWE id.
- **`check` is a gate, not an approval.** It refuses drafts with unticked boxes, leftover secrets, bad vectors or TODOs. There is deliberately no `submit` command: submission is a human action on the platform.
- **Duplicate checking is lexical and local.** Title similarity against `known.txt` catches the obvious cases; it does not replace reading the program's disclosed reports. The [evaluation lesson](#/bugbounty/evaluate) suggests an embedding-based upgrade.

> [!LAB] Report a verified Juice Shop bug
> 1. From your hunt brief, pick an IDOR-style hypothesis such as reading another user's basket. Register two accounts (A and B) in your local Juice Shop.
> 2. Verify by hand in Burp: as A, request B's basket. Confirm you see B's items. Stop there.
> 3. Create `reports/juice-basket-idor/`. Write `notes.md` in your own words (accounts, steps, what you saw), and paste the raw request and response from Burp into `request.txt` and `response.txt`.
> 4. Run the tests and the draft:
>    ```bash
>    pytest -q tests/test_redact.py
>    python report.py draft reports/juice-basket-idor
>    ```
> 5. Read `draft.md` line by line. Fix anything wrong, change the CVSS vector if you disagree (`python -c "from report import score; print(score('CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N'))"` prints the score for any vector), answer the open questions.
> 6. Run `python report.py check reports/juice-basket-idor/draft.md`. It blocks until the boxes are ticked. Tick each only when it is true.
> 7. Compare the model's proposed vectors with yours and note disagreements; they are evaluation data for the next lesson.

## Human-in-the-loop rules

These are the rules this project builds in, and they line up with where platforms and maintainers have landed:

1. **Verify every finding manually** before it leaves your machine. Reproduce it yourself, with your own accounts, against an in-scope asset.
2. **Never submit unverified AI output**: no "possible" findings, no scanner hits you did not reproduce, no impact you did not demonstrate.
3. **You own every word.** If you cannot explain a sentence to a triager without the tool, delete it.
4. **Keep autonomy away from submission.** The agent may suggest; only you submit, comment and respond.
5. **Respect the program's automation and rate rules**; AI-assisted testing is still testing.
6. **Disclose AI assistance** where the program or platform asks you to.

### What platforms and maintainers say (as of late 2026)

Policies are changing quickly; read the current text on the platform and the specific program page before relying on this summary.

- **HackerOne.** Its [Code of Conduct](https://www.hackerone.com/policies/code-of-conduct) now contains explicit AI rules. It permits and encourages responsible use of AI tools, keeps members fully responsible for their submissions, says hackbots must not operate fully autonomously (human experts must validate findings before submission), requires AI-assisted testing to stay within each program's scope and traffic limits, and lists as misuse generating large volumes of low-signal reports, submitting unverified or fabricated claims without a proof of concept, and relying on hallucinated technical details. Misuse can lead to enforcement such as reputation impact and rate limiting. A May 2026 HackerOne blog post on AI-driven report volume describes a surge of more than 100% in report volume after new AI tools arrived in early 2026, and restates that researchers are accountable for what they submit whoever wrote the first draft.
- **Bugcrowd.** In March 2026 Bugcrowd announced policy changes against what it calls "sloptimism" (speculative or AI-generated submissions): permanent bans for submission farming, review of accounts with long runs of invalid reports, identity verification above an invalid-report threshold, and possible 30-day suspensions where automated or AI-generated submissions lacked validation. A May 2026 follow-up added identity verification for Managed Bug Bounty submissions, limits on in-flight reports for low-performing accounts, and human-verified browser submission.
- **The curl project.** Daniel Stenberg wrote about AI-generated bogus reports as early as January 2024. In May 2025 the project began requiring HackerOne reporters to say whether AI was used and announced it would ban reporters of "AI slop". On 26 January 2026 he announced the end of curl's bug bounty, effective 31 January 2026, citing the flood of low-quality and AI-generated reports; he wrote that the share of reports that turned out to be confirmed vulnerabilities fell below 5% from 2025, against more than 15% in earlier years. Since 1 March 2026 curl again takes security reports through HackerOne, with no bounty.

The practical lesson: an AI-assisted hunter who submits fewer, verified, well-written reports is exactly what these policies reward. Everything in this module is built to make that the default.

## Recap

- Strong reports are reproducible in minutes: precise title, minimal steps, redacted PoC, demonstrated impact, specific fix.
- The model proposes a CVSS vector with reasons; code computes the score; you decide the final vector.
- Redact before drafting so placeholders flow through; keep the mapping local; still read everything by eye.
- `report.py check` is a local gate; submission is always manual.
- HackerOne requires a human in the loop and holds you responsible; Bugcrowd penalizes unvalidated AI submissions; curl ended its bounty over AI slop. Verified, human-owned reports are the only kind worth sending.

## Check yourself

1. Why does `report.py` compute CVSS scores with a library instead of asking the model?
   <details><summary>Answer</summary>

   Score calculation is deterministic arithmetic (and in 4.0, a lookup-based algorithm) that language models get wrong easily. The model's useful contribution is the per-metric reasoning, which you can check. The library keeps the score consistent with the vector and catches invalid vectors.

   </details>
2. The LLM is local. Why redact before drafting at all?
   <details><summary>Answer</summary>

   The output does leave the machine: the draft is pasted into a platform, and logs or notes may be shared. Redacting first means the draft is built from placeholders and cannot leak tokens or other people's data by accident. It also keeps secrets out of the `llm_runs` log.

   </details>
3. Rescore the basket IDOR if the response exposed every user's delivery address and partial card number. What changes?
   <details><summary>Answer</summary>

   Confidentiality impact rises from Low to High: `CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N`, which is 6.5 (Medium) instead of 4.3. Confirm with `score()`. Whether C:H is justified depends on what you actually observed, and the program's own severity guidance may weigh personal data differently.

   </details>
4. Under HackerOne's Code of Conduct, can you run a hackbot that finds and submits reports by itself?
   <details><summary>Answer</summary>

   No. Hackbots must not operate fully autonomously; a human must investigate and validate findings before submission, and the operator remains responsible for everything submitted.

   </details>
5. Why is there no `submit` command in `report.py`?
   <details><summary>Answer</summary>

   Submission is the point where accountability transfers to you. Keeping it a manual action on the platform guarantees a human reads the final text, and it matches platform rules against fully autonomous submission.

   </details>

## Go deeper

- [FIRST CVSS v3.1](https://www.first.org/cvss/v3-1/) and [CVSS v4.0](https://www.first.org/cvss/v4-0/) — specifications, user guides and calculators
- [cvss on PyPI](https://pypi.org/project/cvss/) — the library used to compute 3.x and 4.0 scores
- [HackerOne Code of Conduct](https://www.hackerone.com/policies/code-of-conduct) — current rules for AI-assisted research and hackbots
- [Bugcrowd: policy changes to address AI slop submissions](https://www.bugcrowd.com/blog/bugcrowd-policy-changes-to-address-ai-slop-submissions/) — the March 2026 enforcement changes
- [Daniel Stenberg: The end of the curl bug-bounty](https://daniel.haxx.se/blog/2026/01/26/the-end-of-the-curl-bug-bounty/) — a maintainer's view of what AI slop costs
