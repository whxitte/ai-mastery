# Evaluate your fine-tune

A fine-tune that "feels better" in three chats is not a result. This lesson turns your held-out test set into a repeatable scorecard: automatic checks a script can verify, an LLM judge with a rubric (and an honest account of its biases), a blind A/B sheet for your own judgment, a regression set that catches forgetting, and real security benchmarks for a second opinion.

> [!GOALS]
> - Explain why evaluation has to come before you replace your daily model
> - Score outputs automatically: exact match, JSON validity, schema compliance, invented CVEs, refusals
> - Use an LLM-as-judge with a rubric and order swapping, and know its biases
> - Run blind A/B comparisons and a general-capability regression set
> - Build `eval.py`, which compares base and fine-tuned models through LM Studio and writes a CSV and Markdown scorecard
> - Recognise the signs of overfitting in evaluation results

## Why evals

Training loss tells you the model finds your reference answers more predictable. It does not tell you whether its *own* answers are better, whether it still writes decent Python, or whether it started inventing CVE IDs. In security work a confident wrong answer is expensive (see [Trust but verify: hallucinations](#/security/hallucination-verification)), so treat a fine-tune like any other change to a production tool: it ships only after it passes tests that existed *before* you trained it.

Three rules make evaluation trustworthy:

1. **Held-out data.** `test.jsonl` from [Build a cybersecurity dataset](#/finetune/dataset) was split by source document and never trained on. Do not tune hyperparameters against it repeatedly either; use `valid.jsonl` for that and look at `test.jsonl` at the end.
2. **Same conditions for every candidate.** Same system prompt, sampling settings, thinking mode and max tokens. Change one variable: the weights.
3. **Compare against the base.** A score means nothing alone. 70% schema compliance is great if the base gets 20%, and a regression if the base gets 90%.

## Automatic checks

Anything a script can verify, verify with a script. It is cheap, repeatable and unbiased.

| Check | What it catches | Applies to |
|---|---|---|
| **Exact match** (after normalising whitespace and case) | Whether the model reproduces a short, fixed answer | Labels, severity, yes/no; rarely useful for prose |
| **JSON validity** | Output that is not parseable JSON (extra prose, trailing text, broken quotes) | Triage, any tool-facing output |
| **Schema compliance** | Wrong or missing keys, invalid enum values, wrong types | Triage (exact key set, severity in `info`..`critical`) |
| **Field match** | Right format, wrong judgment | Severity versus the reference |
| **Token F1** | Rough word overlap with the reference answer | Prose; a weak signal, use for trends only |
| **Invented CVE** | CVE IDs in the output that appear nowhere in the input or reference | Everything: your hallucination tripwire |
| **Refusal pattern** | "I can't help with that" and friends | Everything; matters doubly for an uncensored base |
| **Regex must-match** | A required phrase or structure in a regression item | General-capability set |

> [!CONCEPT] Schema compliance
> Not just "is it JSON" but "is it *the* JSON your pipeline expects": exact keys, correct types, allowed values. A triage object with `"sev": "HIGH"` is valid JSON and still breaks your agent. Structured-output constraints (see [Structured output and tool calling](#/using/structured-output-tools)) can force this at inference time; measuring it without them tells you whether the model actually learned it.

## LLM-as-judge

Report prose, JS analysis and explanations cannot be scored by regex. The standard approach is to ask a model to grade them.

> [!CONCEPT] LLM-as-judge
> A model is given a rubric, the request, optionally a reference answer, and one or two candidate answers, and returns a score or a preference. **Pairwise** judging ("which of A and B is better?") is usually more reliable than absolute 1–10 scores.

Judges have documented biases (Zheng et al., 2023):

| Bias | What happens | Mitigation in `eval.py` |
|---|---|---|
| **Position bias** | Prefers whichever answer is shown first (or second) | Judge every pair twice with the order swapped; inconsistent verdicts count as a tie |
| **Verbosity bias** | Prefers longer answers | The rubric says not to reward length; the scorecard reports `length_ratio` so you can spot it |
| **Self-preference** | Prefers answers in its own style or from its own family | Use a different or stronger judge where possible; confirm with your blind review |
| **Limited competence** | A 9B judge misses subtle technical errors | Give it the reference answer; read its reasons; trust automatic checks over it for facts |

On 16 GB your judge will usually be your 9B. That is a real limitation when you compare the 9B base against the 9B fine-tune (same family, same size). Treat its verdicts as a hint and your own blind review as the decision.

## Blind A/B: you are the gold standard

You are the domain expert. The most valuable 30 minutes in this module is reading 30 pairs of answers **without knowing which model wrote which** and picking the better one. `eval.py` writes `blind_review.md` with shuffled A/B pairs and keeps the answer key in a separate `blind_key.json`. Do the review, then open the key.

> [!CONCEPT] Blind comparison
> Hiding which system produced each output so expectations cannot bias the judgment, for the same reason a report peer review works better when the reviewer does not know whose report it is.

## A regression set for forgetting

Recall [catastrophic forgetting](#/finetune/concepts): a narrow fine-tune can damage general skills. Keep a small fixed set of general prompts in the same JSONL format, each with an optional `must_match` regex, and run it every time:

```json
{"messages": [{"role": "user", "content": "Write a Python function is_prime(n) that returns True for primes."}, {"role": "assistant", "content": "def is_prime(n): ..."}], "task": "general", "must_match": "def is_prime\\("}
{"messages": [{"role": "user", "content": "Explain the TCP three-way handshake in three sentences."}, {"role": "assistant", "content": "The client sends SYN..."}], "task": "general", "must_match": "SYN.*ACK"}
{"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}, {"role": "assistant", "content": "391"}], "task": "general", "must_match": "\\b391\\b"}
```

Thirty to fifty such items covering coding, explanation, arithmetic, summarising, instruction following (word limits, bullet counts) and any other language you care about are enough to notice a drop. Pass the file with `--extra regression.jsonl`. For standard academic benchmarks, EleutherAI's lm-evaluation-harness can target an OpenAI-compatible local server; check its documentation for the current model-type names and flags.

## Security benchmarks that exist

Public benchmarks give an outside reference point. Use small subsets on 16 GB, check each licence, and never train on them.

| Benchmark | What it tests | Notes |
|---|---|---|
| [CTIBench](https://github.com/xashru/cti-bench) (NeurIPS 2024 Datasets and Benchmarks; data on Hugging Face as `AI4Sec/cti-bench`) | Cyber threat intelligence: multiple-choice knowledge (CTI-MCQ), CVE-to-CWE root-cause mapping (CTI-RCM), CVSS prediction (CTI-VSP), ATT&CK technique extraction (CTI-ATE), threat-actor attribution (CTI-TAA) | CC BY-NC-SA 4.0. RCM and VSP are close to real triage work |
| [CyberMetric](https://github.com/cybermetric/CyberMetric) | Multiple-choice security knowledge in sets of 80, 500, 2,000 and 10,000 questions, human-validated | Ships an evaluator script; the 80 and 500 sets run quickly on a laptop |
| [SecBench](https://huggingface.co/datasets/secbench-hf/SecBench) | Multiple-choice and short-answer questions across security domains | Mostly Chinese; about 3,000 questions released publicly |
| [Cybench](https://cybench.github.io/) | 40 professional CTF tasks solved by an agent in a sandbox | Agentic and heavy (Docker); tests your agent stack more than the raw model |
| [CyberSecEval (PurpleLlama)](https://github.com/meta-llama/PurpleLlama) | Security risks of models: insecure code suggestions, helpfulness to attacks, prompt injection | A safety benchmark; interesting for an uncensored model, but it measures different things than your task skills |

> [!WARNING]
> Public benchmark questions may have leaked into pretraining data, small models score low on most of them, and multiple-choice knowledge is not the same as doing the job. A fine-tune for report style should not move CyberMetric much; if it drops sharply, that is forgetting. Your own test set stays the primary metric.

## Lab: eval.py

> [!LAB] Score base versus fine-tuned
> 1. `cd ~/ai-lab/finetune && source .venv/bin/activate && pip install openai`
> 2. Save the script below as `eval.py`. Optionally write `regression.jsonl` as shown above.
> 3. Load the **base** model in LM Studio (only one 9B at a time on 16 GB), thinking off, and generate its outputs:
>    `python eval.py --only base --candidate base http://localhost:1234/v1 qwen3.5-9b-uncensored-hauhaucs-aggressive --extra regression.jsonl`
> 4. Unload it (`lms unload --all`), load your fine-tune, and generate:
>    `python eval.py --only tuned --candidate tuned http://localhost:1234/v1 qwen35-9b-sec --extra regression.jsonl`
> 5. Score both from the cache, optionally with a judge (load the judge model first):
>    `python eval.py --candidate base http://localhost:1234/v1 qwen3.5-9b-uncensored-hauhaucs-aggressive --candidate tuned http://localhost:1234/v1 qwen35-9b-sec --extra regression.jsonl --judge http://localhost:1234/v1 qwen3.5-9b-uncensored-hauhaucs-aggressive`
> 6. Read `eval/scorecard.md`, then do the blind review in `eval/blind_review.md` before opening `blind_key.json`.
>
> Use your exact model ids from `lms ls`. For the MLX 2B lab, point a candidate at `mlx_lm.server` (`http://localhost:8080/v1`) and use the model name it reports at `/v1/models`.

Generation uses temperature 0 by default so reruns are comparable. Outputs are cached per candidate; add `--regen` after you change a model.

```python title="eval.py"
#!/usr/bin/env python3
"""Compare a base model and a fine-tuned model on your held-out test set.

Each candidate is an OpenAI-compatible endpoint (LM Studio, mlx_lm.server, llama-server).
Outputs are cached per candidate in ~/ai-lab/finetune/eval/outputs/<name>.jsonl, so on 16 GB
you can generate with one model loaded at a time and score everything at the end.

Example:
  python eval.py --only base --candidate base http://localhost:1234/v1 qwen3.5-9b-uncensored-hauhaucs-aggressive
  python eval.py --only tuned --candidate tuned http://localhost:1234/v1 qwen35-9b-sec
  python eval.py --candidate base http://localhost:1234/v1 qwen3.5-9b-uncensored-hauhaucs-aggressive \
                 --candidate tuned http://localhost:1234/v1 qwen35-9b-sec
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

ROOT = Path.home() / "ai-lab" / "finetune"
TRIAGE_KEYS = {"asset", "category", "severity", "confidence", "evidence", "next_steps",
               "in_scope_check_required"}
SEVERITIES = {"info", "low", "medium", "high", "critical"}
CONFIDENCES = {"low", "medium", "high"}
THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}")
REFUSAL_RE = re.compile(
    r"(?i)\b(i can(?:'|no)t (?:help|assist|provide)|i(?:'m| am) (?:unable|not able) to|"
    r"i won't (?:help|provide)|as an ai\b)")

JUDGE_PROMPT = """You are grading two answers to the same request from a security engineer.
Rubric, in priority order:
1. Correctness: no invented facts, CVE IDs, versions or results; claims supported by the input.
2. Instruction following: required format (for example JSON only) and every requested part present.
3. Professional judgment: scope and authorization awareness; flags what needs manual verification.
4. Usefulness and concision. Do NOT prefer an answer because it is longer.

REQUEST:
{request}

REFERENCE ANSWER (written by the user, for guidance only):
{reference}

ANSWER A:
{a}

ANSWER B:
{b}

Reply with JSON: {{"winner": "A" or "B" or "tie", "reason": "one sentence"}}"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]},
                   "reason": {"type": "string"}},
    "required": ["winner", "reason"],
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(name: str, url: str, model: str, examples: list[dict], out_dir: Path,
             temperature: float, max_tokens: int, regen: bool) -> list[dict]:
    path = out_dir / f"{name}.jsonl"
    if path.exists() and not regen:
        print(f"[{name}] using cached outputs from {path}")
        return load_jsonl(path)
    client = OpenAI(base_url=url, api_key="lm-studio")
    rows = []
    for i, rec in enumerate(examples):
        prompt = rec["messages"][:-1]                # everything except the reference answer
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(model=model, messages=prompt,
                                                  temperature=temperature, max_tokens=max_tokens)
            text = resp.choices[0].message.content or ""
        except Exception as exc:                     # keep going; the error is scored as a miss
            text = f"[ERROR] {exc}"
        rows.append({"id": i, "output": THINK_RE.sub("", text).strip(),
                     "seconds": round(time.perf_counter() - t0, 2)})
        print(f"[{name}] {i + 1}/{len(examples)} ({rows[-1]['seconds']}s)")
    write_jsonl(path, rows)
    return rows


def parse_json(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def schema_ok(obj) -> bool:
    return (isinstance(obj, dict) and set(obj) == TRIAGE_KEYS
            and obj["severity"] in SEVERITIES and obj["confidence"] in CONFIDENCES
            and isinstance(obj["evidence"], list) and isinstance(obj["next_steps"], list)
            and isinstance(obj["in_scope_check_required"], bool))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def token_f1(pred: str, ref: str) -> float:
    p, r = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    same = sum((Counter(p) & Counter(r)).values())
    if not p or not r or same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def score(rec: dict, output: str) -> dict:
    prompt_text = " ".join(m["content"] for m in rec["messages"][:-1])
    ref = rec["messages"][-1]["content"]
    task = rec.get("task", "unknown")
    known_cves = set(CVE_RE.findall(prompt_text + " " + ref))
    s = {
        "task": task,
        "exact": int(normalise(output) == normalise(ref)),
        "f1": round(token_f1(output, ref), 3),
        "refusal": int(bool(REFUSAL_RE.search(output))),
        "invented_cve": int(bool(set(CVE_RE.findall(output)) - known_cves)),
        "length_ratio": round(len(output) / max(1, len(ref)), 2),
    }
    if task == "triage":
        obj, ref_obj = parse_json(output), parse_json(ref)
        s["json_valid"] = int(obj is not None)
        s["schema_ok"] = int(schema_ok(obj))
        s["severity_match"] = int(isinstance(obj, dict) and isinstance(ref_obj, dict)
                                  and obj.get("severity") == ref_obj.get("severity"))
    if "must_match" in rec:                         # optional regex check (regression set)
        s["regex_pass"] = int(bool(re.search(rec["must_match"], output, re.I | re.S)))
    return s


def judge_once(client: OpenAI, model: str, request: str, ref: str, a: str, b: str) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=400,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            request=request, reference=ref, a=a, b=b)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
    )
    obj = parse_json(THINK_RE.sub("", resp.choices[0].message.content or ""))
    return obj.get("winner", "tie") if isinstance(obj, dict) else "tie"


def judge_pair(client: OpenAI, model: str, rec: dict, out1: str, out2: str) -> int:
    """Judge twice with the order swapped. 1 = first candidate wins, 2 = second, 0 = tie."""
    request = "\n\n".join(f"[{m['role']}] {m['content']}" for m in rec["messages"][:-1])
    ref = rec["messages"][-1]["content"]
    v1 = judge_once(client, model, request, ref, out1, out2)  # candidate 1 shown as A
    v2 = judge_once(client, model, request, ref, out2, out1)  # candidate 1 shown as B
    w1, w2 = {"A": 1, "B": 2}.get(v1, 0), {"A": 2, "B": 1}.get(v2, 0)
    return w1 if w1 == w2 else 0                              # inconsistent verdicts count as a tie


def write_blind_review(examples: list[dict], outputs: dict, names: list[str],
                       out_dir: Path, n: int, seed: int) -> None:
    """Shuffled A/B sheet for human review, with the answer key in a separate file."""
    rng = random.Random(seed)
    ids = rng.sample(range(len(examples)), min(n, len(examples)))
    key = {}
    lines = ["# Blind review\n\n", "Write A, B or tie under each item. Open blind_key.json only afterwards.\n"]
    for k, i in enumerate(ids, 1):
        pair = names[:2]
        rng.shuffle(pair)
        key[k] = {"A": pair[0], "B": pair[1], "example": i}
        request = examples[i]["messages"][-2]["content"]
        lines += [f"\n## Item {k}\n\n", f"Request:\n\n```text\n{request}\n```\n\n",
                  f"Answer A:\n\n```text\n{outputs[pair[0]][i]['output']}\n```\n\n",
                  f"Answer B:\n\n```text\n{outputs[pair[1]][i]['output']}\n```\n\n",
                  "Verdict:\n"]
    (out_dir / "blind_review.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")


def mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if key in r]
    return round(statistics.mean(vals), 3) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", nargs=3, action="append", required=True,
                    metavar=("NAME", "BASE_URL", "MODEL"))
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test.jsonl")
    ap.add_argument("--extra", type=Path, help="optional regression set in the same JSONL format")
    ap.add_argument("--out", type=Path, default=ROOT / "eval")
    ap.add_argument("--only", help="only generate outputs for this candidate, then stop")
    ap.add_argument("--regen", action="store_true", help="ignore cached outputs")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--judge", nargs=2, metavar=("BASE_URL", "MODEL"))
    ap.add_argument("--blind", type=int, default=30, help="items in the human blind-review sheet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = load_jsonl(args.test) + (load_jsonl(args.extra) if args.extra else [])
    out_dir = args.out
    outputs = {}
    for name, url, model in args.candidate:
        if args.only and name != args.only:
            continue
        outputs[name] = generate(name, url, model, examples, out_dir / "outputs",
                                 args.temperature, args.max_tokens, args.regen)
    if args.only:
        return

    names = [c[0] for c in args.candidate]
    results = []
    for name in names:
        for rec, row in zip(examples, outputs[name]):
            results.append({"model": name, "id": row["id"], "seconds": row["seconds"],
                            **score(rec, row["output"]), "output": row["output"]})

    wins = Counter()
    if args.judge and len(names) >= 2:
        judge_client = OpenAI(base_url=args.judge[0], api_key="lm-studio")
        for i, rec in enumerate(examples):
            w = judge_pair(judge_client, args.judge[1], rec,
                           outputs[names[0]][i]["output"], outputs[names[1]][i]["output"])
            wins[{1: names[0], 2: names[1], 0: "tie"}[w]] += 1
            print(f"judge {i + 1}/{len(examples)}")

    fields = sorted({k for r in results for k in r} - {"output"}) + ["output"]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    metrics = ["f1", "exact", "refusal", "invented_cve", "json_valid", "schema_ok",
               "severity_match", "regex_pass", "length_ratio", "seconds"]
    md = ["# Fine-tune scorecard\n\n", f"Test items: {len(examples)}\n"]
    for task in ["ALL"] + sorted({r["task"] for r in results}):
        md.append(f"\n## {task}\n\n| model | n | " + " | ".join(metrics) + " |\n")
        md.append("|---" * (len(metrics) + 2) + "|\n")
        for name in names:
            rows = [r for r in results
                    if r["model"] == name and (task == "ALL" or r["task"] == task)]
            vals = [mean(rows, m) for m in metrics]
            md.append(f"| {name} | {len(rows)} | "
                      + " | ".join("" if v is None else str(v) for v in vals) + " |\n")
    if wins:
        total = sum(wins.values())
        md.append("\n## Judge (order-swapped; inconsistent verdicts count as tie)\n\n")
        for k in names[:2] + ["tie"]:
            md.append(f"- {k}: {wins[k]} ({wins[k] / total:.0%})\n")
    (out_dir / "scorecard.md").write_text("".join(md), encoding="utf-8")
    if len(names) >= 2:
        write_blind_review(examples, outputs, names, out_dir, args.blind, args.seed)
    print("".join(md))
    print(f"Wrote {out_dir / 'results.csv'}, {out_dir / 'scorecard.md'} and the blind-review sheet")


if __name__ == "__main__":
    main()
```

A scorecard section looks like this (numbers are illustrative):

```text
## triage

| model | n  | f1   | exact | refusal | invented_cve | json_valid | schema_ok | severity_match | ...
| base  | 12 | 0.41 | 0     | 0.0     | 0.083        | 0.667      | 0.25      | 0.5            | ...
| tuned | 12 | 0.63 | 0     | 0.0     | 0.0          | 1.0        | 0.917     | 0.667          | ...
```

### How much difference is real?

With small test sets, noise is large. The standard error of a proportion is `sqrt(p·(1−p)/n)`. At p = 0.5 and n = 50 that is about 0.07, so a 95% interval is roughly ±14 percentage points. A 60% versus 55% judge win rate on 50 items is noise; 92% versus 25% schema compliance is not. Grow the test set over time and trust large, consistent differences.

## Signs of over-fitting

| Symptom in evals | What it means |
|---|---|
| Test loss improved, but outputs copy phrases, hostnames or findings from training writeups | Memorisation of specific examples |
| JSON or report formatting appears in answers to unrelated regression prompts | The habit generalised too far: too many epochs or too narrow data |
| Regression pass rate drops versus base | Catastrophic forgetting |
| `invented_cve` rises | The model learned to *sound* specific; check the data for unsupported details |
| Near-identical answers across different inputs | Collapse onto a template; reduce epochs, add diversity |
| Strong on `valid.jsonl`, weak on `test.jsonl` | Hyperparameters were tuned against the validation set too many times |
| New disclaimers or refusals on legitimate prompts | Teacher data reintroduced them, or the adapter pulled behaviour back toward the official base |

When you see these, go back to the checkpoint with the lowest validation loss, train fewer steps or at a lower learning rate, and fix the data before anything else.

## Recap

- Evaluate on held-out data, under identical conditions, always against the base model.
- Automatic checks first: JSON validity, schema compliance, field match, invented CVEs, refusals, regex checks.
- LLM judges are useful but biased (position, verbosity, self-preference): swap the order, use a rubric and a reference, and confirm with a blind human review.
- Keep a general regression set to catch forgetting; use public security benchmarks (CTIBench, CyberMetric, SecBench, Cybench, CyberSecEval) as secondary signals and never train on them.
- Small test sets are noisy, about ±14 points at n = 50. Look for large, consistent gains.

## Check yourself

1. Why does `eval.py` ask the judge twice with A and B swapped?
   <details><summary>Answer</summary>

   To cancel position bias. If the judge prefers the same underlying answer in both orders, the verdict counts; if it flips with the order, the preference was about position, so it is recorded as a tie.

   </details>
2. Your fine-tune's triage `json_valid` is 1.0 but `schema_ok` is 0.4. What is wrong and where do you look?
   <details><summary>Answer</summary>

   It emits valid JSON with the wrong keys, types or enum values. Look at the failing rows in `results.csv`, then at the training data: inconsistent keys or severity spellings in the dataset are the usual cause.

   </details>
3. The tuned model wins 58% of judged pairs on 40 items. Is that a clear win?
   <details><summary>Answer</summary>

   No. With n = 40 the standard error is about 0.08, so roughly ±16 points at 95%; 58% is consistent with no difference. Check the automatic metrics and your blind review, and grow the test set.

   </details>
4. Why is training on CyberMetric questions a mistake even though they are security content?
   <details><summary>Answer</summary>

   It contaminates the benchmark: the score would then measure memorisation of those questions rather than ability, and you lose an independent yardstick.

   </details>
5. Name two eval signs of catastrophic forgetting.
   <details><summary>Answer</summary>

   A drop in the regression set's pass rate versus base, and task formatting (JSON, report headings) leaking into unrelated answers. A sharp fall on a general security knowledge benchmark is another.

   </details>

## Go deeper
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685) — the reference for judge biases and how to measure judge agreement
- [CTIBench (Alam et al., 2024)](https://arxiv.org/abs/2406.07599) — design of the CTI tasks; a good template for your own triage evals
- [CyberMetric (Tihanyi et al., 2024)](https://arxiv.org/abs/2402.07688) — how the question sets were built and validated
- [Cybench (2024)](https://arxiv.org/abs/2408.08926) — evaluating agents on professional CTF tasks
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — standard benchmarks for general-capability regression checks
