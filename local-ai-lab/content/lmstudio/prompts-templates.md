# System prompts, templates, thinking and presets

Under the friendly chat bubbles, your model sees one long string of text stitched together by a template. This lesson opens that up: what a system prompt really is, how the prompt template turns a conversation into tokens, how Qwen3.5's thinking mode works and how to switch it off when LM Studio gives you no toggle, and how to package everything into reusable presets.

> [!GOALS]
> - See the exact text your model receives, and where the system prompt lands in it
> - Read and safely override the Prompt Template (Jinja or Manual)
> - Control Qwen3.5's thinking mode with or without a UI toggle, and cap its length
> - Create, apply, export and import presets, and know how they layer with per-model defaults
> - Start a personal library of four system prompts

## What the model actually sees

Your chat is a list of messages with roles: system, user, assistant (and tool). The model doesn't understand lists. It understands one token sequence. A **chat template** converts the list into that sequence using special marker tokens the model learned during training ([Base, instruct, thinking and chat templates](#/foundations/model-types) covers why).

For your model (Qwen3.5 uses the ChatML-style markers `<|im_start|>` and `<|im_end|>`), a chat with a system prompt and one question renders as:

```text
<|im_start|>system
You are a terse security assistant. Answer in at most three sentences.<|im_end|>
<|im_start|>user
What is SSRF?<|im_end|>
<|im_start|>assistant
<think>
```

The model then continues from that last line. Three things to notice:

1. **The system prompt is just text in a slot.** It's the first block, wrapped in `system` markers. The model gives it weight because it was trained on millions of examples where that slot held instructions to follow, not because anything enforces it.
2. **The template already opened the thinking block.** The last line is `<think>`: the model starts generating **inside** its reasoning. More on this below.
3. **Everything is re-sent every turn.** The system prompt and full history are part of the prompt for every new message, so a 2,000-token system prompt costs 2,000 tokens of context permanently (engines cache the unchanged prefix, so you don't pay the processing time twice).

> [!CONCEPT] System prompt
> Standing instructions placed in the template's system slot at the start of the context. Security analogy: it's a policy file the model was trained to respect, with no enforcement mechanism. There's no privilege separation between "system" text and "user" text; both are tokens in the same stream. That's exactly why prompt injection works, like SQL injection where data gets executed as instructions. See [Threat-model your own AI setup](#/security/ai-threat-model).

### See it for yourself

LM Studio can show the exact formatted input. In a terminal:

```bash
lms log stream --source model --filter input
```

Send a message in the chat UI and watch the fully rendered prompt appear, special tokens and all. This is the single best debugging tool for templates, system prompts and attachments. Add `--filter input,output` to see the raw model output too, including `</think>`.

### When tools are involved

If tools are defined (MCP plugins, or `tools` in an API call), Qwen3.5's template writes a `# Tools` section listing every tool's JSON schema, plus calling instructions, **into the system block before your system prompt**. Tool definitions can easily cost 1,000+ tokens. That's why agent contexts fill up faster than chats ([Structured output and tool calling](#/using/structured-output-tools)).

## The Prompt Template panel

By default LM Studio reads the chat template embedded in the GGUF file (the `tokenizer.chat_template` metadata; you can see it via My Models, right-click, **Show Raw Metadata**) and you never need to touch it.

### Where it is

- **Per model (recommended for overrides):** My Models, gear ("Edit model default config"), Prompt Template.
- **In the chat sidebar:** the **Prompt Template** box appears automatically only when a model ships without template info. To always show it, enable "Always show prompt template in Chat sidebar" in Settings (or right-click the sidebar and choose **Always Show Prompt Template**).
- **Load-time Chat Template:** newer builds using the llama.cpp Engine Protocol also show a load-time **Chat Template** field, described as the template "used by engine protocol for chat, tools, reasoning, and multimodal formatting". Changing it requires reloading the model.

### Jinja vs Manual

| Type | How it works | Use when |
|---|---|---|
| **Template (Jinja)** | A small program in the Jinja templating language that loops over messages and emits text. Can handle tools, images, thinking, edge cases. | Almost always. It's what the model's authors ship. |
| **Manual** | Six fields: Before System, After System, Before User, After User, Before Assistant, After Assistant, plus **Additional Stop Strings** | Old or custom models with simple formats, quick experiments |

For Qwen3.5, a Manual template would lose tool-call formatting, image placeholders and thinking control. Stay on Jinja.

### When to override

1. The metadata is missing, wrong, or LM Studio can't parse it (you'll see odd tokens like `<|im_end|>` printed in replies, or the model never stops).
2. You need behavior the template supports but LM Studio doesn't expose, such as forcing thinking off (below).
3. A specific use case needs different formatting.

> [!WARNING]
> A broken template doesn't crash anything; it quietly makes the model worse. Before editing, copy the original template into a file (`~/ai-lab/lmstudio/qwen35-original.jinja`) so you can restore it. Change one thing, then verify with `lms log stream --filter input`.

## Thinking mode

### What happens mechanically

Qwen3.5 is a **hybrid thinking** model: it can write a private chain of reasoning inside `<think> ... </think>` before the answer, or skip it. The switch is a template variable, `enable_thinking`. The end of the official Qwen3.5-9B template is:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

So:

- If nobody sets `enable_thinking`, the prompt ends with an open `<think>` and the model reasons first. The official card states that "Qwen3.5 models operate in thinking mode by default".
- If `enable_thinking` is false, the template writes an **empty, already-closed** think block, and the model goes straight to the answer.

You'll find conflicting claims online. Unsloth's guide says the small Qwen3.5 sizes have reasoning off by default, and that's true **for Unsloth's GGUFs**, which ship an edited template that defaults to off. The HauhauCS GGUF embeds the official Qwen template unchanged, so your model thinks unless something explicitly passes `enable_thinking = false`. Whether LM Studio passes it depends on whether it shows you a toggle (below). **Confirm empirically**: if replies come with a collapsible reasoning block, or with `</think>` in raw output, thinking is on.

> [!NOTE]
> Qwen3's `/think` and `/no_think` soft switches inside messages are **not** officially supported by Qwen3.5, per its model card. Don't rely on them.

### How LM Studio separates the thinking

Because the template opens `<think>` in the prompt, the model's generated text starts mid-reasoning and contains only the closing tag:

```text
The user asks about SSRF. Define it, give one example...
</think>

SSRF (server-side request forgery) is ...
```

LM Studio's **Reasoning Section Parsing** setting (start string `<think>`, end string `</think>`) splits this into a collapsible reasoning block plus the answer in the chat UI. The display option "Expand reasoning blocks by default" controls whether it opens.

Over the API:

- The app setting "When applicable, separate `reasoning_content` and `content` in API responses" moves reasoning into a separate field in OpenAI-style responses.
- The native `/api/v1/chat` endpoint returns output items of type `reasoning` and `message`, plus `reasoning_output_tokens` in its stats.
- If neither applies, your client receives `...reasoning...</think>answer` in `content`; split on `</think>` yourself.

**History handling.** The template drops the reasoning from earlier assistant turns when it re-renders history (keeping only final answers, as the Qwen card recommends). Good: old thinking doesn't eat your context. The side effect, the rendered history changing on every turn, is why [Context Checkpoints](#/lmstudio/load-settings) matter for this model.

### Turning thinking off (and on)

Work down this list until one works for your build and model.

**1. The UI toggle, if you have one.** For models LM Studio recognizes as reasoning models (typically via its catalog's `model.yaml`), you get a thinking toggle in the chat input area or an **Enable Thinking** field in the sidebar. For community GGUFs it often doesn't appear: bug reports in 2026 show third-party Qwen3.5 GGUFs missing the toggle while LM Studio's own catalog version has it. If yours has it, you're done.

**2. Reasoning Budget.** The sidebar's **Reasoning Budget** setting is described as "off for unrestricted, 0 for immediate end, N > 0 for token budget". Setting 0 ends reasoning immediately; setting, say, 2,048 caps how long the model may think before it's forced to answer, a great fix for over-thinking on simple questions. This depends on the engine recognizing the model's thinking format, so test it.

**3. From the CLI and API.**

```bash
lms chat qwen3.5-9b-uncensored-hauhaucs-aggressive --reasoning off
```

(`lms chat` also has a `/reasoning` command inside the session.) The native REST API accepts `"reasoning": "off"` on `/api/v1/chat`, but returns an error if LM Studio doesn't believe the model supports the setting.

**4. Edit the template (per-model default).** In My Models, gear, Prompt Template, change the final block so it always writes the empty think block:

```diff
 {%- if add_generation_prompt %}
     {{- '<|im_start|>assistant\n' }}
-    {%- if enable_thinking is defined and enable_thinking is false %}
-        {{- '<think>\n\n</think>\n\n' }}
-    {%- else %}
-        {{- '<think>\n' }}
-    {%- endif %}
+    {{- '<think>\n\n</think>\n\n' }}
 {%- endif %}
```

This forces non-thinking mode everywhere the model is used (chat, server, scripts). To switch back, restore the original. It's blunt, but it works on any build because it's just the template the model already uses. If your build shows the load-time **Chat Template** field (Engine Protocol builds), that's the one the engine actually formats with: make the edit there and reload. Confirm with `lms log stream` either way.

**5. A virtual model with a real toggle (cleanest long-term).** LM Studio's `model.yaml` format lets you wrap an existing model with settings and **custom fields** without copying the weights. The documented way to add a thinking switch is a custom field that sets the Jinja variable:

```yaml title="model.yaml (excerpt)"
customFields:
  - key: enableThinking
    displayName: Enable Thinking
    description: Controls whether the model will think before replying
    type: boolean
    defaultValue: false
    effects:
      - type: setJinjaVariable
        variable: enable_thinking
config:
  operation:
    fields:
      - key: llm.prediction.topKSampling
        value: 20
```

The easiest route is My Models, **Create Virtual Model**, choosing your GGUF as the base; then right-click the new entry, Reveal in Finder, and add the `customFields` block to the generated `model.yaml`. You get an **Enable Thinking** switch in the sidebar for that virtual model. The model.yaml spec is marked "Draft" in LM Studio's docs, so check [the model.yaml docs](https://lmstudio.ai/docs/app/modelyaml) against your version.

> [!TIP]
> Many people end up with two entries: the original model (thinking) and a virtual "no-think" model, each with matching default sampling. Picking a model then picks the mode, in the UI and in API calls alike.

## Presets

A **preset** is a named bundle of a system prompt plus any inference parameters (temperature, top-p, stop strings, structured output, and so on). It's how you switch between "terse code reviewer" and "patient teacher" in one click.

### Create, save, apply

The preset manager is at the top of the chat's right sidebar.

1. Configure a chat the way you like (system prompt plus sidebar settings).
2. Open the preset dropdown, **Save As New...**, name it. (`+ New Preset` starts from empty.)
3. Later edits show as unsaved changes: **Save** commits them (hold the modifier key shown in the tooltip to pick which fields), **Discard Unsaved** reverts.
4. Apply a preset to any chat by selecting it; **Deselect Preset** detaches it.

If an incoming preset conflicts with values you've changed, LM Studio asks which to keep ("Which values to keep?").

### What's in a preset (and what isn't)

- Included: the system prompt and every parameter under the sidebar's settings.
- **Load parameters** (context length, KV cache type...) are not included by default. An experimental app setting, "Enable model load configuration support in presets", lets presets carry load-time configuration; such values are model-wide and need a reload. Without it, put load settings in **per-model defaults** instead.

### How settings layer

Several sources can set the same parameter. As LM Studio's SDK describes its configuration layers, the more specific source wins, roughly:

```text
API request parameters
  > this chat's own changes
    > the selected preset
      > your per-model defaults (My Models, gear)
        > model.yaml / virtual model defaults
          > LM Studio's built-in defaults
```

Practical rule: **load settings** in per-model defaults, **behavior** (system prompt and sampling) in presets, and API calls override both.

### Files, export, import, sharing

- Presets are JSON files in `~/.lmstudio/config-presets/`. From the preset's `•••` menu: **Export File**, **Reveal in Finder**, **Copy Preset Identifier**.
- **Import** accepts `preset.json` or `.tar.gz` files, or a LM Studio Hub URL ("Import from URL...").
- **Publish...** uploads a preset to the LM Studio Hub. Published presets are **public**: anyone on the internet can see them. Hub presets you pull land in `~/.lmstudio/hub/`; **Pull Latest** and **Push Changes to Hub** keep them in sync.
- CLI: `lms get {author}/{preset-name}` downloads a Hub preset (for example `lms get neil/qwen3-thinking`).

> [!DANGER]
> Never publish a preset whose system prompt contains client names, scope details, internal hostnames or anything else from an engagement. Presets are easy to share and easy to forget about; treat them like any other file that might leak.

The file format is shown in [Recommended configs](#/lmstudio/presets-for-you), along with six ready-made presets for your Mac.

## Conversation management and the context indicator

- Each chat remembers its own settings and preset; duplicating a chat copies them (and any attached files).
- **Branch chat after this message** forks a conversation so you can try another path without disturbing the original. **Clear All** empties a chat but keeps its settings.
- The input area shows **Tokens: used/total** and "Context is N% full". The per-message input token count doesn't include file attachments, so the overall indicator is the one to watch.
- When the context fills, the **Context Overflow** policy applies ([Sampling settings](#/lmstudio/sampling-settings)). For long work, start a fresh chat with a summary rather than letting truncation decide what the model forgets.

## A small system prompt library

Four prompts to save as presets. They're deliberately short: on a 9B model, a clear 100-token prompt beats a rambling 1,500-token one, and it leaves context for actual work. Deeper security personas live in [Security personas and system prompts](#/security/system-prompts); prompting technique is in [Prompt engineering for small local models](#/using/prompt-engineering).

### 1. Precise technical assistant

```text
You are a precise technical assistant for an experienced engineer.
- Answer the question asked; skip background the user already knows.
- When you state a fact about a tool, flag, API or version, say how confident you are. If unsure, say so and say how to verify (man page, --help, official docs).
- Prefer exact commands, file paths and code over prose. Code must be complete and runnable.
- If the request is ambiguous, state your assumption in one line and proceed.
- Never invent CVE numbers, flags, package names or URLs.
```

Pair with: non-thinking for quick questions, thinking for design questions.

### 2. Terse expert

```text
You are a senior engineer replying in a busy chat channel.
Rules: at most 5 sentences or 10 lines of code. No preamble, no summary, no apologies.
If the question needs a longer answer, give the shortest correct answer and end with: "Want details?"
```

Pair with: non-thinking mode, Temperature 0.7, Top P 0.8.

### 3. Teacher

```text
You are a patient teacher for a professional who is new to this topic but strong in security engineering.
- Explain each new term before using it, in one plain sentence.
- Go from a concrete example to the general rule.
- Use analogies from security and networking when they genuinely fit.
- End with one short question that checks understanding, and wait for the answer.
```

Pair with: thinking on, Temperature 1.0 (Qwen's general thinking profile).

### 4. Pentest copilot

```text
You are a penetration-testing copilot assisting an authorized security tester.
Context: all work is within a written scope (bug bounty program rules or a signed engagement). The tester will paste scope details when relevant.
- Before suggesting active testing against a target, ask whether it is in scope if that hasn't been established.
- Prefer low-impact techniques first; call out anything that could cause downtime, data modification or account lockout.
- For each finding, separate: observed evidence, hypothesis, and how to verify.
- Give exact commands with safe defaults (rate limits, timeouts, no destructive flags).
- Mark anything you are not sure about as UNVERIFIED. Do not invent tool flags or CVE IDs.
```

Pair with: thinking on for analysis, Presence Penalty 0 when generating commands or code. Remember that an uncensored model won't refuse out-of-scope requests; the scope discipline has to come from you and your tooling ([Why local AI for security work](#/security/why-local)).

## Lab: see, toggle, package

> [!LAB] Templates, thinking and presets on your own model
> 1. In one terminal: `lms log stream --source model --filter input,output`
> 2. In LM Studio, new chat, no system prompt. Ask "What is SSRF?" Find in the log: the `<|im_start|>` markers, where the assistant turn starts, and whether the prompt ends with `<think>`.
> 3. Add the Terse expert system prompt. Ask again. Confirm in the log exactly where your system prompt landed.
> 4. Thinking check: is there a reasoning block in the reply? Note the reply's token count and time.
> 5. Try the toggles in order: (a) any thinking toggle in the UI, (b) Reasoning Budget 0, (c) Reasoning Budget 512 on a harder question, (d) `lms chat <your-model-key> --reasoning off -p "What is SSRF?"`. Record which worked on your build.
> 6. If none worked: save the original template to `~/ai-lab/lmstudio/qwen35-original.jinja`, apply the diff above as a per-model default, reload, and confirm in the log that the prompt now ends with an empty `<think>\n\n</think>`. Compare answer length and speed with step 4.
> 7. Save the four system prompts as presets named `tech-precise`, `terse`, `teacher`, `pentest-copilot`, each with the paired sampling settings.
> 8. Export `pentest-copilot` (`•••`, Export File) to `~/ai-lab/lmstudio/presets/` and open the JSON. Find your system prompt and temperature in it.
> 9. Optional: create a virtual "no-think" model with an Enable Thinking custom field and confirm the switch appears in the sidebar.

## Recap
- The chat template turns messages into one token stream; the system prompt is just text in the first slot, with no enforcement behind it.
- `lms log stream --source model --filter input` shows exactly what the model receives. Use it whenever behavior surprises you.
- LM Studio reads the template from the GGUF; override it per model, preferably in Jinja, and keep the original.
- Qwen3.5's template opens `<think>` unless `enable_thinking` is false; LM Studio splits reasoning using the `<think>`/`</think>` parsing strings; earlier turns' reasoning is dropped from history.
- To control thinking without a UI toggle: Reasoning Budget, `lms chat --reasoning`, `"reasoning"` in `/api/v1/chat`, a template edit, or a virtual model with a `setJinjaVariable` custom field.
- Presets bundle system prompt and inference settings; load settings belong in per-model defaults; presets are JSON in `~/.lmstudio/config-presets/` and published ones are public.

## Check yourself
1. Why can a user message override instructions in the system prompt?
   <details><summary>Answer</summary>

   Both are just tokens in the same sequence. The model weights the system slot because of training, but there's no privilege boundary or enforcement, so persuasive or conflicting user text can win. This is the root of prompt injection.

   </details>
2. Your API client shows the model's whole reasoning plus the answer in `content`, with a `</think>` in the middle but no `<think>`. Why is the opening tag missing?
   <details><summary>Answer</summary>

   The Qwen3.5 template writes the opening `<think>` into the prompt (generation prompt), so the model's output starts already inside the reasoning and only produces the closing tag. Enable the separate `reasoning_content` setting or split on `</think>`.

   </details>
3. You edited the template to force non-thinking mode. Which other places did that change affect?
   <details><summary>Answer</summary>

   Every place the model is loaded with its per-model defaults: the chat UI, the local server (API clients), `lms load`, and the SDKs.

   </details>
4. You want temperature 0.6 for code and 1.0 for brainstorming, but the same 64K context for both. Where does each setting go?
   <details><summary>Answer</summary>

   Context length (a load setting) goes in the per-model default. The temperatures go in two presets, applied per chat.

   </details>
5. What is the risk of publishing a preset to the LM Studio Hub, and what should never be in one?
   <details><summary>Answer</summary>

   Published presets are public. Never include client names, scope details, hostnames, credentials or any engagement data in a system prompt you might publish.

   </details>

## Go deeper
- [LM Studio: Prompt Template](https://lmstudio.ai/docs/app/advanced/prompt-template) — Jinja vs manual templates and where to override them
- [LM Studio: Config Presets](https://lmstudio.ai/docs/app/presets) — saving, importing and publishing presets
- [LM Studio: model.yaml](https://lmstudio.ai/docs/app/modelyaml) — virtual models, baked-in config and custom fields such as Enable Thinking
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) — thinking vs non-thinking usage and best practices
- [Jinja template designer documentation](https://jinja.palletsprojects.com/en/stable/templates/) — the template language itself
