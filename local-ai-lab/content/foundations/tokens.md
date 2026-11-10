# Tokens and tokenization

Your model never sees letters or words. It sees **tokens**: chunks of text mapped to id numbers by a tokenizer. Tokens are the unit for everything you'll measure from now on (context length, memory, speed), and they explain some of the strangest things models do, from botched arithmetic to failing to count the r's in "strawberry". This lesson shows how tokenization works, step by step, and then has you run the real Qwen3.5 tokenizer on security-flavoured text.

> [!GOALS]
> - Explain why models read tokens instead of characters or words
> - Walk through byte-pair encoding (BPE) by hand on a tiny example
> - Describe Qwen3.5's vocabulary, token ids and special tokens
> - Predict how English, code, other languages, numbers, whitespace and encoded blobs tokenize, and why it matters
> - Tokenize real text with the actual Qwen3.5 tokenizer and compare token counts

## Why not characters, why not words?

The model needs text as a list of numbers ([What is an LLM, really?](#/foundations/what-is-an-llm)). There are two obvious ways to do that, and both are bad.

**One number per character.** Simple, and nothing is ever unknown. But sequences become very long: a 4,000-character HTTP response becomes 4,000 positions. Every position costs memory and compute (the attention mechanism in [Inside the transformer](#/foundations/transformer) compares positions with each other), and the model has to learn from scratch that `c`, `u`, `r`, `l` spell a tool name, over and over.

**One number per word.** Sequences are short, but the vocabulary explodes. English alone has hundreds of thousands of word forms. Add every language, every variable name in every codebase, every typo, every UUID and base64 blob, and no fixed list can cover it. Anything outside the list becomes an "unknown" placeholder, which is useless for security work where the interesting part is often a weird string.

**Subwords** sit in the middle. Common words get a single token, and rarer strings are built from smaller pieces that *are* in the vocabulary. `the` is one token; an unusual identifier like `getUserByIdUnsafe` might be four or five pieces; a random hex string falls back to short chunks. Nothing is ever unknown, and common text stays compact.

> [!CONCEPT] Token
> The unit of text a model reads and writes: a chunk that may be a whole word, part of a word, a punctuation mark, a single digit or even a single byte. Each token has an integer id. Your model's context length, speed and memory use are all measured in tokens, not characters or words.

## How a tokenizer turns text into ids

Qwen3.5's tokenizer, like most modern ones, works in a short pipeline:

```mermaid
flowchart LR
    A["Raw text"] --> B["Normalize<br/>(Unicode NFC)"]
    B --> C["Pre-tokenize<br/>(regex split into chunks)"]
    C --> D["Convert chunks<br/>to bytes"]
    D --> E["Apply BPE merges<br/>inside each chunk"]
    E --> F["Token ids"]
```

1. **Normalize.** The text is put into Unicode NFC form, so characters that can be written in two ways (for example an accented letter as one code point or as a letter plus a combining accent) end up identical.
2. **Pre-tokenize.** A regular expression splits the text into chunks: roughly words with their leading space, runs of punctuation, runs of whitespace, and single digits. Merges never cross chunk boundaries.
3. **Bytes.** Each chunk is converted to its UTF-8 bytes. This is what makes the tokenizer **byte-level**: the starting alphabet is the 256 possible byte values, so *any* input (Malayalam, emoji, binary junk) can be represented, and there's no "unknown" token.
4. **Merge.** Within each chunk, adjacent pieces are glued together using a learned list of merge rules, the BPE step explained next.
5. **Look up ids.** Each final piece is looked up in the vocabulary to get its id.

Here's what the pre-tokenizer does to a short string. Qwen3.5's split pattern keeps a word together with the single space before it, splits off punctuation, and puts **every digit in its own chunk**:

```text
Text:    Hello world 2026!
Chunks:  "Hello" | " world" | " " | "2" | "0" | "2" | "6" | "!"
```

Notice that the space before `2026` becomes its own chunk, because digits never absorb a leading space, and that `2026` can never become one token: each digit is always separate.

## Byte-pair encoding, step by step

**Byte-pair encoding (BPE)** is how the merge rules are learned. The algorithm is short enough to run by hand, and doing it once makes tokenizer behaviour much less mysterious.

> [!CONCEPT] Byte-pair encoding (BPE)
> A way to build a subword vocabulary. Start with single characters (or bytes). Count every adjacent pair in a training corpus, merge the most frequent pair into a new symbol, add it to the vocabulary, and repeat until the vocabulary reaches the target size. The ordered list of merges *is* the tokenizer. To tokenize new text, apply the same merges in the same order.

### A tiny training run

Imagine the whole training corpus is these words, with how often each appears:

| Word | Count |
|---|---|
| hack | 6 |
| hacker | 5 |
| hacking | 3 |
| track | 2 |
| tracker | 1 |

**Start:** every word is split into characters. The vocabulary is `a c e g h i k n r t`.

```text
h a c k (x6)    h a c k e r (x5)    h a c k i n g (x3)    t r a c k (x2)    t r a c k e r (x1)
```

**Merge 1.** Count adjacent pairs, weighted by word frequency. `a c` appears in every word: 6 + 5 + 3 + 2 + 1 = **17**. So does `c k`, also 17. Then `h a` = 6 + 5 + 3 = 14, `k e` = 5 + 1 = 6, `e r` = 6, and `t r`, `r a`, `k i`, `i n`, `n g` = 3 each. There's a tie at the top. Real implementations break ties with a fixed rule; we'll take `a c`. New token: **`ac`**.

```text
h ac k    h ac k e r    h ac k i n g    t r ac k    t r ac k e r
```

**Merge 2.** Now the top pair is `ac k` (17). New token: **`ack`**.

```text
h ack    h ack e r    h ack i n g    t r ack    t r ack e r
```

**Merge 3.** `h ack` appears 6 + 5 + 3 = 14 times, more than anything else. New token: **`hack`**.

```text
hack    hack e r    hack i n g    t r ack    t r ack e r
```

**Merge 4.** `e r` (5 + 1 = 6) beats `hack e` (5). New token: **`er`**.

**Merge 5.** `hack er` (5) wins. New token: **`hacker`**.

```text
hack    hacker    hack i n g    t r ack    t r ack er
```

After five merges, the vocabulary is the 10 starting characters plus `ac`, `ack`, `hack`, `er` and `hacker`. Frequent words became single tokens (`hack`, `hacker`), less frequent ones are built from pieces (`hack i n g`, `t r ack er`).

### Tokenizing a word it has never seen

Now tokenize `tracking`, which wasn't in the corpus. Start from characters and apply the merges **in the order they were learned**:

```text
start:          t r a c k i n g
merge a+c:      t r ac k i n g
merge ac+k:     t r ack i n g
merge h+ack:    (no 'h' before 'ack', nothing happens)
merge e+r:      (no 'e r', nothing happens)
merge hack+er:  (nothing)
result:         t | r | ack | i | n | g     -> 6 tokens
```

Unseen words still tokenize; they just cost more tokens. Scale this up and you have a real tokenizer: Qwen3.5's was trained on a huge multilingual corpus and kept merging until it had about a quarter of a million tokens. Common English words, common code keywords and frequent words in many languages ended up as single tokens; rare strings fall back to smaller pieces, down to single bytes.

> [!TIP]
> Want to build one yourself? Karpathy's `minbpe` repository (linked at the end) implements byte-level BPE in a few hundred readable lines of Python. It's a great weekend exercise once you've finished Part 1.

## Vocabulary and token ids

The **vocabulary** is the full list of tokens the model knows, each with an integer id. Qwen3.5's vocabulary is **248,320** entries. That's the padded size of the model's embedding table; the highest id actually used is a little lower. Qwen describes the larger vocabulary (up from about 150K in Qwen3) as part of expanding language coverage to 201 languages and dialects.

Ids are just positions in the list. The first entries are single printable characters, for example:

```text
"!" -> 0      "0" -> 15     "A" -> 32     "a" -> 64
```

Merged tokens come later, with higher ids. Byte-level tokenizers display some bytes with stand-in symbols when you inspect raw tokens: a leading space shows up as `Ġ` and a newline as `Ċ`. So the token for " world" appears as `Ġworld`. That's just display; the underlying token is the space plus the word.

Why does the vocabulary size matter?

- **Bigger vocabulary = fewer tokens per text**, especially for non-English languages and code. That means more text fits in the context and generation finishes sooner.
- **Bigger vocabulary = bigger model.** Each token id is a row in the input embedding table, and the output layer has to score every token at every step. For your model, each of those two tables is 248,320 × 4,096 ≈ 1.02 billion numbers, about 23% of all language-model parameters together ([Inside the transformer](#/foundations/transformer) breaks down the rest).

## Special tokens

Some tokens aren't text at all. **Special tokens** are reserved ids that mark structure: where a message starts and ends, where thinking happens, where an image goes. Here are the main ones in Qwen3.5's tokenizer:

| Token | Id | Purpose |
|---|---|---|
| `<\|endoftext\|>` | 248044 | End of a document; also used for padding |
| `<\|im_start\|>` | 248045 | Start of a chat message ("im" = instant message) |
| `<\|im_end\|>` | 248046 | End of a chat message; the end-of-sequence token for chat |
| `<think>` / `</think>` | 248068 / 248069 | Wrap the model's reasoning when thinking is on |
| `<tool_call>` / `</tool_call>` | 248058 / 248059 | Wrap a tool call the model wants to make |
| `<\|vision_start\|>`, `<\|image_pad\|>` ... | 248053 ... | Placeholders where image and video features are inserted |

When you chat in LM Studio, your messages are wrapped in these tokens using a **chat template** before the model sees them. Roughly:

```text
<|im_start|>user
What does HTTP 418 mean?<|im_end|>
<|im_start|>assistant
```

The model then generates the assistant's reply and emits `<|im_end|>` when it's done, which is how LM Studio knows to stop. Chat templates get a full lesson in [Base, instruct, thinking and chat templates](#/foundations/model-types).

> [!WARNING]
> Special tokens are a security boundary. If text a user (or a scraped web page) supplies can turn into a real `<|im_end|>` or `<|im_start|>` token, an attacker can fake the end of their message and start a new "system" or "assistant" turn, a direct cousin of SQL injection where data breaks out of its quoted context. Whether a literal `<|im_end|>` string in input becomes the special token depends on the software doing the tokenizing. You'll check one case in the lab and dig into the threat in [Threat-model your own AI setup](#/security/ai-threat-model).

## How different kinds of text tokenize

Try the estimator below. Type an English sentence, then a curl command, then a JWT or a line of Malayalam, and watch how the splits change. It approximates the pre-tokenization step (the regex split), not the full BPE result, so treat its counts as rough. The lab gives you exact numbers from the real tokenizer.

<div data-widget="tokens"></div>

### English prose

Common words are usually one token each, with the leading space included (` the`, ` request`, ` server`). Rare and long words split into a few pieces. English is what tokenizers are most optimized for, so it's the cheapest text you'll feed the model.

### Code

Code costs more tokens per character than prose. Keywords like `function` or `return` are often single tokens, but identifiers (`getUserByIdUnsafe`), operators (`!==`, `=>`), brackets and string contents split into many small pieces. Runs of spaces used for indentation are grouped together, but deep indentation still adds up over a long file.

### Other languages

How efficiently a language tokenizes depends on how much of it the tokenizer saw during training. Scripts like Malayalam and Devanagari (Hindi) use 3 bytes per character in UTF-8, so if few merges were learned for them, a sentence can cost several times more tokens than its English translation. Qwen3.5's split pattern keeps letters together with their combining marks (the vowel signs in Malayalam and Hindi are combining marks), so words aren't torn apart at every vowel sign before merging even starts. The lab measures the real cost.

### Numbers

Qwen3.5 splits **every digit into its own token**. `8443` is four tokens, `1234567` is seven. That has two consequences:

- Numbers are represented consistently: `1`, `12` and `123` always share the same building blocks, which helps the model learn digit-by-digit patterns. (Some other tokenizers merge digits into chunks of up to three, so the same number can split differently depending on context.)
- Long numbers are expensive, and arithmetic is still done by learned patterns, not by a calculator. Even with clean digit tokens, a 9B model will get multi-digit multiplication wrong. Give it a tool for math ([Structured output and tool calling](#/using/structured-output-tools)).

### Whitespace and punctuation

The space belongs to the *following* word: ` Paris` and `Paris` are different tokens with different ids. That has a subtle effect: if your prompt ends with a trailing space, the model has to continue from an unusual position (most words in its training data started *with* a space token), and output quality can dip slightly. Leave trailing spaces off prompts.

Pretty-printed JSON spends tokens on newlines and indentation. Minified JSON carries the same information in fewer tokens, which matters when you paste large API responses.

### Encoded and random strings

Base64, hex, hashes, JWTs, API keys and UUIDs are close to random, so they contain few patterns BPE could have learned. They break into many short pieces, often only a few characters per token. A single JWT can cost more tokens than a paragraph of English. The model also can't "see" inside these strings reliably: ask it to decode base64 in its head and it will often produce plausible garbage. Decode with a tool, then give the model the decoded text.

## Why this matters in practice

**Context budget.** Your model's native context is 262,144 tokens, but on 16 GB you'll often run 32K–64K ([Memory math](#/foundations/memory-math)). Tokens, not characters, fill that budget. A big minified JavaScript bundle or a long Burp response can eat tens of thousands of tokens on its own.

**Speed.** The model processes your prompt at a rate measured in tokens per second (and generates its answer the same way). On an Air, reading a prompt runs at very roughly 100–250 tokens per second ([How text gets generated](#/foundations/generation)), so a 20,000-token paste can mean over a minute before the first word of the answer. Trimming irrelevant headers and boilerplate before pasting is a real speed-up.

**Character-level tasks.** Ask "how many r's are in strawberry?" and the model doesn't see `s-t-r-a-w-b-e-r-r-y`. It sees something like two or three token ids (for example `str` + `aw` + `berry`, though the real split depends on the tokenizer). Counting letters means recalling the spelling of each token from memory, which models do poorly. The same goes for reversing strings, checking exact character positions in a payload, or spotting a one-character difference between two hashes.

> [!TIP]
> For character-level work, make the characters into tokens. Ask the model to first write the word letter by letter separated by spaces (`s t r a w b e r r y`), then count. Each letter now becomes its own token and the task gets much easier. Better still, for anything exact (lengths, offsets, encodings, hashes), have it write a one-line Python command and run that.

### Rules of thumb

These are widely used approximations from GPT-style tokenizers, useful for quick estimates. They are **not** measured on Qwen3.5. The lab measures yours.

| Content | Rough rule of thumb | Why |
|---|---|---|
| English prose | ~4 characters per token, ~0.75 words per token | Common words are single tokens |
| Source code | Noticeably fewer characters per token than prose | Symbols, identifiers, indentation |
| JSON, logs | Similar to code; pretty-printing costs extra | Punctuation and whitespace |
| Base64, hex, JWTs, keys | Often only 2–3 characters per token | Little learnable structure |
| Non-Latin scripts | Varies widely, often several times more than English | Multi-byte characters, fewer merges |
| Numbers (Qwen3.5) | Exactly 1 token per digit | Each digit is split off before merging |

## Lab: tokenize real text with the Qwen3.5 tokenizer

You'll download only the tokenizer files for `Qwen/Qwen3.5-9B` (about 13 MB, mostly `tokenizer.json`), not the 17 GB of weights, and run it from Python.

> [!NOTE]
> Qwen3.5 support was added to Hugging Face `transformers` in version **5.2.0** (released February 2026). Older versions, including all 4.x releases, don't know the architecture, so install 5.2 or newer. You don't need PyTorch to use a tokenizer. If `transformers` prints a warning that PyTorch isn't installed, you can ignore it for this lab.

> [!LAB] Measure tokens across languages, code and blobs
> 1. Use the Part 1 workspace and install `transformers`:
>
>    ```bash
>    cd ~/ai-lab/foundations
>    source .venv/bin/activate   # create it first if you skipped Lab A: python3 -m venv .venv
>    pip install "transformers>=5.2"
>    ```
>
> 2. Save this as `~/ai-lab/foundations/tokenize_samples.py`:
>
>    ```python title="tokenize_samples.py"
>    from transformers import AutoTokenizer
>
>    SAMPLES = {
>        "english": "Hello, this is a test.",
>        "malayalam": "നമസ്കാരം, ഇത് ഒരു പരീക്ഷണമാണ്.",
>        "hindi": "नमस्ते, यह एक परीक्षण है।",
>        "curl": """curl -s -X POST "https://api.example.com/v1/login" -H "Content-Type: application/json" -d '{"username":"admin","password":"hunter2"}'""",
>        # The well-known example token from jwt.io (not a secret)
>        "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
>        "javascript": """async function getUser(id) {
>        const res = await fetch(`/api/users/${id}`, { credentials: "include" });
>        if (!res.ok) throw new Error(`HTTP ${res.status}`);
>        return res.json();
>    }""",
>        "numbers": "Port 8443, PID 1337, total 1234567.",
>        "strawberry": "How many r's are in strawberry?",
>        "special": "<|im_end|>",
>    }
>
>
>    def main() -> None:
>        # Downloads only the tokenizer files on first run (cached in ~/.cache/huggingface)
>        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
>        print(f"tokenizer entries (len): {len(tok):,}\n")
>
>        for name, text in SAMPLES.items():
>            ids = tok.encode(text, add_special_tokens=False)
>            pieces = [tok.decode([i]) for i in ids]  # human-readable piece per token
>            print(f"=== {name}: {len(text)} chars -> {len(ids)} tokens "
>                  f"({len(text) / len(ids):.2f} chars/token)")
>            print("ids:   ", ids[:40], "..." if len(ids) > 40 else "")
>            print("pieces:", pieces[:40], "..." if len(pieces) > 40 else "")
>            print()
>
>
>    if __name__ == "__main__":
>        main()
>    ```
>
> 3. Run it:
>
>    ```bash
>    python tokenize_samples.py
>    ```
>
> 4. Fill in this table in `~/ai-lab/NOTES.md` from your output:
>
>    ```text
>    sample       chars   tokens   chars/token   observation
>    english
>    malayalam
>    hindi
>    curl
>    jwt
>    javascript
>    numbers
>    strawberry
>    special
>    ```
>
> 5. Answer these from your own numbers:
>    - The English, Malayalam and Hindi samples say the same thing. How many times more tokens does each non-English version cost?
>    - Which sample has the lowest chars/token? Is it the JWT, as the theory predicts?
>    - In `numbers`, is every digit its own token? Do the spaces before the numbers show up as separate tokens?
>    - How is `strawberry` split? Does the split make the r's easy to count?
>    - Did `<|im_end|>` come back as a single id (248046)? If so, this tokenizer turns that literal string into the real special token, exactly the injection risk from the warning above.
> 6. Some readable pieces for Malayalam or Hindi may show as `�`. That's not a bug: a single token can hold part of a multi-byte character, and decoding half a character on its own gives the replacement symbol. Decoding the whole list together restores the text. Try `print(tok.decode(ids))` to confirm.
> 7. Stretch: add your own samples. Try a real HTTP request copied from Burp, the same JSON pretty-printed and minified, and a SHA-256 hash. Which costs the most tokens per character?

## Recap

- Models read **tokens**: subword chunks with integer ids. Characters make sequences too long; words make the vocabulary unbounded.
- Qwen3.5's tokenizer normalizes to NFC, splits text with a regex (words with their leading space, punctuation runs, whitespace runs, single digits), converts to bytes, then applies **BPE** merges.
- BPE learns merges by repeatedly gluing the most frequent adjacent pair; tokenizing applies those merges in order. Unseen words still work, just with more tokens.
- The vocabulary has 248,320 entries. The embedding and output tables together hold about 2 billion of the model's parameters.
- **Special tokens** like `<|im_start|>`, `<|im_end|>` and `<think>` mark structure, and they're a security boundary.
- English is cheapest; code, other scripts and encoded blobs cost more. Every digit is a separate token.
- Tokens drive context budget and speed, and explain weakness at character-level tasks. Use tools for exact work.

## Check yourself

1. Why is a byte-level tokenizer never stuck with an "unknown" token?
   <details><summary>Answer</summary>

   Its starting alphabet is all 256 possible byte values, and any text (any language, emoji, binary-ish junk) is a sequence of bytes. In the worst case a string is tokenized byte by byte, so it can always be represented, just at a higher token count.

   </details>
2. In the BPE example, why did `hacker` become a single token while `tracking` ended up as six tokens?
   <details><summary>Answer</summary>

   `hacker` was frequent in the training corpus, so the pairs that build it (`a c`, `ac k`, `h ack`, `e r`, `hack er`) were among the most common and got merged. `tracking` never appeared. Applying the learned merges only produced `ack`, leaving `t`, `r`, `i`, `n` and `g` as single characters.

   </details>
3. A colleague pastes a 60 KB minified JavaScript bundle and a long JWT into the chat and is surprised by the slow response and a context-full warning. Explain using tokens.
   <details><summary>Answer</summary>

   Code and high-entropy strings tokenize poorly, with few characters per token, so 60 KB of minified JS plus a JWT can easily be tens of thousands of tokens. That fills a large part of the context window, and the prompt must be processed token by token at roughly 100–250 tokens/s on an Air, so there's a long wait before the first output token. Trim the input to the relevant parts and decode blobs with a tool first.

   </details>
4. Why is "how many r's are in strawberry?" hard for an LLM, and how can you make it easier?
   <details><summary>Answer</summary>

   The model sees a few token ids for the word, not its letters, so it must recall the spelling of each token from training, which it does unreliably. Ask it to spell the word out with spaces between letters first (so each letter becomes its own token) and then count, or have it write and run code for exact character work.

   </details>
5. Why might a literal `<|im_end|>` string inside a scraped web page be dangerous when that page is fed into your model?
   <details><summary>Answer</summary>

   If the tokenizer turns that literal string into the real end-of-message special token, the scraped content can fake the end of its message and start what looks like a new turn (for example a fake system or assistant message). That's a form of prompt injection, analogous to breaking out of a quoted string in SQL injection.

   </details>

## Go deeper

- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — two hours building a byte-level BPE tokenizer from scratch, with all the weird consequences explained.
- [karpathy/minbpe](https://github.com/karpathy/minbpe) — minimal, readable BPE implementations to study or extend.
- [Hugging Face LLM Course: Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) — the algorithm with a worked example and Python code.
- [Sennrich et al., Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — the paper that brought BPE to language models.
- [Qwen3.5-9B on Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B) — the model card and the `tokenizer.json` / `tokenizer_config.json` files you just used.
