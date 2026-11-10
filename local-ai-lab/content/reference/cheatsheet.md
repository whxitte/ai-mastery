# Command cheatsheet

The commands you'll use most, grouped by tool, with a one-line reminder of what each does. Each section links to the lesson that explains the flags properly. Flags change between versions, so when something fails, check `--help` first.

> [!GOALS]
> - Find the right command for a task without digging through lessons
> - Remember the key flags for LM Studio, llama.cpp, MLX, Hugging Face and the recon tools
> - Know which lesson explains each command in depth

> [!TIP]
> Every code block has a Copy button in its top-right corner.

## Your Mac

Details: [Your setup](#/start/your-setup), [Memory math](#/foundations/memory-math), [Performance](#/lmstudio/performance)

```bash
sysctl -n machdep.cpu.brand_string                  # chip name
system_profiler SPHardwareDataType                   # model, chip, cores, memory
sysctl hw.memsize                                    # memory in bytes
sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu   # performance / efficiency cores
memory_pressure                                      # memory headroom
sysctl vm.swapusage                                  # swap in use (should be ~0 while generating)
sysctl iogpu.wired_limit_mb                          # GPU memory cap (0 = macOS default)
sudo sysctl iogpu.wired_limit_mb=12288               # raise the cap to 12 GiB until reboot (advanced)
sudo sysctl iogpu.wired_limit_mb=0                   # back to the default
sudo powermetrics --samplers thermal -i 5000 -n 1    # thermal pressure
pmset -g therm                                       # thermal warnings
caffeinate -i <command>                              # keep the Mac awake while a job runs
```

## LM Studio CLI (`lms`)

Details: [Tour](#/lmstudio/tour), [Server and API](#/lmstudio/server-api)

```bash
~/.lmstudio/bin/lms bootstrap                        # put lms on your PATH (once)
lms ls                                               # downloaded models (--llm, --json)
lms ps                                               # loaded models
lms get <name>[@quant]                               # download (--mlx / --gguf)
lms import <file.gguf> --copy                        # import a local GGUF (moves it without --copy)
lms load <key> --context-length 32768 --gpu max      # load with options
lms load <key> --estimate-only --context-length 65536   # memory estimate, no load
lms unload --all                                     # free memory
lms server start                                     # API on http://localhost:1234
lms server status                                    # is it running?
lms server stop
lms chat <key> -p "question" --stats                 # one-shot chat in the terminal
lms log stream --source model --filter input,output  # see the exact rendered prompt
lms runtime ls                                       # installed llama.cpp / MLX runtimes
lms runtime update --all                             # update runtimes
```

## LM Studio API

Details: [Server and API](#/lmstudio/server-api), [Structured output](#/using/structured-output-tools)

```bash
curl http://localhost:1234/v1/models                 # health check + model ids

curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-9b-uncensored-hauhaucs-aggressive",
       "messages": [{"role": "user", "content": "Explain SSRF in two sentences."}],
       "temperature": 0.7, "top_p": 0.8}'

curl http://localhost:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "text-embedding-nomic-embed-text-v1.5", "input": "IDOR in /api/orders"}'
```

```python title="client.py"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
reply = client.chat.completions.create(
    model="qwen3.5-9b-uncensored-hauhaucs-aggressive",   # check yours with `lms ls`
    messages=[{"role": "user", "content": "Summarise this nmap output: ..."}],
    temperature=0.7,
)
print(reply.choices[0].message.content)
```

## Qwen3.5 sampling profiles

Details: [Sampling settings](#/lmstudio/sampling-settings), [Recommended configs](#/lmstudio/presets-for-you)

| Profile | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 0 | 1.5 |
| Thinking, precise coding | 0.6 | 0.95 | 20 | 0 | 0 |
| Non-thinking, general | 0.7 | 0.8 | 20 | 0 | 1.5 |
| Non-thinking, reasoning | 1.0 | 0.95 | 20 | 0 | 1.5 |

Keep repeat penalty at 1.0 (off). The model card has a second, slightly different non-thinking reasoning profile; the sampling lesson explains it.

## llama.cpp

Details: [Ecosystem](#/using/ecosystem), [Export to GGUF](#/finetune/export-gguf), [Steering](#/weights/steering)

```bash
brew install llama.cpp
find ~/.lmstudio/models -name "*.gguf" -exec ls -lh {} \;   # where LM Studio keeps models

llama-server -m "$MODEL" --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 -fa on --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0

#   -ctk q8_0 -ctv q8_0             quantize the KV cache
#   --mmproj <mmproj.gguf>          enable image input
#   --lora <adapter.gguf>           apply a LoRA adapter
#   --control-vector-scaled cv.gguf:0.6 --control-vector-layer-range 8 24   steering

curl http://127.0.0.1:8080/health

python convert_hf_to_gguf.py <hf_dir> --outtype bf16 --outfile model-bf16.gguf
llama-quantize model-bf16.gguf model-Q4_K_M.gguf Q4_K_M
llama-imatrix -m model-bf16.gguf -f calib.txt -o imatrix.gguf -ngl 99
python convert_lora_to_gguf.py <lora_dir> --base-model-id Qwen/Qwen3.5-9B --outtype f16 --outfile lora.gguf
llama-cvector-generator -m model.gguf --positive-file p.txt --negative-file n.txt --method mean -o cv.gguf
```

## Ollama

Details: [Ollama basics](#/ollama/basics), [Modelfiles](#/ollama/modelfile), [Server tuning](#/ollama/tuning), [API and security](#/ollama/api-integration)

```bash
brew install --cask ollama-app                       # menu-bar app (or: brew install ollama && brew services start ollama)
ollama -v                                            # version
curl -s localhost:11434/api/version                  # is the server up?

ollama pull qwen3.5:9b                               # download (add --runner llamacpp|mlx to choose the engine)
ollama pull hf.co/<user>/<repo>:<quant>              # pull any GGUF from Hugging Face
ollama list                                          # installed models
ollama show <model> --modelfile                      # the Modelfile behind a model (also --parameters, --template)
ollama run <model> --verbose                         # chat, with speed stats after each reply
ollama run <model> --think=false                     # turn thinking off (true / false / high)
ollama ps                                            # what is loaded: size, CPU/GPU split, context, runner, until
ollama stop <model>                                  # unload now
ollama rm <model>                                    # delete from disk

# inside `ollama run`:  /set parameter num_ctx 16384   /set think   /set nothink   /show info   /save <name>   /bye
```

```text title="Modelfile"
FROM /Users/you/.lmstudio/models/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/<file>.gguf
PARAMETER num_ctx 16384
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
SYSTEM "You are a precise security analyst. Say when you are unsure."
```

```bash
ollama create qwen-sec -f Modelfile                  # build the custom model
grep "server config" ~/.ollama/logs/server.log | tail -1     # what the server actually loaded

# server settings (menu-bar app: set, then restart the app)
launchctl setenv OLLAMA_CONTEXT_LENGTH 16384
launchctl setenv OLLAMA_KEEP_ALIVE 10m
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
ollama serve --help                                  # lists every environment variable

lsof -nP -iTCP:11434 -sTCP:LISTEN                    # must show 127.0.0.1, not *
```

The OpenAI-compatible endpoint is `http://localhost:11434/v1`. To point any script in this course at Ollama, change `BASE_URL` to that and the model name to your Ollama tag.

## Choosing and fetching models

Details: [Open-model landscape](#/models/landscape), [Right model for each job](#/models/use-cases), [Evals](#/models/evaluate)

```bash
lms get <search-term>[@q4_k_m]                       # LM Studio: download by search term and quant
lms get https://huggingface.co/<user>/<repo>         # LM Studio: from a Hugging Face URL
lms load <key> --identifier router --context-length 4096   # keep a small model resident under a name
lms load <key> --speculative-draft-simple --speculative-draft-model <draft-key>
ollama pull <tag>                                    # Ollama: from its library

llama-server -m <reranker.gguf> --reranking --port 8090    # serve a reranker

python model_shootout.py run --repeats 3             # same tasks, every candidate model
python model_shootout.py score                       # scorecard of pass rate, speed and memory
```

## GGUF inspection

Details: [Quantization](#/foundations/quantization), [Threat model](#/security/ai-threat-model)

```bash
pip install gguf
gguf-dump --no-tensors "$MODEL"                      # metadata only
gguf-dump --json --no-tensors "$MODEL"               # full strings, including the chat template
shasum -a 256 "$MODEL"                               # compare with the hash on Hugging Face
```

## Hugging Face

Details: [Inspect weights](#/weights/inspect-weights), [Hardening](#/infra/hardening)

```bash
pip install -U huggingface_hub "transformers>=5.3"
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
hf download <repo> <file> --revision <commit> --local-dir <dir>   # pin an exact commit
curl -sL https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json | python3 -m json.tool
export HF_HUB_OFFLINE=1                              # never touch the network
```

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")          # tokenizer files only
print(tok.tokenize("curl -s https://example.com"))
print(tok.apply_chat_template([{"role": "user", "content": "hi"}],
      tokenize=False, add_generation_prompt=True, enable_thinking=False))
```

## Fine-tuning with MLX

Details: [Fine-tune with MLX](#/finetune/mlx-mac)

```bash
pip install "mlx-lm[train]"
mlx_lm.convert --hf-path Qwen/Qwen3.5-2B -q --q-bits 4 --mlx-path models/qwen35-2b-4bit
mlx_lm.lora -c lora_config.yaml                      # train (rank / scale live in the YAML)
mlx_lm.lora --model <m> --data data --test --adapter-path adapters
mlx_lm.generate --model <m> --adapter-path adapters --prompt "..."
mlx_lm.fuse --model <m> --adapter-path adapters --save-path fused
```

## Recon toolchain

Details: [Toolchain](#/bugbounty/setup-tools), [Recon pipeline](#/bugbounty/recon-pipeline). Only against assets you're authorised to test.

```bash
brew install subfinder dnsx httpx katana nuclei ffuf jq
which -a httpx                                       # make sure it's ProjectDiscovery's, not Python's
nuclei -ut                                           # update templates

docker run -d --name juice-shop --rm -p 127.0.0.1:3000:3000 bkimminich/juice-shop   # practice target

echo localhost | httpx -p 3000 -sc -title -td -json -silent
katana -u http://localhost:3000 -d 2 -jc -rl 10 -jsonl -silent
nuclei -u http://localhost:3000 -id tech-detect -rl 10 -silent
ffuf -u http://localhost:3000/FUZZ -w wordlists/small.txt -rate 10 -ac -s
```

## Network and integrity checks

Details: [Hardening](#/infra/hardening)

```bash
lsof -nP -iTCP -sTCP:LISTEN                          # what is listening, and on which address
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
fdesetup status                                      # FileVault on?
shasum -a 256 -c SHA256SUMS                          # verify a model manifest
```

## Claude Code

Details: [Claude lab](#/weights/claude-lab)

```bash
curl -fsSL https://claude.ai/install.sh | bash       # or: brew install --cask claude-code
claude --version
claude doctor
cd ~/ai-lab/weights/astra-lab && claude              # start a session in the lab folder
```

## Recap

- `lms` controls LM Studio from the terminal; `lms log stream` shows exactly what the model receives.
- `llama-server` is the same engine without the app, with every flag exposed.
- `gguf-dump` and `shasum` are your inspection and integrity tools for model files.
- Recon tools run only against authorised scope, rate-limited, through your scope guard.

## Check yourself

1. Which command shows the exact prompt string, chat template included, that your model receives?
   <details><summary>Answer</summary>

   `lms log stream --source model --filter input`

   </details>
2. How do you check whether a context length will fit before loading?
   <details><summary>Answer</summary>

   `lms load <key> --estimate-only --context-length <N>`, or the memory calculator in [Memory math](#/foundations/memory-math).

   </details>
