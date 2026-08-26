# AAS Local Validation with NVIDIA NIM

This repository turns the workflow from `aas-local-validation.ipynb` into a local CLI. It keeps
the original `aicomp_sdk`, Gym environment, public `OptimalGuardrail`, replay, and scoring logic,
while replacing the two local GGUF model servers with NVIDIA NIM's OpenAI-compatible Chat
Completions API.

By default, the repository uses:

- `openai/gpt-oss-20b`
- `google/gemma-4-31b-it`

Gemma 4 31B is the closest model currently available on NIM to the Gemma 4 26B GGUF model used
in the notebook. You can pass any NIM model that supports function/tool calling with `--model`.

## Installation

Python 3.11 or 3.12 is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

The CLI automatically detects the downloaded SDK at `./input/aicomp_sdk`. This avoids
downloading the SDK again and installing heavyweight local-model dependencies such as `torch`
and `transformers` when only NIM is needed.

Set `NIM_API_KEY` in `.env`. You can create a key for the hosted endpoint at
[build.nvidia.com](https://build.nvidia.com/). Do not commit the `.env` file.

The SDK is already included under `input/` in this repository. If you move it elsewhere, set:

```dotenv
AAS_SDK_PATH=/absolute/path/to/ai-agent-security-multi-step-tool-attacks
```

That directory must contain `aicomp_sdk/`. The tested version is `aicomp-sdk==3.1.2`, matching
the competition SDK bundled with the repository when it was created.

If the `input` directory is unavailable, install the SDK from PyPI with
`pip install -e '.[sdk]'`.

## Running Validation

Place the submission you want to test in `attack.py`, then run:

```bash
aas-nim validate --attack attack.py
```

To run one model with a short time budget:

```bash
aas-nim validate \
  --attack attack.py \
  --model openai/gpt-oss-20b \
  --budget-s 60
```

To run multiple models, repeat `--model`:

```bash
aas-nim validate \
  --model openai/gpt-oss-20b \
  --model google/gemma-4-31b-it \
  --budget-s 300
```

The default is `--env gym`, which most closely matches the notebook and Kaggle public scorer.
Use `--env sandbox` for faster local iteration. The original notebook uses a 9,000-second budget
**per model**; this repository defaults to 60 seconds to prevent accidental API quota usage.
Increase `AAS_BUDGET_S` when needed.

## Self-Hosted NIM or Another OpenAI-Compatible Endpoint

```dotenv
NIM_BASE_URL=http://localhost:8000/v1
NIM_API_KEY=
NIM_MODELS=openai/gpt-oss-20b
```

For local endpoints, an empty API key is automatically replaced with the `not-used`
placeholder.

## Artifacts

Each validation run creates a directory based on the attack filename and start time:

```text
artifacts/runs/<filename>_<YYYYMMDD_HHMMSS_microseconds>/
```

For example, `--attack solution/live_fill_m47.py` may create
`artifacts/runs/live_fill_m47_20260812_143005_123456/`. Each model writes to its own `<model>/`
subdirectory within that run:

- `summary.json`: score, finding/cell counts, and timing data
- `findings.jsonl`: findings replayed and confirmed by the evaluator
- `transcript.log`: evaluator stdout and stderr
- `framework.jsonl`: framework event log
- `agent-debug.jsonl`: NIM requests and responses with the API key excluded

The run-level `summary.json` contains per-model scores and the `local_public_mean`, calculated in
the same way as the notebook. You can still use `--artifacts-dir` to specify a custom location.

## Attack Variants and Leaderboard

Store independent variants in [`solution/`](solution/README.md). The filename and timestamp keep
the results of each run separate automatically:

```bash
aas-nim validate \
  --attack solution/live_fill_m47.py \
  --budget-s 300
```

Build a leaderboard from all summaries under `artifacts/`:

```bash
aas-nim leaderboard
```

The command prints a table in the terminal and creates `artifacts/leaderboard.html`. To also
export CSV:

```bash
aas-nim leaderboard --csv artifacts/leaderboard.csv
```

## Validation on a Real GPU with GGUF and llama.cpp

`aas-nim validate` is only an approximation: the hosted NIM models and runtime differ
substantially from the actual GGUF models used by the original notebook on two T4 GPUs. Direct
measurements show NIM latency of roughly 1–4 seconds per candidate, while comments in the attack
variants report roughly 8.5–20 seconds per candidate on the real GGUF backend. As a result,
time-sensitive tuning such as `MARGIN_S` and `SPLIT_THRESHOLD_S` cannot be validated reliably
through NIM.

`aas-nim validate-gguf` runs the same two GGUF files used by the Kaggle notebook directly through
`llama-cpp-python` on your GPU. This provides more representative timing. Renting a powerful GPU,
such as an H100 80 GB, instead of using Kaggle's two free T4 GPUs can also make each validation
cycle much faster.

### Installation

```bash
pip install -e '.[gguf]'
# On a CUDA machine, reinstall the GPU wheel; the default CPU build is very slow:
pip install --upgrade --no-cache-dir llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Download both GGUF files in advance and set their paths with environment variables or CLI flags:

```dotenv
GPT_OSS_MODEL_PATH=/data/models/gpt-oss-20b-Q4_K_M.gguf
GEMMA_MODEL_PATH=/data/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
```

If no paths are set, `validate-gguf` downloads the models automatically from Hugging Face
(`unsloth/gpt-oss-20b-GGUF` and `unsloth/gemma-4-26B-A4B-it-GGUF`). Downloading them once and
setting explicit paths avoids repeated downloads.

### Usage

```bash
aas-nim validate-gguf --attack attack.py --budget-s 600
```

By default, both `gpt_oss` and `gemma` run sequentially: one model is loaded, evaluated, and
unloaded before the next model is loaded. This matches the real grader, which keeps only one
model in memory at a time. To select models individually, repeat `--model gpt_oss` or
`--model gemma` as needed.

The following llama.cpp performance options have defaults suitable for a fast, high-memory GPU
such as an H100. Larger batches and flash attention reduce per-candidate latency relative to the
SDK defaults:

```bash
aas-nim validate-gguf --attack attack.py --budget-s 9000 \
  --n-batch 2048 --n-ubatch 1024 \
  --n-gpu-layers -1 \
  --main-gpu 0
# Use --no-flash-attn if your llama-cpp-python build lacks a flash-attention kernel.
```

Artifacts use the same `artifacts/runs/<attack>_<timestamp>/<model>/` structure as NIM runs, so
`aas-nim leaderboard` can combine both validation methods.

## Important Differences

- This is a local approximation. NIM models, runtime, and sampling may differ from the GGUF
  models and public scorer, so local scores are not guaranteed to match the leaderboard.
- The adapter calls `/v1/chat/completions`, sends AAS tool schemas, and disables parallel tool
  calls because `AgentProtocol` accepts only one action per hop.
- Hosted NIM charges quota for every model call. AAS also replays candidates after the attack
  phase, so the request count may exceed the number of prompts in `attack.py`.

## Testing

```bash
pip install -e '.[dev]'
pytest -q
```

Unit tests do not require an API key.
