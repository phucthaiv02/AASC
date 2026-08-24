# AAS local validation với NVIDIA NIM

Repo này chuyển luồng trong `aas-local-validation.ipynb` thành CLI chạy local. Phần
`aicomp_sdk`, Gym environment, public `OptimalGuardrail`, replay và scoring vẫn giữ nguyên;
hai GGUF model server được thay bằng NVIDIA NIM API tương thích OpenAI Chat Completions.

Mặc định repo chạy:

- `openai/gpt-oss-20b`
- `google/gemma-4-31b-it`

Gemma 4 31B là model NIM hiện có gần nhất với Gemma 4 26B GGUF trong notebook. Có thể truyền
bất kỳ model NIM nào hỗ trợ function/tool calling bằng `--model`.

## Cài đặt

Yêu cầu Python 3.11 hoặc 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

CLI tự phát hiện SDK đã tải tại `./input/aicomp_sdk`, vì vậy không tải lại SDK và cũng không
cài các dependency model-local nặng như `torch`/`transformers` chỉ để gọi NIM.

Điền `NIM_API_KEY` trong `.env`. Key cho hosted endpoint có thể tạo tại
[build.nvidia.com](https://build.nvidia.com/). Không commit file `.env`.

Repo hiện tại đã có SDK ở `input/`. Nếu di chuyển SDK sang nơi khác, đặt:

```dotenv
AAS_SDK_PATH=/absolute/path/to/ai-agent-security-multi-step-tool-attacks
```

Thư mục đó phải chứa `aicomp_sdk/`. Version được kiểm thử là `aicomp-sdk==3.1.2`, trùng với
SDK đi kèm competition tại thời điểm tạo repo.

Nếu không có thư mục `input`, có thể cài SDK từ PyPI bằng `pip install -e '.[sdk]'`.

## Chạy validation

Đặt submission cần test tại `attack.py`, sau đó chạy:

```bash
aas-nim validate --attack attack.py
```

Chạy một model với budget ngắn:

```bash
aas-nim validate \
  --attack attack.py \
  --model openai/gpt-oss-20b \
  --budget-s 60
```

Chạy nhiều model bằng cách lặp `--model`:

```bash
aas-nim validate \
  --model openai/gpt-oss-20b \
  --model google/gemma-4-31b-it \
  --budget-s 300
```

Mặc định là `--env gym` để gần notebook/Kaggle public scorer. Có thể dùng `--env sandbox`
để iteration local nhẹ hơn. Notebook gốc dùng budget 9000 giây **cho mỗi model**; repo đặt
mặc định 60 giây để tránh vô tình dùng nhiều API quota. Tăng `AAS_BUDGET_S` khi cần.

## Dùng NIM self-hosted hoặc endpoint OpenAI-compatible khác

```dotenv
NIM_BASE_URL=http://localhost:8000/v1
NIM_API_KEY=
NIM_MODELS=openai/gpt-oss-20b
```

Với local endpoint, API key trống được tự chuyển thành placeholder `not-used`.

## Artifacts

Mỗi lần validate tự tạo một thư mục theo tên file attack và thời điểm bắt đầu:

```text
artifacts/runs/<tên-file>_<YYYYMMDD_HHMMSS_microseconds>/
```

Ví dụ, `--attack solution/live_fill_m47.py` có thể tạo
`artifacts/runs/live_fill_m47_20260812_143005_123456/`. Mỗi model ghi vào thư mục con
`<model>/` của run đó:

- `summary.json`: score, số finding/cell và timing
- `findings.jsonl`: các finding đã được evaluator replay và xác nhận
- `transcript.log`: stdout/stderr của evaluator
- `framework.jsonl`: event log của framework
- `agent-debug.jsonl`: request/response NIM đã loại trừ API key

`summary.json` ở thư mục run chứa score từng model và trung bình `local_public_mean` giống
notebook. Vẫn có thể truyền `--artifacts-dir` nếu cần chỉ định thủ công một vị trí khác.

## Attack variants và leaderboard

Lưu các biến thể độc lập trong [`solution/`](solution/README.md). Tên file và timestamp được dùng
tự động để giữ riêng kết quả của từng lần chạy:

```bash
aas-nim validate \
  --attack solution/live_fill_m47.py \
  --budget-s 300
```

Tạo bảng xếp hạng từ toàn bộ summary bên dưới `artifacts/`:

```bash
aas-nim leaderboard
```

Lệnh in bảng trong terminal và tạo `artifacts/leaderboard.html`. Có thể xuất thêm CSV:

```bash
aas-nim leaderboard --csv artifacts/leaderboard.csv
```

## Validate trên GPU thật (GGUF qua llama.cpp, ví dụ thuê H100)

`aas-nim validate` (NIM) chỉ là một xấp xỉ: model host trên NIM và timing của nó khác hẳn model
GGUF thật mà notebook gốc chạy trên 2xT4 (đã kiểm chứng trực tiếp — latency qua NIM chỉ
~1-4s/candidate trong khi comment của các attack variant ghi nhận backend GGUF thật ~8.5-20s/
candidate). Điều đó có nghĩa mọi tinh chỉnh phụ thuộc thời gian (`MARGIN_S`,
`SPLIT_THRESHOLD_S`, ...) không thể kiểm chứng đáng tin cậy qua đường NIM.

`aas-nim validate-gguf` chạy thẳng 2 file GGUF thật (cùng file notebook Kaggle dùng) qua
`llama-cpp-python`, trên GPU của bạn — dùng để có timing sát thực tế hơn, và nếu thuê một GPU
mạnh (vd. H100 80GB) thay vì 2xT4 miễn phí của Kaggle thì mỗi lượt validate xong nhanh hơn nhiều,
lặp lại được nhiều vòng hơn trong cùng thời gian.

### Cài đặt

```bash
pip install -e '.[gguf]'
# Trên máy có CUDA, cài lại bằng wheel GPU (bản CPU mặc định sẽ rất chậm):
pip install --upgrade --no-cache-dir llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Tải sẵn 2 file GGUF (giống notebook gốc) rồi trỏ đường dẫn qua biến môi trường hoặc cờ CLI:

```dotenv
GPT_OSS_MODEL_PATH=/data/models/gpt-oss-20b-Q4_K_M.gguf
GEMMA_MODEL_PATH=/data/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
```

Nếu không đặt sẵn, `validate-gguf` sẽ tự tải từ Hugging Face (`unsloth/gpt-oss-20b-GGUF`,
`unsloth/gemma-4-26B-A4B-it-GGUF`) — nên tải một lần rồi trỏ path để không tải lại mỗi lần chạy.

### Chạy

```bash
aas-nim validate-gguf --attack attack.py --budget-s 600
```

Mặc định chạy tuần tự cả `gpt_oss` và `gemma` (load model này, chạy xong, unload, load model kia
— giống cách grader thật chỉ giữ một model trong bộ nhớ tại một thời điểm). Có thể chọn riêng
từng model bằng `--model gpt_oss` / `--model gemma` (lặp lại để chạy cả hai).

Các cờ tinh chỉnh hiệu năng llama.cpp (mặc định đã đặt hợp lý cho một GPU nhanh, nhiều VRAM như
H100 — batch lớn hơn + flash attention giúp giảm thời gian mỗi candidate so với mặc định của SDK):

```bash
aas-nim validate-gguf --attack attack.py --budget-s 9000 \
  --n-batch 2048 --n-ubatch 1024 \
  --n-gpu-layers -1 \
  --main-gpu 0
# --no-flash-attn nếu bản llama-cpp-python của bạn không có kernel flash attention
```

Artifacts ghi cùng cấu trúc `artifacts/runs/<attack>_<timestamp>/<model>/` như đường NIM, nên
`aas-nim leaderboard` gộp chung được cả hai loại lượt chạy.

## Khác biệt cần lưu ý

- Đây là local approximation: NIM model/runtime có thể khác model GGUF và sampling của public
  scorer, nên score không đảm bảo giống leaderboard.
- Adapter dùng `/v1/chat/completions`, gửi tool schemas của AAS và tắt parallel tool calls vì
  `AgentProtocol` chỉ nhận một action ở mỗi hop.
- Hosted NIM tính quota theo mọi lần model được gọi. AAS còn replay các candidate sau giai đoạn
  attack, vì vậy số request có thể lớn hơn số prompt trong `attack.py`.

## Kiểm thử

```bash
pip install -e '.[dev]'
pytest -q
```

Không cần API key để chạy unit tests.
